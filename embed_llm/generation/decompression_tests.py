
import os
import torch
import json
import numpy as np
import pandas as pd
import random
from tqdm import tqdm, trange
import argparse
from pathlib import Path
import torch.nn.functional as F
from embed_llm.generation.utils import ensure_reproducibility
from embed_llm.models.mistral.generate import generate as mistral_generate
from embed_llm.models.augmented_model import EmbedAugPipeline, load_pipeline
from embed_llm.training.loss import compute_ce_loss_with_mask, compute_bpt_loss
from embed_llm.generation.evaluation import get_gpu_memory
from embed_llm.generation.metrics import (
    word_overlap,
    get_bleu_score,
    get_meteor,
    get_em,
    get_f1_score,
    metric_max_over_ground_truths,
    get_approx_em,
)



def evaluate_toydecompression(
    run_name: str,
    ckpt: int | None = None,
    max_seq_len: int = 256,
    temp: float = 0.0,
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    pipeline: EmbedAugPipeline | None = None,
    instruct_name: str = None,
    data: str = None,
    decompress_task: str = "reversed",
    random_prefix: bool = False,
):

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"
    
    data_path_ID = '/lustre/scwpod02/client/kyutai-interns/datasets/crawl_2/train_en_00_of_18.jsonl'
    data_path_eval_ID = '/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/crawl/eval_en_00_of_18.jsonl'
    data_path_less_ID = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_train.jsonl'
    data_path_minor_OOD = '/lustre/scwpod02/client/kyutai-interns/datasets/finemath/train.jsonl'
    data_very_OOD = 'random'
    data_esp_lang = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/eswiki-20220120.jsonl'
    
    if data == 'random':
        data_path = data_very_OOD
    elif data == 'ID':
        data_path = data_path_ID
    elif data == 'eval_ID':
        data_path = data_path_eval_ID
    elif data == 'less_ID':
        data_path = data_path_less_ID
    elif data == 'minor_OOD':
        data_path = data_path_minor_OOD
    elif data == 'esp_lang':
        data_path = data_esp_lang
    elif data == 'lit_lang':
        data_path = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/ltwiki-20220120.jsonl'
    elif data == 'code':
        data_path = '/lustre/scwpod02/client/kyutai-interns/datasets/thestack/shuf/python.jsonl'
    else:
        raise NotImplementedError(f"Data {data} not implemented")

    # Loading model
    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
    device_count = torch.cuda.device_count()
    other_device = torch.device("cuda:1") if device_count > 1 else device

    pipeline, ckpt = load_pipeline(
        run_name=run_name,
        tmp_path=tmp_path,
        llm_path=llm_path,
        device=device,
        max_bs=max_bs,
        pipeline=pipeline,
        mistral=False,
        instruct_name=instruct_name,
        ckpt=ckpt,
    )
    
    text_conditioning = []
    if Path(data_path).exists():
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if data != 'code':
                    sample = json.loads(line)['text']
                else:
                    sample = json.loads(line)['content']
                text_conditioning.append(sample)
                if i+1 == n_samples:
                    break
    elif data_path == 'random':
        for _ in range(n_samples):
            sample = pipeline.tokenizer.decode([random.randint(0,pipeline.tokenizer.n_words-1) for _ in range(max_seq_len)])
            text_conditioning.append(sample)     
    
    full_x = [
        pipeline.tokenizer.encode(text, bos=True, eos=True)[:max_seq_len]
        for text in text_conditioning
    ]
    
    llm = pipeline.model.llm.to(other_device)
    
    # Sequence loading with modifs depending on training. 
    if decompress_task == 'from_prefix_reconstruct':
        if not random_prefix:
            full_encoded_prompt_post = [toks[:5] for toks in full_x]
            ppl_tok_input = [toks[:-1][5:] for toks in full_x]
            ppl_tok_output = [toks[1:][5:] for toks in full_x] 
            gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
            gen_len = max_seq_len - 5
        else:
            full_encoded_prompt_post = []
            ppl_tok_input = []
            ppl_tok_output = []
            gt_text = []
            gen_lengths = []
            for toks in full_x:
                start_gen = random.randint(0, len(toks)-10)
                full_encoded_prompt_post.append(toks[start_gen:start_gen+5])
                ppl_tok_input.append(toks[start_gen+5:-1])
                ppl_tok_output.append(toks[start_gen+6:])
                gt_text.append(pipeline.tokenizer.decode(toks[start_gen+6:]))
                gen_lengths.append(len(toks[start_gen+6:]))
            gen_len = max(gen_lengths)
                
                
    elif decompress_task == 'middle_reconstruct':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input =  [toks[:-1][len(toks)//2:] for toks in full_x]
        ppl_tok_output =  [toks[1:][len(toks)//2:] for toks in full_x] 
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len//2
    elif decompress_task == 'reversed':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input = [toks[:-1][::-1] for toks in full_x]
        ppl_tok_output = [toks[1:][::-1] for toks in full_x] 
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len
    elif decompress_task == 'true_reversed':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input = [toks[::-1][:-1] for toks in full_x]
        ppl_tok_output = [toks[::-1][1:] for toks in full_x] 
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len
    elif decompress_task == 'one_over_two_reconstruction':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input = [toks[:-1][::2] for toks in full_x]
        ppl_tok_output = [toks[1:][::2] for toks in full_x] 
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len//2
    elif decompress_task == 'full_reconstruct':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input = [toks[:-1]for toks in full_x]
        ppl_tok_output = [toks[1:]for toks in full_x]
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len
    elif decompress_task == 'reconstruct_one_every_two':
        full_encoded_prompt_post = [[pipeline.tokenizer.bos_id] for _ in full_x]
        ppl_tok_input = [toks[:-1][::2] for toks in full_x]
        ppl_tok_output = [toks[1:][::2] for toks in full_x] 
        gt_text = [pipeline.tokenizer.decode(toks) for toks in ppl_tok_output] 
        gen_len = max_seq_len//2
    else:
        raise NotImplementedError(f"Decompression task {decompress_task} not implemented")



    ppl = 0  
    bpc_w_embedding = 0
    bpc_wo_embedding = 0
    generated_seq = []
    for i in trange(0,len(full_x),max_bs):
        x = full_x[i:i+max_bs]
        encoded_prompt_post = full_encoded_prompt_post[i:i+max_bs]
        seqlens = [len(tokens) for tokens in x]
        x = torch.tensor([int(toks) for l_tokens in x for toks in l_tokens]).to(device
        )

        with torch.no_grad():
            embeddings = pipeline.model.trainable_embedder(
                input_ids=x, embeddings=None, seqlens=seqlens
            )

        embeddings = pipeline.model.pooling_module(x=embeddings.to(pipeline.pipeline_args.param_dtype), seqlens=seqlens)
        embed_seqlens =  [1]*len(seqlens)
        embeddings = F.normalize(embeddings, p=2, dim=-1) 
        
        if pipeline.pipeline_args.mlp_project.type == "mlp":
            cat_embeddings = pipeline.model.mlp_project(
                embeddings.to(pipeline.pipeline_args.param_dtype)
            )
        else:
            cat_embeddings = pipeline.model.mlp_project(
                embeddings.to(pipeline.pipeline_args.param_dtype),
                seqlens=embed_seqlens,
            )
        embeddings = cat_embeddings if other_device is None else cat_embeddings.to(other_device)
    
      
            
        with torch.no_grad():
            input = torch.tensor(sum(ppl_tok_input[i:i+max_bs],[]), dtype = torch.long).to(other_device)
            logits_w_embeds = llm.forward(
                input.detach(),
                seqlens=[len(p) for p in ppl_tok_input[i:i+max_bs]],
                embeddings=embeddings,
                embed_seqlens=embed_seqlens,
                cat_embeddings=cat_embeddings,
                show_attention=False,
            )
            logits_wo_embeds = llm.forward(
                input.detach(),
                seqlens=[len(p) for p in ppl_tok_input[i:i+max_bs]],
                embeddings=None,
                embed_seqlens=None,
                cat_embeddings=None,
                show_attention=False,
            )
            batch_bpc_wo = 0
            batch_bpc = 0
            batch_ce = 0
            ind = 0
            for toks in ppl_tok_output[i:i+max_bs]:
                loss_in_bits = torch.sum(compute_bpt_loss(logits_w_embeds[ind:ind+len(toks),...], 
                                                          torch.tensor(toks).to(other_device), None)).item() 
                batch_bpc += loss_in_bits / len(pipeline.tokenizer.decode(toks))
                batch_ce += compute_ce_loss_with_mask(logits_wo_embeds[ind:ind+len(toks),...],
                                                        torch.tensor(toks).to(other_device), None).item() 
                batch_bpc_wo += torch.sum(compute_bpt_loss(logits_wo_embeds[ind:ind+len(toks),...],
                                                            torch.tensor(toks).to(other_device), None)).item() / len(pipeline.tokenizer.decode(toks))
                ind += len(toks)
                            

            bpc_w_embedding += batch_bpc/len(ppl_tok_output[i:i+max_bs])
            bpc_wo_embedding += batch_bpc_wo/len(ppl_tok_output[i:i+max_bs])
            ppl += 2**(batch_ce/len(ppl_tok_output[i:i+max_bs]))
        with torch.no_grad():
            generated_tokens = mistral_generate(
                prompt_pre_embed=[[]*len(text_conditioning)],
                prompt_post_embed=encoded_prompt_post,
                embeddings=embeddings,
                model = llm,
                max_tokens=gen_len,
                temperature=temp,
                chunk_size=None,
                eos_id=pipeline.tokenizer.eos_id,
                embed_seqlens=embed_seqlens,
                cat_embeddings= cat_embeddings 
            )

        produced_text = [
            pipeline.tokenizer.decode(generated_tokens[l])
            for l in range(len(generated_tokens))
        ]
        generated_seq.extend(produced_text)

    print('Ground_truth:', gt_text[0])
    
    print('Generated:', generated_seq[0])
    bleu_score = get_bleu_score(gt_text, generated_seq)
    bleu_score_avg = get_bleu_score( gt_text, generated_seq,avg=True)
  
    metrics = {
        'ppl': ppl/len(range(0,len(full_x),max_bs)),
        'bpc_wo_embed': bpc_wo_embedding/len(range(0,len(full_x),max_bs)),
        'bpc_w_embed': bpc_w_embedding/len(range(0,len(full_x),max_bs)),
        'temp': temp,
        'bleu': bleu_score, 
        'bleu_avg': bleu_score_avg,
        "decompression_task": decompress_task,
        "data": data,
        'run_name': run_name,
        'instruct_name': instruct_name if instruct_name is not None else 'None',
        'n_samples': n_samples,
        'max_seq_len': max_seq_len,
        'random_prefix': random_prefix,
    }

    
    with open(output_file, "a") as f:
        json.dump(metrics, f)
        f.write("\n")
        
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--out_file", type=str, default='/home/hippolytepilchen/code/embed_llm/results/toy_tests/decompression_tests_2.jsonl')
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument("--random_prefix", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--decompress_task", type=str, default='reversed')
    parser.add_argument("--temp", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    ensure_reproducibility(args.seed)
    
    if not Path(args.out_file).exists():
        with open(args.out_file, "w") as f:
            json.dump({}, f)
        
        
    # Full reconstruct:    ToyPretraining_LLM_False_Emb_True_MaxEmb_1_pure_reconstruct_16BS
    # Hybrid: Hybrid_LLM_False_Emb_True_MaxEmb_1_PNoEmbed_0.0_StartPoint_0.0_16BS
    
    # Toy tests
    # ToyDecompressingTests_LLM_FT_MaxEmb_1_rec_from_prefix
    # ToyDecompressingTests_LLM_FT_MaxEmb_1_reversed_2
    # ToyDecompressingTests_LLM_FT_MaxEmb_1_one_over_2
    # ToyDecompressingTests_LLM_FT_MaxEmb_1_middle_reconstruct
    

    
    evaluate_toydecompression(
        run_name = args.run_name,
        ckpt = args.ckpt,
        max_seq_len = args.max_seq_len,
        temp = args.temp,
        max_bs = args.bs,
        output_file = args.out_file,
        n_samples = args.n_passages,
        tmp_path = '/lustre/scwpod02/client/kyutai-interns/hippop/tmp/',
        pipeline=None,
        instruct_name = args.instruct_name,
        data = args.data,
        decompress_task = args.decompress_task,
        random_prefix = args.random_prefix
    )
        
    
    
    