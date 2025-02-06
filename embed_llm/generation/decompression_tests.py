
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
from embed_llm.training.loss import compute_ce_loss_with_mask
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
):

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"


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
    if Path(data).exists():
        with open(data, "r") as f:
            for i, line in enumerate(f):
                sample = json.loads(f)['text']
                text_conditioning.append(sample)
                if i+1 == n_samples:
                    break
    elif data == 'random':
        for _ in range(n_samples):
            sample = pipeline.tokenizer.decode([random.choice(pipeline.tokenizer.vocab) for _ in range(max_seq_len)])
            text_conditioning.append(sample)     
    
    full_x = [
        pipeline.tokenizer.encode(text, bos=True, eos=True)
        for text in text_conditioning
    ]
    
    
    # Sequence loading with modifs depending on training. 
    if decompress_task == 'from_prefix_reconstruct':
        full_encoded_prompt_post = [toks[:5] for toks in full_x]
        gt_text = [pipeline.tokenizer.decode(toks[5:]) for toks in full_x]
        gen_len = max_seq_len - 5
    elif decompress_task == 'middle_reconstruct':
        gt_text = [pipeline.tokenizer.decode(toks[len(toks)//2:]) for toks in full_x]
        full_encoded_prompt_post = [[] for _ in full_x]
        gen_len = max_seq_len//2
    elif decompress_task == 'reversed':
        full_encoded_prompt_post = [[] for _ in full_x]
        gt_text = [pipeline.tokenizer.decode(toks[::-1]) for toks in full_x]
        gen_len = max_seq_len
    elif decompress_task == 'one_over_two_reconstruction':
        full_encoded_prompt_post = [[] for _ in full_x]
        gt_text = [pipeline.tokenizer.decode(toks[::2]) for toks in full_x]
        gen_len = max_seq_len//2
    else:
        raise NotImplementedError(f"Decompression task {decompress_task} not implemented")
    
    ppl = 0
    generated_seq = []
    for i in range(0,len(full_x),max_bs):
        
        x = full_x[i:i+max_bs]
        encoded_prompt_post = full_encoded_prompt_post[i:i+max_bs]
        
        seqlens = [len(tokens) for tokens in x]
        x = torch.from_numpy(np.array([el for el in x])).to(
            device
        )
        embeddings = pipeline.model.trainable_embedder(
            input_ids=x, embeddings=None, seqlens=seqlens
        )

        embeddings = pipeline.model.pooling_module(x=embeddings.to(pipeline.pipeline_args.param_dtype), seqlens=seqlens)
        embed_seqlens = [len(l_text) for l_text in text_conditioning]
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
        
        llm = pipeline.model.llm.to(other_device)

        input = [pipeline.tokenizer.encode(seq, bos = False, eos = False)[:-1] for seq in gt_text]
        target = [pipeline.tokenizer.encode(seq, bos = False, eos = False)[1:] for seq in gt_text]
        with torch.no_grad():
            logits = llm.forward(
                torch.tensor(sum(input,[]), device = llm.device, dtype = torch.long),
                seqlens=[len(p) for p in x],
                embeddings=None,
                embed_seqlens=None,
                cat_embeddings=None,
                show_attention=False,
            )
            ce = compute_ce_loss_with_mask(
                logits,
                torch.tensor(sum(target,[]), device = llm.device, dtype = torch.long),
            ).item()
            
            ppl += 2**ce
                            
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
            pipeline.tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]
        generated_seq.extend(produced_text)


    bleu_score = get_bleu_score(gt_text, generated_seq)
    bleu_score_avg = get_bleu_score( gt_text, generated_seq,avg=True)
  
    metrics = {
        'temp': temp,
        'ppl': ppl/len(range(0,len(full_x),max_bs)),
        "decompression_task": decompress_task,
        "data": data,
        'run_name': run_name,
        'bleu': bleu_score, 
        'bleu_avg': bleu_score_avg,
    }

    
    with open(output_file, "a") as f:
        json.dump(metrics + "\n", f)
        
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--out_file", type=str, default='/home/hippolytepilchen/code/embed_llm/results/toy_tests/decompression_tests.jsonl')
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument("--seed", type=float, default=0.42)
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
        
        
    tmp_path = '/lustre/scwpod02/client/kyutai-interns/hippop/tmp'
    data_path_ID = '/lustre/scwpod02/client/kyutai-interns/datasets/crawl_2/train_en_00_of_18.jsonl'
    data_path_eval_ID = '/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/crawl/eval_en_00_of_18.jsonl'
    data_path_less_ID = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_train.jsonl'
    data_path_minor_OOD = '/lustre/scwpod02/client/kyutai-interns/datasets/finemath/train.jsonl'
    data_very_OOD = 'random'
    
    if args.data == 'random':
        data = data_very_OOD
    elif args.data == 'ID':
        data = data_path_ID
    elif args.data == 'eval_ID':
        data = data_path_eval_ID
    elif args.data == 'less_ID':
        data = data_path_less_ID
    elif args.data == 'minor_OOD':
        data = data_path_minor_OOD
    else:
        raise NotImplementedError(f"Data {args.data} not implemented")
    
    evaluate_toydecompression(
        run_name = args.run_name,
        ckpt = args.ckpt,
        max_seq_len = args.max_seq_len,
        temp = args.temp,
        max_bs = args.bs,
        output_file = args.out_file,
        n_samples = args.n_passages,
        tmp_path = None,
        pipeline=None,
        instruct_name = args.instruct_name,
        data = data,
        decompress_task = args.decompress_task,
    )
        
    
    
    