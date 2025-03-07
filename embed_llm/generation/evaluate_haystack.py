import torch
import json
from tqdm import tqdm, trange
import argparse
import logging
import subprocess as sp
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from embed_llm.models.augmented_model import EmbedAugPipeline, load_pipeline
from embed_llm.models.utils import is_torchrun
from embed_llm.generation.utils import eval_logger_info
from embed_llm.monitoring.utils import set_logger
from embed_llm.generation.llm_needle_haystack_creator import generate_haystacks

logger = logging.getLogger(__name__)

# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info




def evaluate_haystack(
    run_name: str,
    ckpt: int | None = None,
    context_lengths: list[int] = [256, 512, 1024, 4096, 8192],
    document_depth_percents: list[int] = [10, 20, 40, 50, 60, 80, 100],
    temps: list[float] = [0, 0.5, 0.7, 1],
    tmp_path: str = '/lustre/scwpod02/client/kyutai-interns/hippop/tmp/',
    pipeline: EmbedAugPipeline | Transformer | None = None,
    mistral: bool = False,
    max_multi_passage: int = 1,
    instruct_name: str = None,
    needle: str = '\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n',
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"
    
    # Loading model
    if not is_torchrun():
        device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
        device_count = torch.cuda.device_count()
        other_device = torch.device("cuda:1") if device_count > 1 else device
    else:
        device = 'cuda'
        other_device = None


    pipeline, ckpt = load_pipeline(
        run_name=run_name,
        tmp_path=tmp_path,
        llm_path=llm_path,
        device=device,
        max_bs=1,
        pipeline=pipeline,
        mistral=mistral,
        instruct_name=instruct_name,
        ckpt=ckpt,
    )

    if mistral:
        mistral_tokenizer = MistralTokenizer.from_file(
            "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/tokenizer.model.v3"
        ).instruct_tokenizer.tokenizer
        mistral_model = pipeline
    else:
        mistral_tokenizer = pipeline.tokenizer
        
    context_dict = generate_haystacks(needle=needle,
                                      tokenizer=mistral_tokenizer, 
                                      context_lengths=context_lengths, 
                                      document_depth_percents=document_depth_percents)
    

    results = {}
    for temp in tqdm(temps):
        results[str(temp)] = {}  
        for context_length in context_lengths:  
            results[str(temp)][str(context_length)] = {}
            for depth_percent in document_depth_percents:
                eval_logger_info(logger, f'Temp: {temp}, CT Length: {context_length}, Depth prct: {depth_percent}')
                ctx = context_dict[context_length][depth_percent]
                tokens = mistral_tokenizer.encode(ctx, bos=True, eos=False)
                
                if not mistral:
                    # TO modify
                    splitted_tokens = []
                    splitted_tokens = [tokens[i:i + len(tokens)//max_multi_passage] for i in range(0, len(tokens), len(tokens)//max_multi_passage)]
                    splitted_ctx = [mistral_tokenizer.decode(token) for token in splitted_tokens]
                
                    generated_sequence, embed_tokens, embeds = pipeline.generate(
                        prompt_pre_embed=[""], 
                        prompt_post_embed=['\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer:'],
                        text_conditioning=[splitted_ctx],
                        temperature=temp,
                        max_tokens=128,
                        truncate_line=False,
                        device=device,
                        device_generation=other_device,
                        give_n_tokens= True,
                        w_scores = None
                    )
                
                    results[str(temp)][str(context_length)][str(depth_percent)] = generated_sequence
                else:

            

                    generated_sequence, logprobs = generate(
                        model=mistral_model,
                        encoded_prompts=tokens + mistral_tokenizer.encode('\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer:', bos=False, eos=False),
                        max_tokens=128,
                        temperature=temp,
                        eos_id=mistral_tokenizer.eos_id,
                    )

         
                    results[str(temp)][str(context_length)][str(depth_percent)] = mistral_tokenizer.decode(generated_sequence[0])
                 

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if run_name is not None:
                with open(
                    "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
                    + run_name
                    + "/results_haystack.json",
                    "a",
                ) as f:
                    json.dump(results, f)
        elif instruct_name is not None:
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
                + instruct_name
                + "/results_haystack.json",
                "a",
            ) as f:
                json.dump(results, f)
    if mistral:
        return mistral_model

    return pipeline, ckpt


def arg_parser():
    parser = argparse.ArgumentParser()
    

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument('--temps', nargs='+', type=float, default=[0, 0.5, 0.7])
    parser.add_argument('--context_lengths', nargs='+', type=int, default=[256, 512, 1024, 4096, 8192])
    parser.add_argument('--document_depth_percents', nargs='+', type=int, default=[10, 20, 40, 50, 60, 80, 100])

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)
  
  

    args = arg_parser()
    evaluate_haystack(
        run_name=args.run_name,
        ckpt=args.ckpt,
        context_lengths=args.context_lengths,
        document_depth_percents=args.document_depth_percents,
        temps=args.temps,
        mistral=args.mistral,
        instruct_name=args.instruct_name,
        max_multi_passage=args.multi_passages,
        
    )