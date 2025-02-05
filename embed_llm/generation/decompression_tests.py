
import os
import torch
import json
import numpy as np
import pandas as pd
import random
from tqdm import tqdm, trange
import argparse
import logging
import subprocess as sp
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from nltk.tokenize import sent_tokenize
from embed_llm.generation.utils import eval_logger_info
from embed_llm.models.augmented_model import EmbedAugPipeline
from embed_llm.models.utils import is_torchrun
from embed_llm.monitoring.utils import set_logger
from embed_llm.generation.metrics import (
    word_overlap,
    get_bleu_score,
    get_meteor,
    get_em,
    get_f1_score,
    metric_max_over_ground_truths,
    get_approx_em,
)






#Must have a params json for pipeline
llm_path = '/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B'

# run_name = 'Hybrid_LLM_True_Emb_True_MaxEmb_1_PNoEmbed_0.01_StartPoint_0.0_16BS
# run_name = 'Hybrid_LLM_False_Emb_False_MaxEmb_1_PNoEmbed_0.0_StartPoint_0.8_16BS'
# run_name = 'Hybrid_LLM_False_Emb_True_MaxEmb_1_PNoEmbed_0.0_StartPoint_0.8_16BS'
run_name = 'ToyPretraining_LLM_False_Emb_True_MaxEmb_1_pure_reconstruct_16BS'
run_name = 'Hybrid_LLM_False_Emb_True_MaxEmb_1_PNoEmbed_0.0_StartPoint_0.5_16BS' # model pourri
last_ckpt = '030000' # '008500' #'010000' 



device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.device_count() > 1:
    device = torch.device('cuda:0')
    print(f'Using {device} for loading')

w_embeds = True
max_batch_size = 4

instruct_ckpt = None # "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/ToyDecompressingTests_LLM_FT_MaxEmb_1_reversed/checkpoints/checkpoint_005000"


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--eval_reconstruction", action="store_true")
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--wo_embeds", action="store_false")
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--reconstruct_seq_len", type=int, default=256)
    parser.add_argument("--reconstruct_npassages", type=int, default=500)
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument("--colbert", action="store_true")
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--prompt_before_embed", action="store_true")
    parser.add_argument("--split_to_multipassage", action="store_true")
    parser.add_argument("--seed", type=float, default=0.42)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()