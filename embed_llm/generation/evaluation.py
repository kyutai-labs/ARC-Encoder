import os
import torch
import json
import numpy as np
import random
from tqdm import tqdm, trange
from pathlib import Path
import torch.distributed as dist
from embed_llm.models.augmented_model import EmbedAugPipeline
from embed_llm.generation.metrics import (
    word_overlap,
    get_bleu_score,
    get_meteor,
    get_em,
    get_f1_score,
    get_rougel_score,
    metric_max_over_ground_truths,
)


EVAL_DATA_PATH = {
    "NQ": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA/nq_data.jsonl",
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA/triviaqa_data.jsonl",
}

METRIC_EVALUATION = {"NQ": get_em, "TRIVIAQA": get_em}


def set_global_seed(seed=42):
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Additional PyTorch reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_reproducibility(seed=42):
    # Set seeds as before
    set_global_seed(seed)

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_model(
    run_name: str,
    benchmarks: list[str],
    ckpt: int | None = None,
    max_seq_len: int = 256,
    temps: list[float] = [0, 0.5, 0.7, 1],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
):
    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    results = {benchmark: {} for benchmark in benchmarks}
    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"

    # Get last checkpoint
    last_ckpt = sorted(
        [
            ckpt_name
            for ckpt_name in os.listdir(
                tmp_path
                + run_name
                + "/checkpoints/"
            )
            if (
                Path(
                    tmp_path
                    + run_name
                    + "/checkpoints/"
                )
                / ckpt_name
                / "params.json"
            ).exists()
        ]
    )[-1]

    pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
        llm_path=llm_path,
        ckpt_path=tmp_path
        + run_name
        + "/checkpoints/"
        + last_ckpt,
        device=device,
        llm_name="Mistral7B",
        embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
        max_batch_size=max_bs,
    )
    ckpt = int(last_ckpt.split("_")[-1])
    print("Evaluating checkpoint", ckpt)

    device_count = torch.cuda.device_count()
    other_device = torch.device("cuda:1") if device_count > 1 else device
    metrics = []
    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=len(benchmarks)
    ):
        eval_data = EVAL_DATA_PATH[benchmark]
        context = []
        questions = []
        answers = []

        with open(eval_data, "r") as f:
            for line in f:
                data = json.loads(line)
                questions.append(data["question"])

                if isinstance(data["answer"], str):
                    answers.append([data["answer"]])

                else:
                    answers.append(data["answer"])
                # Take the first ranked retrieved passage
                context.append(data["passages"][0])
        print("Evaluation dataset loaded for", benchmark)
        for temp in temps:
            generated_sequences = []
            n_samples = len(questions) if n_samples is None else n_samples
            for i in trange(0, n_samples, max_bs):
                generated_sequence, logprobs = pipeline.generate(
                    prompt_pre_embed=['']*len(questions[i : i + max_bs]),
                    prompt_post_embed=questions[i : i + max_bs],
                    text_conditioning=context[i : i + max_bs],
                    temperature=temp,
                    max_tokens=max_seq_len,
                    truncate_double_space=True,
                    device=device,
                    device_generation=other_device,
                )

                generated_sequences.extend(generated_sequence)

            value = sum(
                [
                    metric_max_over_ground_truths(
                        METRIC_EVALUATION[benchmark], pred, gts
                    )
                    for pred, gts in zip(generated_sequences, answers)
                ]
            ) / len(questions)

            metrics.append(
                {
                    "CKPT": ckpt,
                    "temp": temp,
                    "n_samples": n_samples,
                    benchmark: value,
                }
            )

    with open(
        "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
        + run_name
        + "/results_generation.json",
        "a",
    ) as f:
        json.dump(results, f)

    with open(
        output_file,
        "r",
    ) as f:
        overall_results = json.load(f)

    if run_name not in overall_results:
        overall_results[run_name] = metrics
    else:
        overall_results[run_name].extend(metrics)
    with open(
        output_file,
        "w",
    ) as f:
        json.dump(overall_results, f)


def evaluate_reconstruction_model(
    run_name: str,
    ckpt: int | None = None,
    pipeline: EmbedAugPipeline | None = None,
    output_file: str = None,
    temperatures: list[float] = [0, 0.5, 0.7, 1],
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    tmp_path: str = None,
    eval_data_type: str = 'atlas',
):
    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    if eval_data_type == 'atlas':
        eval_data = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/wiki_passages_pretraining/valid_atlas_enwiki-dec2021_standard.jsonl"
    elif eval_data_type == 'standard_dump':
        eval_data = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_valid.jsonl'
    else:
        raise ValueError("Invalid eval_data_type")
        
        
    print("RUN NAME => ", run_name)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"

    if ckpt is None and pipeline is None:

        # Get last checkpoint
        last_ckpt = sorted(
            [
                ckpt_name
                for ckpt_name in os.listdir(
                    tmp_path
                    + run_name
                    + "/checkpoints/"
                )
                if (
                    Path(
                        tmp_path
                        + run_name
                        + "/checkpoints/"
                    )
                    / ckpt_name
                    / "params.json"
                ).exists()
            ]
        )[-1]

        pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
            llm_path=llm_path,
            ckpt_path=tmp_path
            + run_name
            + "/checkpoints/"
            + last_ckpt,
            device=device,
            llm_name="Mistral7B",
            embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
            max_batch_size=max_batch_size,
        )
        ckpt = int(last_ckpt.split("_")[-1])
        print("Evaluating checkpoint", ckpt)

    elif pipeline is None:
        pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
            llm_path=llm_path,
            ckpt_path=tmp_path
            + run_name
            + "/checkpoints/checkpoint_"
            + str(ckpt).zfill(6),
            device=device,
            llm_name="Mistral7B",
            embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
            max_batch_size=max_batch_size,
        )
        print("Evaluating checkpoint", str(ckpt).zfill(6))

    else:
        assert ckpt is not None
        pipeline: EmbedAugPipeline = pipeline

    n_passages = 100

    lim_toks = max_seq_len
    valid_passage = []

    with open(eval_data, "r") as f:
        for i, line in enumerate(f):
            if i == n_passages:
                break

            if eval_data_type == 'standard_dump':
                valid_passage.append(
                    pipeline.tokenizer.decode(
                        pipeline.tokenizer.encode(
                            json.loads(line)["text"].split("\n\n")[1], eos=True, bos=True
                        )[:lim_toks]
                    )
                )
            elif eval_data_type == 'atlas': 
                valid_passage.append(
                    pipeline.tokenizer.decode(
                        pipeline.tokenizer.encode(
                            json.loads(line)["text"], eos=True, bos=True
                        )[:lim_toks]
                    )
                )
                
    

    max_tokens = lim_toks

    results_generation = {}

    n_passages = len(valid_passage)
    assert n_passages == len(valid_passage)

    device_count = torch.cuda.device_count()
    other_device = device if device_count <= 1 else torch.device("cuda:1")

    for temp in temperatures:
        print(f"Temperature: {temp}")
        generated_sequences = []

        generated_sequences = []
        for i in range(0, n_passages, max_batch_size):
            passage = valid_passage[i : i + max_batch_size]
            generated_sequence, logprobs = pipeline.generate(
                prompt_pre_embed = (['']*len(passage) if not pipeline.pipeline_args.w_prefix_prompt 
                else ['In other words, background: ']*len(passage)), 
                prompt_post_embed = (['']*len(passage) if not pipeline.pipeline_args.w_prefix_prompt 
                else [' is just another way of saying: ']*len(passage)),
                text_conditioning=passage,
                temperature=temp,
                max_tokens=max_tokens,
                truncate_double_space=False,
                device=device,
                device_generation=other_device,
            )

            generated_sequences.extend(generated_sequence)
        results_generation[str(temp)] =  generated_sequences
        

    metrics = []
    for temp in results_generation.keys():
        # for split in results_generation[temp].keys():

            generated_sequences = results_generation[str(temp)]
            gt_passage = valid_passage  # train_passage if split == 'train' else valid_passage
            overlap = word_overlap(gt_passage, generated_sequences)
            bleu_score = get_bleu_score(gt_passage, generated_sequences)
            bleu_score_avg = get_bleu_score(
                gt_passage, generated_sequences, avg=True
            )
            meteor_score = get_meteor(gt_passage, generated_sequences)
            em = np.mean(
                np.array(
                    [
                        get_em(gt, pred)
                        for gt, pred in zip(gt_passage, generated_sequences)
                    ]
                )
            )

            print(
                f"CKPT: {ckpt}, Temperature: {temp}, Overlap: {overlap}",
                "Bleu Score:",
                bleu_score,
                "EM:",
                em,
                "Meteor:",
                meteor_score,
                "Bleu Score Avg:",
                bleu_score_avg,
            )
            metrics.append(
                {
                    "CKPT": ckpt,
                    "temp": temp,
                    "overlap": overlap,
                    "bleu_score": bleu_score,
                    "em": em,
                    "Meteor": meteor_score,
                    "Bleu Score Avg": bleu_score_avg,
                }
            )

    with open(
        "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
        + run_name
        + "/results_generation.json",
        "a",
    ) as f:
        json.dump(metrics, f)

    with open(
        output_file,
        "r",
    ) as f:
        overall_results = json.load(f)

    if run_name not in overall_results:
        overall_results[run_name] = metrics
    else:
        overall_results[run_name].extend(metrics)

    with open(
        output_file,
        "w",
    ) as f:
        json.dump(overall_results, f)


if __name__ == "__main__":

    ensure_reproducibility(29)
    output_file = "/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/pretraining.json"
    tmp_path = "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
    # tmp_path = '/lustre/scwpod02/client/kyutai-interns/hippop/tmp/experiments/'

    run_names = [
        # 'LT_FN_Truemean_3_MLP_8_TRUNC_True_CA_16_CAL_atend_True_DB'
        "pretrain_llm_trained_rec_singpassage_054f63f8",
        "pretrain_both_trained_cont_singpassage_17c38ada",
        "pretrain_llm_trained_cont_singpassage_5daaa6bc",
        "pretrain_llm_trained_rec_multipassage_054f63f8",
    ]

    max_seq_len = 128
    
    for i, run_name in enumerate(run_names):

        evaluate_reconstruction_model(run_name, output_file=output_file, temperatures = [0, 0.5, 0.7, 1], max_seq_len=max_seq_len, tmp_path = tmp_path, eval_data_type = 'standard_dump') # 'atlas','standard_dump'
        
        evaluate_model(
            run_name,
            ["NQ", "TRIVIAQA"],
            temps=[0, 0.5, 0.7],
            max_bs=4,
            output_file=output_file,
            n_samples=100,
            max_seq_len=max_seq_len,
            tmp_path = tmp_path
        )

        torch.cuda.empty_cache()
