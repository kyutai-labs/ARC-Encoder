import os
import torch
import json
import numpy as np
import random
from tqdm import tqdm, trange
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
    lim_toks: int = 128,
    temps: list[float] = [0, 0.5, 0.7, 1],
    max_bs: int = 4,
    output_path: str = "/lustre/scwpod02/client/kyutai-interns/hippop/experiments/results_eval",
):
    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {benchmark: {} for benchmark in benchmarks}

    pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
        llm_path=llm_path,
        ckpt_path="/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
        + run_name
        + "/checkpoints/checkpoint_"
        + str(ckpt).zfill(6),
        device=device,
        llm_name="Mistral7B",
        embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
        max_batch_size=max_bs,
    )
    print("Evaluating checkpoint", str(ckpt).zfill(6))

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
            for i in trange(0, len(questions), max_bs):
                generated_sequence, logprobs = pipeline.generate(
                    prompts=questions[i : i + max_bs],
                    text_conditioning=context[i : i + max_bs],
                    temperature=temp,
                    max_tokens=lim_toks,
                    truncate_double_space=True,
                    device=device,
                    device_generation=(
                        device
                        if torch.cuda.device_count() <= 1
                        else torch.device("cuda:1")
                    ),
                )

                generated_sequences.extend(generated_sequence)

            results[benchmark][str(temp)] = sum(
                [
                    metric_max_over_ground_truths(
                        METRIC_EVALUATION[benchmark], pred, gts
                    )
                    for pred, gts in zip(generated_sequences, answers)
                ]
            ) / len(questions)

    with open(os.path.join(output_path, run_name + "_results_eval.json"), "w") as f:
        json.dump(results, f)
    print(
        "Results saved at", os.path.join(output_path, run_name + "_results_eval.json")
    )


def evaluate_reconstruction_model(
    run_name: str, ckpt: int | None = None, pipeline: EmbedAugPipeline | None = None
):
    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"
    max_batch_size = 4

    if "yaml" in run_name:
        run_name = run_name.replace(".yaml", "")

    print("RUN NAME => ", run_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt is None and pipeline is None:

        # Get last checkpoint
        last_ckpt = sorted(
            os.listdir(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
                + run_name
                + "/checkpoints/"
            )
        )[-1]
        pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
            llm_path=llm_path,
            ckpt_path="/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
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
            ckpt_path="/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
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

    lim_toks = 128
    eval_data = "/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_valid.jsonl"
    valid_passage = []

    with open(eval_data, "r") as f:
        for i, line in enumerate(f):
            if i == n_passages:
                break
            valid_passage.append(
                pipeline.tokenizer.decode(
                    pipeline.tokenizer.encode(
                        json.loads(line)["text"].split("\n\n")[1], eos=True, bos=True
                    )[:lim_toks]
                )
            )

    temperatures = [0, 0.5, 0.7, 1]
    max_tokens = 128

    results_generation = {
        "0": {"valid": {"word_prompt": {}, "empty_prompt": {}}},
        "0.5": {"valid": {"word_prompt": {}, "empty_prompt": {}}},
        "0.7": {"valid": {"word_prompt": {}, "empty_prompt": {}}},
        "1": {"valid": {"word_prompt": {}, "empty_prompt": {}}},
    }

    n_passages = len(valid_passage)
    assert n_passages == len(valid_passage)

    for temp in temperatures:
        print(f"Temperature: {temp}")
        generated_sequences = []

        generated_sequences = []
        for i in range(0, n_passages, max_batch_size):
            passage = valid_passage[i : i + max_batch_size]
            generated_sequence, logprobs = pipeline.generate(
                prompts=[""] * len(passage),
                text_conditioning=passage,
                temperature=temp,
                max_tokens=max_tokens,
                truncate_double_space=False,
                device=device,
                device_generation=(
                    device if torch.cuda.device_count() <= 1 else torch.device("cuda:1")
                ),
            )

            generated_sequences.extend(generated_sequence)
        results_generation[str(temp)]["valid"]["empty_prompt"] = {
            "seq": generated_sequences
        }

    metrics = []
    for temp in results_generation.keys():
        # for split in results_generation[temp].keys():
        for prompt_type in results_generation[temp]["valid"].keys():
            if prompt_type == "empty_prompt":
                generated_sequences = results_generation[str(temp)]["valid"][
                    prompt_type
                ]["seq"]
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
                    f"CKPT: {ckpt}, Temperature: {temp}, Split: valid, Prompt Type: {prompt_type}, Overlap: {overlap}",
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
                        "split": "valid",
                        "prompt_type": prompt_type,
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
        "w",
    ) as f:
        json.dump(metrics, f)

    with open(
        "/home/hippolytepilchen/code/embed_llm/config/experiments/overall_results_w_output.json",
        "r",
    ) as f:
        overall_results = json.load(f)
    overall_results[run_name] = metrics
    with open(
        "/home/hippolytepilchen/code/embed_llm/config/experiments/overall_results_w_output.json",
        "w",
    ) as f:
        json.dump(overall_results, f)


if __name__ == "__main__":

    ensure_reproducibility(29)
    # evaluate_model("LT_FN_False_1_MLP_Latt_True_CA_2_CAL_every_True_DB", ckpt=20000, benchmarks = ['NQ'])


    run_names = [file_name for file_name in os.listdir('/lustre/scwpod02/client/kyutai-interns/hippop/tmp/') if 'LT_FN' in file_name]
    print("Number of runs:", len(run_names))
    for run_name in sorted(run_names):
        evaluate_reconstruction_model(run_name, ckpt=20000)
        # print("Memory:", torch.cuda.memory_allocated() / 1024**3)
        # print("Memory Cached:", torch.cuda.memory_reserved() / 1024**3)
        print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / 1024**3)
        # print("Reset memory ! ")
        torch.cuda.empty_cache()
