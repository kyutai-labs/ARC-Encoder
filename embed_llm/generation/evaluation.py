from torcheval.metrics import BLEUScore
from nltk.translate import meteor_score
import re
import os
import string
import torch
import json
import numpy as np
import random
from embed_llm.models.augmented_model import EmbedAugPipeline
import nltk

nltk.download("wordnet")


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


def word_overlap(ground_truth: list[str] | str, predicted: list[str] | str) -> float:
    if isinstance(ground_truth, str) and isinstance(predicted, str):
        ground_truth = set(ground_truth.split(" "))
        predicted = set(predicted.split(" "))
        assert len(ground_truth) > 0, "Ground truth set is empty"
        return len(ground_truth.intersection(predicted)) / len(ground_truth)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        avg_word_overlap = 0
        n_words = 0
        for gt_text, pred_text in zip(ground_truth, predicted):
            gt_text = set(gt_text.split(" "))
            pred_text = set(pred_text.split(" "))
            assert len(gt_text) > 0, "Ground truth set is empty"
            n_words += len(gt_text)
            avg_word_overlap += len(gt_text.intersection(pred_text))
        return avg_word_overlap / n_words


def get_bleu_score(
    ground_truth: list[str] | str, predicted: list[str] | str, avg: bool = False
) -> float:
    if not avg:
        metric = BLEUScore(n_gram=4)
        if isinstance(ground_truth, str) and isinstance(predicted, str):
            assert len(ground_truth) > 0, "Ground truth set is empty"
            metric.update(predicted, [ground_truth])
            return metric.compute().item()
        elif isinstance(ground_truth, list) and isinstance(predicted, list):
            for gt_text, pred_text in zip(ground_truth, predicted):
                assert len(gt_text) > 0, "Ground truth set is empty"
                try:
                    metric.update(pred_text, [gt_text])
                except:
                    print(
                        "Error with update:",
                        "\nGround-Truth: ",
                        gt_text,
                        "\nPred: ",
                        pred_text,
                    )
            return metric.compute().item()
    else:
        metrics = [BLEUScore(n_gram=i) for i in range(1, 5)]
        if isinstance(ground_truth, str) and isinstance(predicted, str):
            assert len(ground_truth) > 0, "Ground truth set is empty"
            for metric in metrics:
                metric.update(predicted, [ground_truth])
            result = np.array([metric.compute().item() for metric in metrics])
            return result.mean()
        elif isinstance(ground_truth, list) and isinstance(predicted, list):
            for gt_text, pred_text in zip(ground_truth, predicted):
                assert len(gt_text) > 0, "Ground truth set is empty"
                try:
                    for metric in metrics:
                        metric.update(pred_text, [gt_text])
                except:
                    print(
                        "Error with update:",
                        "\nGround-Truth: ",
                        gt_text,
                        "\nPred: ",
                        pred_text,
                    )
            result = np.array([metric.compute().item() for metric in metrics])
            return result.mean()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_accuracy(pred: str, ground_truth: str) -> int:
    return int(ground_truth == pred)


def get_em(pred: str, ground_truth: str) -> int:
    return int(normalize_answer(ground_truth) == normalize_answer(pred))


def get_meteor(ground_truth: list[str] | str, predicted: list[str] | str) -> float:

    if isinstance(ground_truth, str) and isinstance(predicted, str):
        assert len(ground_truth) > 0, "Ground truth set is empty"
        l_ground_truth = ground_truth.split(" ")
        l_predicted = predicted.split(" ")
        return meteor_score.single_meteor_score(l_ground_truth, l_predicted)
    elif isinstance(ground_truth, list) and isinstance(predicted, list):
        meteor_avg_score = 0
        for gt_text, pred_text in zip(ground_truth, predicted):
            assert len(gt_text) > 0, "Ground truth set is empty"
            l_ground_truth = gt_text.split(" ")
            l_predicted = pred_text.split(" ")
            meteor_avg_score += meteor_score.single_meteor_score(
                l_ground_truth, l_predicted
            )
        return meteor_avg_score / len(ground_truth)


def evaluate_model(
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
    # train_data = '/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_train.jsonl'
    # train_passage = []
    valid_passage = []

    # with open(train_data, 'r') as f:
    #     for i, line in enumerate(f):
    #         if i == n_passages:
    #             break
    #         train_passage.append(pipeline.tokenizer.decode(pipeline.tokenizer.encode(json.loads(line)['text'].split('\n\n')[1], eos = True, bos = True)[:lim_toks]))

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

    # results_generation = {'0':{'train': {'word_prompt':{}, 'empty_prompt':{}}, 'valid': {'word_prompt':{}, 'empty_prompt':{}}},
    #                         '0.5':{'train': {'word_prompt':{}, 'empty_prompt':{}}, 'valid': {'word_prompt':{}, 'empty_prompt':{}}},
    #                         '0.7':{'train': {'word_prompt':{}, 'empty_prompt':{}}, 'valid': {'word_prompt':{}, 'empty_prompt':{}}},
    #                         '1':{'train': {'word_prompt':{}, 'empty_prompt':{}}, 'valid': {'word_prompt':{}, 'empty_prompt':{}}},
    #                         '1.5':{'train': {'word_prompt':{}, 'empty_prompt':{}}, 'valid': {'word_prompt':{}, 'empty_prompt':{}}}}
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

        # for i in range(0, n_passages, max_batch_size):
        #     passage = train_passage[i:i+max_batch_size]
        #     generated_sequence, logprobs = pipeline.generate(prompts = [text.split(' ')[0] for text in passage],
        #                                 text_conditioning = passage,
        #                                 temperature = temp,
        #                                 max_tokens = max_tokens,
        #                                 truncate_double_space = False,
        #                                 device = device,
        #                                 device_generation = device if torch.cuda.device_count() <= 1 else torch.device('cuda:1'))

        #     generated_sequences.extend(generated_sequence)
        # results_generation[str(temp)]['train']['word_prompt'] = {'seq':generated_sequences}
        # generated_sequences = []
        # for i in range(0, n_passages, max_batch_size):
        #     passage = train_passage[i:i+max_batch_size]
        #     generated_sequence, logprobs = pipeline.generate(prompts = [''] * len(passage),
        #                                 text_conditioning = passage,
        #                                 temperature = temp,
        #                                 max_tokens = max_tokens,
        #                                 truncate_double_space = False,
        #                                 device = device,
        #                                 device_generation = device if torch.cuda.device_count() <= 1 else torch.device('cuda:1'))

        #     generated_sequences.extend(generated_sequence)
        # results_generation[str(temp)]['train']['empty_prompt'] = {'seq':generated_sequences}

        # generated_sequences = []
        # for i in range(0, n_passages, max_batch_size):
        #     passage = valid_passage[i:i+max_batch_size]
        #     generated_sequence, logprobs = pipeline.generate(prompts = [text.split(' ')[0] for text in passage],
        #                                 text_conditioning = passage,
        #                                 temperature = temp,
        #                                 max_tokens = max_tokens,
        #                                 truncate_double_space = False,
        #                                 device = device,
        #                                 device_generation = device if torch.cuda.device_count() <= 1 else torch.device('cuda:1'))

        #     generated_sequences.extend(generated_sequence)
        # results_generation[str(temp)]['valid']['word_prompt'] = {'seq':generated_sequences}

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
                # elif prompt_type == "word_prompt":
                #     gt_passage = valid_passage  # train_passage if split == 'train' else valid_passage
                #     gt_passage = [" ".join(text.split(" ")[1:]) for text in gt_passage]
                #     overlap = word_overlap(gt_passage, generated_sequences)
                #     bleu_score = get_bleu_score(gt_passage, generated_sequences)
                #     bleu_score_avg = get_bleu_score(
                #         gt_passage, generated_sequences, avg=True
                #     )
                #     meteor_score = get_meteor(gt_passage, generated_sequences)
                #     em = np.mean(
                #         np.array(
                #             [
                #                 get_em(gt, pred)
                #                 for gt, pred in zip(gt_passage, generated_sequences)
                #             ]
                #         )
                #     )
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
        "/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/overall_results.json",
        "r",
    ) as f:
        overall_results = json.load(f)
    overall_results[run_name] = metrics
    with open(
        "/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/overall_results.json",
        "w",
    ) as f:
        json.dump(overall_results, f)


if __name__ == "__main__":

    ensure_reproducibility(29)

    # run_names = os.listdir(
    #     "/home/hippolytepilchen/code/embed_llm/config/experiments/mistral"
    # )
    # with open(
    #     "/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/overall_results.json",
    #     "r",
    # ) as f:
    #     overall_results = json.load(f)

    # for key in overall_results.keys():
    #     run_names.remove(key + ".yaml")

    # print(run_names)
    # print("Number of runs:", len(run_names))
    run_names = [
        "128_SL_FN_Truemean_0_MLP_8_TRUNC_True_CA_16_CAL_False_SKV_False_DB_old_gate_more_params"
    ]

    for run_name in run_names:
        evaluate_model(run_name, ckpt=9500)
        print("Memory:", torch.cuda.memory_allocated() / 1024**3)
        print("Memory Cached:", torch.cuda.memory_reserved() / 1024**3)
        print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / 1024**3)
        print("Reset memory ! ")
        torch.cuda.empty_cache()
