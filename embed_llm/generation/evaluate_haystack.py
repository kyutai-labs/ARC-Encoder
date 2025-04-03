import torch
import json
from tqdm import tqdm
import pandas as pd
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
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import transformers
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


EVALUATOR_TEMPLATE = {
    "template": """You are an expert grader of student answers relative to a reference answer. \n 
            Your goal is to provide a numerical score between 0 and 10, where 0 indicates no similarity or correctness, and 10 indicates a perfect match according to the following criteria: \n 
            Score 1: The answer is completely unrelated to the reference.\n
            Score 3: The answer has minor relevance but does not align with the reference.\n
            Score 5: The answer has moderate relevance but contains inaccuracies.\n
            Score 7: The answer aligns with the reference but has minor omissions.\n
            Score 10: The answer is completely accurate and aligns perfectly with the reference.\n
            Only respond with a numberical score.\n
            Here is the student answer: \n --- --- --- \n {answer}
            Here is the reference answer: \n --- --- --- \n {reference}"""
}


# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


def get_viz(score_file_path: str):
    with open(score_file_path, "r") as f:
        scores = json.load(f)

    temps = list(scores.keys())
    data_scores = {t: [] for t in temps}
    context_lengths = list(scores[temps[0]].keys())
    sorted_context_lengths = [
        str(ctx_l) for ctx_l in sorted([int(cl) for cl in context_lengths])
    ]
    depth_percents = list(scores[temps[0]][context_lengths[0]].keys())
    sorted_depth_percents = [
        str(ctx_l) for ctx_l in sorted([int(dp) for dp in depth_percents])
    ]
    for i, temp in enumerate(temps):
        for j, context_length in enumerate(context_lengths):
            for k, depth_percent in enumerate(depth_percents):
                data_scores[temp].append(
                    {
                        "Document Depth": depth_percent,
                        "Context Length": context_length,
                        "Score": 1
                        if isinstance(scores[temp][context_length][depth_percent], str)
                        else scores[temp][context_length][depth_percent],
                    }
                )

    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    fig, ax = plt.subplots(3, 1, figsize=(15, 8))

    for i, temp in enumerate(temps):
        df = pd.DataFrame(data_scores[temp])
        pivot_table = pd.pivot_table(
            df,
            values="Score",
            index=["Document Depth", "Context Length"],
            aggfunc="mean",
        ).reset_index()
        pivot_table = pivot_table.pivot(
            index="Document Depth", columns="Context Length", values="Score"
        )
        pivot_table = pivot_table.reindex(sorted_depth_percents)
        pivot_table = pivot_table.reindex(sorted_context_lengths, axis=1)

        sns.heatmap(
            pivot_table,
            # annot=True,
            fmt="g",
            cmap=cmap,
            cbar_kws={"label": "Score"},
            ax=ax[i],
            vmin=1,
            vmax=10,
        )

        # More aesthetics
        ax[i].set_title(
            f"Pressure Testing Llama 70B Context with temperature {temp}"
        )  # Adds a title
        ax[i].set_xlabel("Token Limit")  # X-axis label
        ax[i].set_ylabel("Depth Percent")  # Y-axis label
        ax[i].xaxis.set_tick_params(
            rotation=45
        )  # Rotates the x-axis labels to prevent overlap
        ax[i].yaxis.set_tick_params(
            rotation=0
        )  # Ensures the y-axis labels are horizontal

    plt.tight_layout()  # Fits everything neatly into the figure area
    fig.savefig(score_file_path.replace("score.json", ".png"))
    plt.show()


def get_haystack_scores(
    result_file_path: str,
    needle: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    with open(result_file_path, "r") as f:
        results = json.load(f)
    scores = {}
    for temp in results:
        scores[temp] = {}
        for context_length in results[temp]:
            scores[temp][context_length] = {}
            for depth_percent in results[temp][context_length]:
                print(
                    f"Temp: {temp}, Context Length: {context_length}, Depth Percent: {depth_percent}"
                )
                if isinstance(results[temp][context_length][depth_percent], str):
                    generated = results[temp][context_length][depth_percent].split(
                        "\n"
                    )[0]
                elif isinstance(results[temp][context_length][depth_percent], list):
                    generated = results[temp][context_length][depth_percent][0].split(
                        "\n"
                    )[0]
                else:
                    print("Error", results[temp][context_length][depth_percent])
                    continue
                messages = [
                    {
                        "role": "system",
                        "content": """ You are an expert grader of student answers relative to a reference answer. \n 
                        Your goal is to provide a numerical score between 0 and 10, where 0 indicates no similarity or correctness, and 10 indicates a perfect match according to the following criteria: \n
                        Score 1: The answer is completely unrelated to the reference.\n
                        Score 3: The answer has minor relevance but does not align with the reference.\n
                        Score 5: The answer has moderate relevance but contains inaccuracies.\n
                        Score 7: The answer aligns with the reference but has minor omissions.\n
                        Score 10: The answer is completely accurate and aligns perfectly with the reference.\n""",
                    },
                    {
                        "role": "user",
                        "content": (
                            """Here is the student answer: \n --- --- --- \n {answer}
                        Here is the reference answer: \n --- --- --- \n {reference}"""
                        ).format(answer=generated, reference=needle),
                    },
                ]
                ouptputs = pipeline(messages, max_new_tokens=32)
                try:
                    score = re.findall(
                        r"\b\d+\b",
                        ouptputs[0]["generated_text"][-1]["content"].split("\n\n")[0],
                    )
                    if len(score) > 0:
                        scores[temp][context_length][depth_percent] = int(score[0])
                    else:
                        scores[temp][context_length][depth_percent] = ouptputs[0][
                            "generated_text"
                        ][-1]["content"].split("\n\n")[0]
                except Exception as e:
                    print(e)
                    scores[temp][context_length][depth_percent] = ouptputs[0][
                        "generated_text"
                    ][-1]["content"].split("\n\n")[0]
    if os.path.exists(result_file_path.replace(".json", "_score.json")):
        with open(result_file_path.replace(".json", "_score.json"), "r") as f:
            old_scores = json.load(f)
            scores.update(old_scores)
    with open(result_file_path.replace(".json", "_score.json"), "w") as f:
        json.dump(scores, f)


def evaluate_haystack(
    run_name: str,
    ckpt: int | None = None,
    context_lengths: list[int] = [256, 512, 1024, 4096, 8192],
    document_depth_percents: list[int] = [10, 20, 40, 50, 60, 80, 100],
    temps: list[float] = [0, 0.5, 0.7, 1],
    tmp_path: str = "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/",
    pipeline: EmbedAugPipeline | Transformer | None = None,
    mistral: bool = False,
    max_multi_passage: int = 1,
    instruct_name: str = None,
    weak_needle: bool = False,
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    # Loading model
    if not is_torchrun():
        device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
        device_count = torch.cuda.device_count()
        other_device = torch.device("cuda:1") if device_count > 1 else device
    else:
        device = "cuda"
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

    if weak_needle:
        needle = "\n\nQuestion: What is the best thing to do in San Francisco\nAnswer: The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    else:
        needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"

    context_dict = generate_haystacks(
        needle=needle,
        tokenizer=mistral_tokenizer,
        context_lengths=context_lengths,
        document_depth_percents=document_depth_percents,
    )

    results = {}
    for temp in tqdm(temps):
        results[str(temp)] = {}
        for context_length in context_lengths:
            results[str(temp)][str(context_length)] = {}
            for depth_percent in document_depth_percents:
                eval_logger_info(
                    logger,
                    f"Temp: {temp}, CT Length: {context_length}, Depth prct: {depth_percent}",
                )
                ctx = context_dict[context_length][depth_percent]
                tokens = mistral_tokenizer.encode(ctx, bos=True, eos=False)

                if not mistral:
   

                    generated_sequence, embed_tokens, embeds = pipeline.generate(
                        batch_list_prompts=[
                            "\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer:"
                        ],
                        text_conditioning=ctx,
                        temperature=temp,
                        max_tokens=128,
                        truncate_line=False,
                        device=device,
                        device_generation=other_device,
                        give_n_tokens=True,
                        w_scores=None,
                    )

                    results[str(temp)][str(context_length)][str(depth_percent)] = (
                        generated_sequence
                    )
                else:
                    generated_sequence, logprobs = generate(
                        model=mistral_model,
                        encoded_prompts=[
                            tokens
                            + mistral_tokenizer.encode(
                                "\n\nQuestion: What is the best thing to do in San Francisco?\nAnswer:",
                                bos=False,
                                eos=False,
                            )
                        ],
                        max_tokens=128,
                        temperature=temp,
                        eos_id=mistral_tokenizer.eos_id,
                    )

                    results[str(temp)][str(context_length)][str(depth_percent)] = (
                        mistral_tokenizer.decode(generated_sequence[0])
                    )

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if run_name is not None:
            with open(
                "/home/hippolytepilchen/code/embed_llm/results/haystack/"
                + (run_name + "_weak" if weak_needle else run_name)
                + ".json",
                "r",
            ) as f:
                old_results = json.load(f)
                results.update(old_results)
            with open(
                "/home/hippolytepilchen/code/embed_llm/results/haystack/"
                + (run_name + "_weak" if weak_needle else run_name)
                + ".json",
                "w",
            ) as f:
                json.dump(results, f)
        elif instruct_name is not None:
            with open(
                "/home/hippolytepilchen/code/embed_llm/results/haystack/"
                + (instruct_name + "_weak" if weak_needle else instruct_name)
                + ".json",
                "r",
            ) as f:
                old_results = json.load(f)
                results.update(old_results)
            with open(
                "/home/hippolytepilchen/code/embed_llm/results/haystack/"
                + (instruct_name + "_weak" if weak_needle else instruct_name)
                + ".json",
                "w",
            ) as f:
                json.dump(results, f)
        elif mistral:
            if weak_needle:
                with open(
                    "/home/hippolytepilchen/code/embed_llm/results/haystack"
                    + "/mistral_weak.json",
                    "r",
                ) as f:
                    old_results = json.load(f)
                    results.update(old_results)
                with open(
                    "/home/hippolytepilchen/code/embed_llm/results/haystack"
                    + "/mistral_weak.json",
                    "w",
                ) as f:
                    json.dump(results, f)
            elif not weak_needle:
                with open(
                    "/home/hippolytepilchen/code/embed_llm/results/haystack"
                    + "/mistral.json",
                    "r",
                ) as f:
                    old_results = json.load(f)
                    results.update(old_results)
                with open(
                    "/home/hippolytepilchen/code/embed_llm/results/haystack"
                    + "/mistral.json",
                    "w",
                ) as f:
                    json.dump(results, f)
            else:
                raise ValueError("Need to specify run_name or instruct_name")
        else:
            raise ValueError("Need to specify run_name or instruct_name")
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
    parser.add_argument("--temps", nargs="+", type=float, default=[0, 0.5, 0.7])
    parser.add_argument(
        "--context_lengths", nargs="+", type=int, default=[256, 512, 1024, 4096, 8192]
    )
    parser.add_argument(
        "--document_depth_percents",
        nargs="+",
        type=int,
        default=[10, 20, 40, 50, 60, 80, 100],
    )
    parser.add_argument("--weak_needle", action="store_true")
    parser.add_argument("--get_scores", action="store_true")
    parser.add_argument("--rslt_file", type=str, default=None)
    parser.add_argument("--viz_path", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    args = arg_parser()

    if args.get_scores:
        get_haystack_scores(
            result_file_path=args.rslt_file,
            needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
            model_id="meta-llama/Llama-3.3-70B-Instruct",
        )

    elif args.viz_path is not None:
        score_files = [file for file in os.listdir(args.viz_path) if "score" in file]
        for file in score_files:
            get_viz(Path(args.viz_path) + file)
    else:
        evaluate_haystack(
            run_name=args.run_name,
            ckpt=args.ckpt,
            context_lengths=args.context_lengths,
            document_depth_percents=args.document_depth_percents,
            temps=args.temps,
            mistral=args.mistral,
            instruct_name=args.instruct_name,
            max_multi_passage=args.multi_passages,
            weak_needle=args.weak_needle,
        )
