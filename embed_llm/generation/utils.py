import warnings
import pandas as pd
import logging
import os
import random
import numpy as np
import torch
from embed_llm.models.utils.utils import is_torchrun
from embed_llm.models.utils.loading import (
    load_state_dict,
)
import safetensors.torch
from pathlib import Path

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


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


def eval_logger_info(logger, message: str) -> None:
    if not is_torchrun() or torch.distributed.get_rank() == 0:
        logger.info(message)


def format_results(results: dict, benchmark: str, icae: bool = False) -> pd.DataFrame:
    if (
        benchmark.lower() == "nq"
        or benchmark.lower() == "triviaqa"
        or benchmark.lower() == "hotpotqa"
        or benchmark.lower() == "squad"
    ):
        key_list = [
            "run_name",
            "ckpt",
            "EM Metric",
            "F1",
            "temp",
            "n_samples",
            "icl_examples",
            "context_in_examples",
            "context_w_query",
            "Prop_a_in_cont",
            "n_passages",
            "compress_ratio",
            "EM approx_Metric",
            "xRAG metric",
        ]
    elif benchmark.lower() == "factkg":
        key_list = [
            "run_name",
            "ckpt",
            "temp",
            "n_samples",
            "icl_examples",
            "context_in_examples",
            "Metric",
            "Prop_a_in_cont",
            "n_passages",
        ]
    elif benchmark.lower() == "traduction":
        key_list = [
            "run_name",
            "ckpt",
            "temp",
            "n_samples",
            "language",
            "Bleu",
            "compress_ratio",
            "fine_tuned",
        ]
    else:
        raise ValueError("Invalid benchmark")

    formated_results = pd.DataFrame(columns=key_list)

    for run_name in results.keys():
        for ckpt in results[run_name].keys():
            if benchmark.lower() == "traduction":
                for metric in [
                    "BLEU",
                ]:
                    if metric not in results[run_name][ckpt].keys():
                        continue
                    for temp in results[run_name][ckpt][metric].keys():
                        for result in results[run_name][ckpt][metric][temp]:
                            formated_results = pd.concat(
                                [
                                    formated_results,
                                    pd.DataFrame(
                                        {
                                            "run_name": run_name,
                                            "ckpt": int(ckpt),
                                            "temp": float(temp),
                                            "Metric": result["Metric"],
                                            "n_samples": result["n_samples"],
                                            "language": result["language"],
                                            "fine_tuned": result.get(
                                                "fine_tuned", True
                                            ),
                                            "compress_ratio": result.get(
                                                "compress_ratio", None
                                            ),
                                        },
                                        index=[0],
                                    ),
                                ]
                            )
                formated_results = (
                    formated_results.groupby(
                        [
                            "run_name",
                            "ckpt",
                            "temp",
                            "n_samples",
                            "language",
                            "fine_tuned",
                            "compress_ratio",
                        ]
                    )
                    .first()
                    .reset_index()
                )

            elif (
                benchmark.lower() == "nq"
                or benchmark.lower() == "triviaqa"
                or benchmark.lower() == "hotpotqa"
                or benchmark.lower() == "squad"
            ):
                for metric in ["EM", "F1"]:
                    if benchmark not in results[run_name][ckpt].keys():
                        continue
                    if metric not in results[run_name][ckpt][benchmark].keys():
                        continue
                    for temp in results[run_name][ckpt][benchmark][metric].keys():
                        for res in results[run_name][ckpt][benchmark][metric][temp]:
                            if metric == "EM":
                                formated_results = pd.concat(
                                    [
                                        formated_results,
                                        pd.DataFrame(
                                            {
                                                "run_name": run_name,
                                                "ckpt": int(ckpt),
                                                "temp": float(temp),
                                                "n_samples": res["n_samples"],
                                                "icl_examples": res["icl_examples"],
                                                "EM Metric": res.get("Metric", None),
                                                "compress_ratio": res.get(
                                                    "compress_ratio", None
                                                ),
                                                "compressed_icl": res.get(
                                                    "compressed_icl", False
                                                ),
                                                "context_in_examples": res[
                                                    "w_context_in_examples"
                                                ],
                                                "context_w_query": res.get(
                                                    "w_context_w_query", False
                                                ),
                                                "n_passages": res.get("n_passages", 1),
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "xRAG metric": res.get(
                                                    "xRAG metric", None
                                                ),
                                                "EM approx_Metric": res.get(
                                                    "approx_Metric", None
                                                ),
                                            },
                                            index=[0],
                                        ),
                                    ]
                                )
                            else:
                                df_res = pd.DataFrame(
                                    {
                                        "run_name": run_name,
                                        "ckpt": int(ckpt),
                                        "temp": float(temp),
                                        "n_samples": res["n_samples"],
                                        "icl_examples": res["icl_examples"],
                                        "F1": res.get("Metric", None),
                                        "compress_ratio": res.get(
                                            "compress_ratio", None
                                        ),
                                        "compressed_icl": res.get(
                                            "compressed_icl", False
                                        ),
                                        "context_in_examples": res[
                                            "w_context_in_examples"
                                        ],
                                        "context_w_query": res.get(
                                            "w_context_w_query", False
                                        ),
                                        "Prop_a_in_cont": res.get(
                                            "Prop context containing the answer",
                                            None,
                                        ),
                                        "n_passages": res.get("n_passages", 1),
                                    },
                                    index=[0],
                                )

                                formated_results = pd.concat([formated_results, df_res])

                if icae:
                    formated_results = (
                        formated_results.groupby(
                            [
                                "run_name",
                                "ckpt",
                                "temp",
                                "n_samples",
                                "icl_examples",
                                "context_in_examples",
                                "n_passages",
                                "compress_ratio",
                            ]
                        )
                        .first()
                        .reset_index(allow_duplicates=True)
                    )
                else:
                    formated_results = (
                        formated_results.groupby(
                            [
                                "run_name",
                                "ckpt",
                                "temp",
                                "n_samples",
                                "icl_examples",
                                "context_in_examples",
                                "n_passages",
                                "compressed_icl",
                                "compress_ratio",
                            ]
                        )
                        .first()
                        .reset_index(allow_duplicates=True)
                    )

    return formated_results


def DARE_merging(
    pretrain_path: str,
    fine_tune_paths: list,
    output_path: str,
    coeff: float,
    drop_rate: float = 0.3,
    seed: int = 42,
) -> None:
    """
    Language Models are Super Mario
    Merges the pre-trained model with fine-tuned models for DARE.

    Args:
        pretrain_path (str): Path to the pre-trained model.
        fine_tune_paths (list): List of paths to fine-tuned models.
        output_path (str): Path to save the merged model.
    """
    with open(Path(pretrain_path) / "params.json", "r") as f:
        params = f.read()
    Path(output_path + "checkpoints/checkpoint_000000/").mkdir(
        parents=True, exist_ok=True
    )
    with open(
        Path(output_path + "checkpoints/checkpoint_000000/") / "params.json", "w"
    ) as f:
        f.write(params)

    with open(Path(pretrain_path + "../../") / "args.yaml", "r") as f:
        params = f.read()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "args.yaml", "w") as f:
        f.write(params)

    pretrain_state_dict = load_state_dict(
        Path(pretrain_path) / "embedder", dtype=torch.float32
    )

    fine_tune_state_dicts = [
        load_state_dict(Path(path) / "embedder", dtype=torch.float32)
        for path in fine_tune_paths
    ]
    generator = torch.Generator().manual_seed(seed)
    new_state_dict = {}
    for k, v in pretrain_state_dict.items():
        delta = torch.zeros_like(v)
        for ft_state_dict in fine_tune_state_dicts:
            m = torch.bernoulli(torch.ones_like(v) * drop_rate, generator=generator).to(
                v.device
            )
            delta_param = (1 - m) * (ft_state_dict[k] - v) / (1 - drop_rate)
            delta = delta + delta_param
            if k in ft_state_dict.keys():
                delta = delta + ft_state_dict[k] - v
            else:
                raise ValueError(
                    f"Key {k} not found in fine-tuned state dicts. Ensure all fine-tuned models have the same architecture."
                )
        new_state_dict[k] = pretrain_state_dict[k] + coeff * delta

    if not Path(output_path + "checkpoints/checkpoint_000000/embedder/").exists():
        Path(output_path + "checkpoints/checkpoint_000000/embedder/").mkdir(
            parents=True, exist_ok=True
        )

    safetensors.torch.save_file(
        new_state_dict,
        Path(output_path + "checkpoints/checkpoint_000000/embedder")
        / "consolidated.safetensors",
    )

    if Path(pretrain_path + "/llm/decoder").exists():
        pretrain_state_dict = load_state_dict(
            Path(pretrain_path) / "llm/decoder", dtype=torch.float32
        )

        fine_tune_state_dicts = [
            load_state_dict(Path(path) / "llm/decoder", dtype=torch.float32)
            for path in fine_tune_paths
        ]

        if not Path(
            output_path + "checkpoints/checkpoint_000000/llm/decoder/"
        ).exists():
            Path(output_path + "checkpoints/checkpoint_000000/llm/decoder/").mkdir(
                parents=True, exist_ok=True
            )
        new_state_dict = {}
        for k, v in pretrain_state_dict.items():
            delta = torch.zeros_like(v)
            for ft_state_dict in fine_tune_state_dicts:
                m = torch.bernoulli(
                    torch.ones_like(v) * drop_rate, generator=generator
                ).to(v.device)
                delta_param = (1 - m) * (ft_state_dict[k] - v) / (1 - drop_rate)
                delta = delta + delta_param
                if k in ft_state_dict.keys():
                    delta = delta + ft_state_dict[k] - v
                else:
                    raise ValueError(
                        f"Key {k} not found in fine-tuned state dicts. Ensure all fine-tuned models have the same architecture."
                    )
            new_state_dict[k] = pretrain_state_dict[k] + coeff * delta

        safetensors.torch.save_file(
            new_state_dict,
            Path(output_path + "checkpoints/checkpoint_000000/llm/decoder/")
            / "consolidated.safetensors",
        )
    elif Path(pretrain_path + "/llm/consolidated.safetensors").exists():
        pretrain_state_dict = load_state_dict(
            Path(pretrain_path) / "llm/", dtype=torch.float32
        )

        fine_tune_state_dicts = [
            load_state_dict(Path(path) / "llm/", dtype=torch.float32)
            for path in fine_tune_paths
        ]

        if not Path(output_path + "checkpoints/checkpoint_000000/llm/").exists():
            Path(output_path + "checkpoints/checkpoint_000000/llm/").mkdir(
                parents=True, exist_ok=True
            )
        new_state_dict = {}
        for k, v in pretrain_state_dict.items():
            delta = torch.zeros_like(v)
            for ft_state_dict in fine_tune_state_dicts:
                m = torch.bernoulli(
                    torch.ones_like(v) * drop_rate, generator=generator
                ).to(v.device)
                delta_param = (1 - m) * (ft_state_dict[k] - v) / (1 - drop_rate)
                delta = delta + delta_param
                if k in ft_state_dict.keys():
                    delta = delta + ft_state_dict[k] - v
                else:
                    raise ValueError(
                        f"Key {k} not found in fine-tuned state dicts. Ensure all fine-tuned models have the same architecture."
                    )
            new_state_dict[k] = pretrain_state_dict[k] + coeff * delta
        safetensors.torch.save_file(
            new_state_dict,
            Path(output_path + "checkpoints/checkpoint_000000/llm/")
            / "consolidated.safetensors",
        )
