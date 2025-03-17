import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os 
import torch
import random
import numpy as np
from pathlib import Path
import logging
from embed_llm.models.utils import is_torchrun
import torch

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

    if benchmark.lower() == "nq" or benchmark.lower() == "triviaqa" or benchmark.lower() == "hotpotqa" or benchmark.lower() == "squad":
        key_list = [
            "run_name",
            "EM Metric",
            "EM approx_Metric",
            "F1",
            "xRAG metric",
            "ckpt",
            "temp",
            "n_samples",
            "icl_examples",
            "context_in_examples",
            "context_w_query",
            "Prop_a_in_cont",
            "n_passages",
            "compress_ratio",
            "w_scores",
            "fine_tuned",
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
            "w_scores"
        ]
    elif benchmark.lower() == "reconstruction":
        key_list = [
            "run_name",
            "ckpt",
            "temp",
            "n_samples",
            "eval_data_type",
            "Bleu",
            "Trunc Bleu",
            "AVG Bleu",
            "Meteor",
            "EM",
            "Overlap",
        ]
    else:
        raise ValueError("Invalid benchmark")

    formated_results = pd.DataFrame(columns=key_list)

    for run_name in results.keys():
        for ckpt in results[run_name].keys():
            if benchmark.lower() == "reconstruction":
                for metric in [
                    "Bleu",
                    "Trunc Bleu",
                    "AVG Bleu",
                    "Meteor",
                    "EM",
                    "Overlap",
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
                                            metric: result["Metric"],
                                            "n_samples": result["n_samples"],
                                            "eval_data_type": result["eval_data_type"],
                                        },
                                        index=[0],
                                    ),
                                ]
                            )
                formated_results = (
                    formated_results.groupby(
                        ["run_name", "ckpt", "temp", "n_samples", "eval_data_type"]
                    )
                    .first()
                    .reset_index()
                )
            
            elif benchmark.lower() == "nq" or benchmark.lower() == "triviaqa" or benchmark.lower() == "hotpotqa" or benchmark.lower() == "squad":
                for metric in ["EM", "F1"]:
                    if benchmark not in results[run_name][ckpt].keys():
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
                                                "context_in_examples": res[
                                                    "w_context_in_examples"
                                                ],
                                                "context_w_query": res.get(
                                                    "w_context_w_query",False),
                                                "xRAG metric": res.get("xRAG metric",None),
                                                "EM Metric": res.get("Metric",None),
                                                "EM approx_Metric": res.get(
                                                    "approx_Metric"
                                                ,None),
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "n_passages": res.get("n_passages", 1),
                                                "compress_ratio": res.get("compress_ratio", None),
                                                "fine_tuned": res.get("fine_tuned", None),
                                                "w_scores": res.get("w_scores", 0.),
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
                                                "context_in_examples": res[
                                                    "w_context_in_examples"
                                                ],
                                                "context_w_query": res.get(
                                                    "w_context_w_query", False),
                                                "F1": res.get("Metric",None),
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "n_passages": res.get("n_passages", 1),
                                                "w_scores": res.get("w_scores", 0.),
                                                "fine_tuned": res.get("fine_tuned", None),
                                            },
                                            index=[0],
                                        )
        
                                    
                                formated_results = pd.concat([formated_results, df_res])
            else:
                if benchmark not in results[run_name][ckpt].keys():
                    continue
                for temp in results[run_name][ckpt][benchmark].keys():
                    for k in range(len(results[run_name][ckpt][benchmark][temp]['n_samples'])):
                        res = results[run_name][ckpt][benchmark][temp]
                        df_res = pd.DataFrame(
                                    {
                                        "run_name": run_name,
                                        "ckpt": int(ckpt),
                                        "temp": float(temp),
                                        "n_samples": res["n_samples"][k],
                                        "icl_examples": res["icl_examples"][k],
                                        "context_in_examples": res[
                                            "w_context_in_examples"
                                        ][k],
                                        "Metric": res["Metric"][k],
                                        "Prop_a_in_cont": res.get(
                                            "Prop context containing the answer",
                                            [None]*100,
                                        )[k],
                                        "n_passages": res.get("n_passages", 1)[k],
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
                            "fine_tuned",
                            "w_scores",
                        ]
                    )
                    .first()
                    .reset_index(allow_duplicates=True)
                )
            else:
                formated_results = (formated_results.groupby(
                        [
                            "run_name",
                            "ckpt",
                            "temp",
                            "n_samples",
                            "icl_examples",
                            "context_in_examples",
                            "n_passages",
                            "w_scores",

                        ]
                    )
                    .first()
                    .reset_index(allow_duplicates=True)
                )
    return formated_results
