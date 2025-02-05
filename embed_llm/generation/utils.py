import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os 
import torch
import random
import numpy as np
from pathlib import Path
import logging
from embed_llm.models.augmented_model import EmbedAugPipeline
from mistral_inference.transformer import Transformer
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

def load_pipeline(
    run_name: str | None,
    tmp_path: str,
    llm_path: str,
    device: str,
    max_bs: int,
    pipeline: EmbedAugPipeline | Transformer | None = None,
    mistral: bool = False,
    ckpt: int | None = None,
    instruct_name: str = None,
) -> EmbedAugPipeline | Transformer:
    if pipeline is None and is_torchrun():
            torch.distributed.init_process_group()
            torch.cuda.set_device(torch.distributed.get_rank())
            device = "cuda"
            num_pipeline_ranks = torch.distributed.get_world_size()
    else:
        num_pipeline_ranks = 1
     
    if not mistral:
        if pipeline is None:
            # Get last checkpoint
            if instruct_name is None:
                assert run_name is not None
                last_ckpt = (
                    sorted(
                        [
                            ckpt_name
                            for ckpt_name in os.listdir(
                                tmp_path + run_name + "/checkpoints/"
                            )
                            if (
                                Path(tmp_path + run_name + "/checkpoints/")
                                / ckpt_name
                                / "params.json"
                            ).exists()
                        ]
                    )[-1]
                    if ckpt is None
                    else f"checkpoint_{ckpt:06d}"
                )

                pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
                    llm_path=llm_path,
                    ckpt_path=tmp_path + run_name + "/checkpoints/" + last_ckpt,
                    device=device,
                    llm_name="Mistral7B",
                    embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
                    max_batch_size=max_bs,
                    instruct_ckpt=None,
                    num_pipeline_ranks=num_pipeline_ranks,
                )
            else:
                last_ckpt = sorted(
                    [
                        ckpt_name
                        for ckpt_name in os.listdir(
                            tmp_path + instruct_name + "/checkpoints/"
                        )
                        if (
                            Path(tmp_path + instruct_name + "/checkpoints/")
                            / ckpt_name
                            / "params.json"
                        ).exists()
                    ]
                )[-1]
                
                last_ckpt_run_name =sorted(
                    [
                        ckpt_name
                        for ckpt_name in os.listdir(
                            tmp_path + run_name + "/checkpoints/"
                        )
                        if (
                            Path(tmp_path + run_name + "/checkpoints/")
                            / ckpt_name
                            / "params.json"
                        ).exists()
                    ]
                )[-1]
                pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
                    llm_path=llm_path,
                    ckpt_path=tmp_path +  run_name + "/checkpoints/" + last_ckpt_run_name,
                    device=device,
                    llm_name="Mistral7B",
                    embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
                    max_batch_size=max_bs,
                    instruct_ckpt=tmp_path
                    + instruct_name
                    + "/checkpoints/"
                    + last_ckpt,
                    num_pipeline_ranks=num_pipeline_ranks,
                )

            ckpt = int(last_ckpt.split("_")[-1])
            eval_logger_info(logger,f"Evaluating checkpoint {ckpt}")
        else:
            pipeline: EmbedAugPipeline = pipeline
            ckpt = ckpt

        return pipeline, ckpt
    else:
        if pipeline is None:
            mistral_model = Transformer.from_folder(
                "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/",
                device=device,
                max_batch_size=max_bs,
                dtype=torch.float32,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        else:
            mistral_model = pipeline

        return mistral_model, None



def format_results(results: dict, benchmark: str):

    if benchmark.lower() == "nq" or benchmark.lower() == "triviaqa":
        key_list = [
            "run_name",
            "ckpt",
            "temp",
            "n_samples",
            "icl_examples",
            "context_in_examples",
            "context_w_query",
            "EM Metric",
            "EM approx_Metric",
            "F1",
            "Prop_a_in_cont",
            "n_passages",
            "colbert",
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

            else:
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
                                                "context_w_query": res[
                                                    "w_context_w_query"
                                                ],
                                                "EM Metric": res["Metric"],
                                                "EM approx_Metric": res[
                                                    "approx_Metric"
                                                ],
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "n_passages": res.get("n_passages", 1),
                                                "colbert": res.get("colbert", None),
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
                                                "context_w_query": res[
                                                    "w_context_w_query"
                                                ],
                                                "F1": res["Metric"],
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "n_passages": res.get("n_passages", 1),
                                                "colbert": res.get("colbert", None),
                                            },
                                            index=[0],
                                        )
        
                                    
                                formated_results = pd.concat([formated_results, df_res])
                                
                formated_results = (
                    formated_results.groupby(
                        [
                            "run_name",
                            "ckpt",
                            "temp",
                            "n_samples",
                            "icl_examples",
                            "context_in_examples",
                            "context_w_query",
                            "n_passages",
                        ]
                    )
                    .first()
                    .reset_index()
                )
    return formated_results
