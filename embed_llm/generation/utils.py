import pandas as pd
from models.utils import is_torchrun
import logging
import torch

def main_logger_info(logger, message: str) -> None:
    if not is_torchrun() or torch.distributed.get_rank() == 0:
        logger.info(message)

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
                    for temp in results[run_name][ckpt][metric].keys():
                        for result in results[run_name][ckpt][metric][temp]:
                            for res in result:
                                formated_results = pd.concat(
                                    [
                                        formated_results,
                                        pd.DataFrame(
                                            {
                                                "run_name": run_name,
                                                "ckpt": int(ckpt),
                                                "temp": float(temp),
                                                metric: res["Metric"],
                                                "n_samples": res["n_samples"],
                                                "eval_data_type": res["eval_data_type"],
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
                                            },
                                            index=[0],
                                        ),
                                    ]
                                )
                            else:
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
                                                "F1": res["Metric"],
                                                "Prop_a_in_cont": res.get(
                                                    "Prop context containing the answer",
                                                    None,
                                                ),
                                                "n_passages": res.get("n_passages", 1),
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
