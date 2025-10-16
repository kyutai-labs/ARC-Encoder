import warnings
import logging
import os
import random
import numpy as np
import torch
from embed_llm.models.utils.utils import is_torchrun
from embed_llm.generation.metrics import get_em, get_rouge_score
from embed_llm import DATA_PATH

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


EVAL_DATA_PATH = {
    "NQ":  DATA_PATH + "w_retrieved/nq_validation.jsonl",
    "TRIVIAQA":  DATA_PATH + "w_retrieved/triviaqa_validation.jsonl",
    "SQUAD": DATA_PATH + "raw/squad_validation.jsonl",
    "DistractorHotpotQA": DATA_PATH + "raw/hotpotqa_validation.jsonl",
    "CNN": DATA_PATH + "raw/cnn_validation.jsonl",
}

METRIC_EVALUATION = {
    "NQ": get_em,
    "TRIVIAQA": get_em,
    "SQUAD": get_em,
    "DistractorHotpotQA": get_em,
    "CNN": get_rouge_score,
}


TRAD_DATA_PATH = {
    "English":  DATA_PATH + "flores/eng_Latn.jsonl",
    "Spanish":  DATA_PATH + "flores/spa_Latn.jsonl",
    "French":  DATA_PATH + "flores/fra_Latn.jsonl",
    "German":  DATA_PATH + "flores/deu_Latn.jsonl",
    "Danish":  DATA_PATH + "flores/dan_Latn.jsonl",
}


def create_prompt(
    prefix_prompt: list[str],
    prefix_embed: list[str] | None,
    doc: list[str],
    query: str,
    wdoc: bool = True,
    w_embeds: bool = True,
    cat_multi_passages: bool = False,
) -> tuple[list[str], list[str] | None]:
    list_prompt = prefix_prompt.copy()

    if prefix_embed is None and w_embeds:
        list_embed = []
    elif not w_embeds:
        list_embed = []
    else:
        list_embed = prefix_embed.copy()

    assert int(wdoc) * int(w_embeds) == 0, (
        "Cannot use both text context and embeddings as the document in the same time"
    )

    if wdoc:
        doc = "\n".join(doc)
        list_prompt.append(f"Document: {doc.strip()}\nQuestion: {query}\nAnswer:")
        return list_prompt, list_embed
    else:
        if w_embeds:
            last_prompt = list_prompt[-1]
            list_prompt[-1] = "".join([last_prompt, "Document: "])

            if len(doc) == 1 or cat_multi_passages:
                doc = "\n".join(doc)
                list_embed.append(doc.strip())
            else:
                for d in doc:
                    list_embed.append(d.strip())
                    list_prompt.append(
                        ""
                    )  # Add an empty string to separate passages so that there are embedded separately
                list_prompt = list_prompt[:-1]  # Remove the last empty string
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")
        else:
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")
        return list_prompt, list_embed


def create_prompt_prefix(
    queries: list[str],
    answers: list[str],
    docs: list[list[str]] | None = None,
    max_examples: int | None = None,
    cat_multi_passages: bool = False,
    compressed_doc_in_icl: bool = True,
) -> tuple[list[str], list[str] | None]:
    max_examples = max_examples if max_examples is not None else len(queries)
    prompt_str = []
    to_embed_str = []

    prompt = ""
    if docs is not None:
        if compressed_doc_in_icl:
            for query, answer, doc, index in zip(
                queries, answers, docs, range(max_examples)
            ):
                if len(doc) == 1:
                    doc = doc[0]
                elif len(doc) > 1 and not cat_multi_passages:
                    doc = "\n".join(doc)

                if index == 0:
                    prompt_str.append("Document: ")

                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())

                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                    )
                elif index == max_examples - 1:
                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())
                    prompt_str.append(f"\nQuestion: {query}\nAnswer: {answer}\n\n")
                else:
                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())
                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                    )

            if max_examples == 0:
                prompt_str.append("")
        else:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
                doc = "\n".join(doc)
                prompt += f"Document: {doc}\nQuestion: {query}\nAnswer: {answer}\n\n"

            to_embed_str = None
            prompt_str.append(prompt)

    else:
        for query, answer, _ in zip(queries, answers, range(max_examples)):
            prompt += f"Question: {query}\nAnswer: {answer}\n\n"

        prompt_str.append(prompt)
        to_embed_str = None

    return prompt_str, to_embed_str


def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB - 6}GB"
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


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
