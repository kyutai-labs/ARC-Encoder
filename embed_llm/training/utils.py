import contextlib
import dataclasses
import datetime
import logging
import time
import torch
from typing import Protocol
import numpy as np
import random
import json
from embed_llm.data.args import DataArgs


def create_data_args(params_path: str):
    train_data = ""
    eval_data = ""
    adapt_seq_len = False
    n_datasets = 0
    with open(params_path, "r") as f:
        for i, line in enumerate(f):
            try:
                if line.strip():
                    params = json.loads(line)
                else:
                    continue
            except json.JSONDecodeError:
                print(f"Error in line {i}: {line}")
            if i == 0:
                adapt_seq_len = params["adapt_seq_len"]
                data_types = [params["data_types"]]
                eval_data_cpp = params["eval_data_common_path_prefix"]
                train_data_cpp = params["train_data_common_path_prefix"]
                shuffle = params["shuffle"]
                continue

            if "train_data" in params.keys():
                weight = (
                    1
                    if "weight" not in params["train_data"].keys()
                    else params["train_data"]["weight"]
                )
                train_data += (
                    train_data_cpp + params["train_data"]["path"] + ":" + weight + ","
                )
                n_datasets += 1

            if "eval_data" in params.keys():
                weight = (
                    1
                    if "weight" not in params["eval_data"].keys()
                    else params["eval_data"]["weight"]
                )
                eval_data += (
                    eval_data_cpp + params["eval_data"]["path"] + ":" + weight + ","
                )

    return DataArgs(
        train_data=train_data[:-1],  # Remove last ','
        eval_data=eval_data[:-1],
        adapt_seq_len=adapt_seq_len,
        data_types=data_types * n_datasets,  # Useful to add prompt prefix for training
        shuffle=shuffle,
    )


PARAPHRASE_PROMPT = [
    {"prefix": "Background: ", "suffix": " means the same as "},
    {
        "prefix": "Background: ",
        "suffix": " Can you put the above sentences in your own terms? ",
    },
    {
        "prefix": "",
        "suffix": "Please provide a reinterpretation of the preceding background text. ",
    },
    {
        "prefix": "These two expressions are equivalent in essence:(1) ",
        "suffix": " (2) ",
    },
    {"prefix": "Background: ", "suffix": " is a paraphrase of what? "},
    {
        "prefix": "",
        "suffix": "Could you give me a different version of the background sentences above? ",
    },
    {
        "prefix": "In other words, background: ",
        "suffix": " is just another way of saying: ",
    },
    {
        "prefix": "You’re getting across the same point whether you say background: ",
        "suffix": " or ",
    },
    {
        "prefix": "",
        "suffix": "After uppacking the ideas in the background information above, we got: ",
    },
    {
        "prefix": "",
        "suffix": "Please offer a restatement of the background sentences I’ve just read. ",
    },
    {"prefix": "Background: ", "suffix": " , which also means: "},
    {
        "prefix": "Strip away the mystery, and you’ll find ",
        "suffix": " is simply another rendition of: ",
    },
    {
        "prefix": "The essence of background: ",
        "suffix": " is captured again in the following statement: ",
    },
]

INSTRUCT_PROMPT = [
    {"prefix": "Context: ", "suffix": ""},
    {"prefix": "Based on this document", "suffix": "follow the instruction below\n"},
    {"prefix": "", "suffix": ""},
    {"prefix": "", "suffix": ""},
]

CONTINUATION_PROMPT = [
    {
        "prefix": "",
        "suffix": " Complete this thought by adding what logically follows: ",
    },
    {
        "prefix": "",
        "suffix": " provide the natural continuation of this statement: ",
    },
    {
        "prefix": "",
        "suffix": "What would be the most appropriate way to complete this?\n",
    },
    {
        "prefix": "",
        "suffix": "Extend this statement to its full meaning:\n",
    },
    {
        "prefix": "",
        "suffix": "finish expressing the complete idea.",
    },
    {
        "prefix": "",
        "suffix": "Continue this sentence in a way that maintains coherence and relevance: ",
    },
    {
        "prefix": "",
        "suffix": " Carry on from here with what follows naturally: ",
    },
    {
        "prefix": "",
        "suffix": " develop this idea further with a logical continuation: ",
    },
    {
        "prefix": "",
        "suffix": "How would this statement logically proceed?\n",
    },
    {
        "prefix": "",
        "suffix": "What follows from this idea?\n",
    },
    {
        "prefix": "",
        "suffix": " expand on this thought to complete its meaning: ",
    },
]

logger = logging.getLogger("utils")


@dataclasses.dataclass
class TrainState:
    max_steps: int
    step: int = 0
    elapsed_time: float = 0.0
    n_seen_tokens: int = 0
    this_step_time: float = 0.0
    begin_step_time: float = 0.0
    this_eval_perplexity_rec: float | None = None
    this_eval_perplexity_textcont: float | None = None
    this_eval_perplexity_embcont: float | None = None
    this_eval_loss_rec: float | None = None
    this_eval_loss_textcont: float | None = None
    this_eval_loss_embcont: float | None = None
    this_eval_kl_loss: float | None = None
    this_eval_loss_nocontext: float | None = None
    this_eval_perplexity_nocontext: float | None = None

    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens

        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += self.this_step_tokens

        self.begin_step_time = time.time()

    @property
    def wps(self):
        return self.this_step_tokens / self.this_step_time

    @property
    def avg_wps(self):
        return self.n_seen_tokens / self.elapsed_time

    @property
    def eta(self):
        steps_left = self.max_steps - self.step
        avg_time_per_step = self.elapsed_time / self.step

        return steps_left * avg_time_per_step


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    """
    Logging the closing to be sure something is not hanging at exit time
    """
    try:
        setattr(thing, "wrapped_by_closing", True)
        yield
    finally:
        logger.info(f"Closing: {name}")
        try:
            thing.close()
        except Exception:
            logger.error(f"Error while closing {name}!")
            raise
        logger.info(f"Closed: {name}")


def now_as_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
