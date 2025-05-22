import contextlib
import dataclasses
import datetime
import logging
import time
import torch
from typing import Protocol
import numpy as np
import random


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
    comp_rate: float = 0.0

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
