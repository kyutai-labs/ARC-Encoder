import logging
import random
from dataclasses import dataclass
from embed_llm.models.utils.mistral_tokenizer import MistralTokenizer
from embed_llm.models.utils.llama_tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.utils.olmo_tokenizer import Tokenizer as OlmoTokenizer
from embed_llm.data.utils import (
    TEMPLATES_FOR_QA,
)

logger = logging.getLogger("tokenize")

Sequence = list[int]
Mask = list[bool]


@dataclass()
class Tokenizer:
    tokenizer: MistralTokenizer | LlamaTokenizer | OlmoTokenizer
    model_name: str


@dataclass()
class EmbedPassage:
    tokens: list[Sequence]
    text: list[str]


@dataclass()
class TokenSample:
    tokens: Sequence
    masks: Mask
    passages: EmbedPassage
    data_type: str | None = None
    instruction: str | None = (
        None  # Used for instruction data to indicate if the sample is an instruction or not
    )


def encode(
    data: dict[str, object],
    llm_tokenizer: Tokenizer | None = None,  # type: ignore
    embed_tokenizer: Tokenizer | None = None,  # type: ignore
    max_passages: int = 1,
    instruct: bool = False,
    instruct_decoder: bool = False,
) -> TokenSample | None:
    return get_sample(
        data,
        llm_tokenizer,
        embed_tokenizer,
        max_passages,
        instruct,
        instruct_decoder=instruct_decoder,
    )


def get_sample(
    data: dict[str, object],
    llm_tokenizer,
    embed_tokenizer,
    max_passages: int = 1,
    instruct: bool = False,
    instruct_decoder: bool = False,
) -> str:
    if instruct:
        question = data.get("question", "")

        assert isinstance(data["answer"], str) or isinstance(data["answer"], list)
        if isinstance(data["answer"], list):
            answer = random.choice(data["answer"])
        else:
            answer = data["answer"]

        if "passage" in data.keys() or "passages" in data.keys():
            pass_key = "passage" if "passage" in data.keys() else "passages"
            assert isinstance(data[pass_key], str) or isinstance(data[pass_key], list)
            if isinstance(data[pass_key], list):
                if max_passages <= -1:
                    embed_passage = data[pass_key][:-max_passages]
                elif max_passages == 1:
                    embed_passage = data[pass_key][:max_passages]
                elif max_passages > 1:
                    n_embed = random.randint(1, max_passages)
                    embed_passage = data[pass_key][:n_embed]
            else:
                embed_passage = [data[pass_key]]
        else:
            raise ValueError("No passage or passages key found in data")

        if "instruction" in data.keys():
            instruction = data["instruction"]
        else:
            instruction = None

        assert isinstance(question, str), question

        if question == "":
            q_tokens = []
            a_tokens = llm_tokenizer.tokenizer.encode(
                answer, bos=False, eos=True if not instruct_decoder else False
            )
        else:
            question = random.choice(TEMPLATES_FOR_QA).format(question=question)
            q_tokens = llm_tokenizer.tokenizer.encode(question, bos=False, eos=False)
            a_tokens = llm_tokenizer.tokenizer.encode(
                answer, bos=False, eos=True if not instruct_decoder else False
            )

        masks = [False] * len(q_tokens) + [True] * len(a_tokens)

        passages = EmbedPassage(
            [
                embed_tokenizer.tokenizer.encode(passage_sample, bos=False, eos=False)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(
            q_tokens + a_tokens,
            masks,
            passages,
            data_type="instruct",
            instruction=instruction,
        )

    else:
        sample = data["text"]
        embed_passage = [sample]

        assert isinstance(sample, str), sample

        tokens = llm_tokenizer.tokenizer.encode(
            sample,
            bos=True if not instruct_decoder else False,
            eos=True if not instruct_decoder else False,
        )

        masks = [True] * len(tokens)

        passages = EmbedPassage(
            [
                embed_tokenizer.tokenizer.encode(passage_sample, bos=False, eos=False)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(tokens, masks, passages, data_type="reconstruction")
