import logging
import random
from dataclasses import dataclass
from embed_llm.models.utils.mistral_tokenizer import MistralTokenizer
from embed_llm.models.utils.llama_tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.utils.olmo_tokenizer import Tokenizer as OlmoTokenizer

logger = logging.getLogger("tokenize")

Sequence = list[int]
Mask = list[bool]

TEMPLATE_FOR_QA = "\nQuestion: {question}\nAnswer: "

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


def encode(
    data: dict[str, object],
    llm_tokenizer: Tokenizer | None = None,  # type: ignore
    embed_tokenizer: Tokenizer | None = None,  # type: ignore
    data_path: str | None = None,
    max_passages: int = 1,
) -> TokenSample | None:
    return get_sample(data, data_path, llm_tokenizer, embed_tokenizer, max_passages)


def get_sample(
    data: dict[str, object],
    llm_tokenizer,
    embed_tokenizer,
    max_passages: int = 1,
    instruct: bool = False,
) -> str:
    if instruct:
        question = data["question"]

        assert isinstance(data["answer"], str) or isinstance(data["answer"], list)
        if isinstance(data["answer"], list):
            answer = random.choice(data["answer"])
        else:
            answer = data["answer"]

        if "passage" in data.keys():
            assert isinstance(data["passage"], str) or isinstance(data["passage"], list)
            if isinstance(data["passage"], list):
                if max_passages <= -1:
                    embed_passage = data["passage"][:-max_passages]
                elif max_passages == 1:
                    embed_passage = data["passage"][:max_passages]
                elif max_passages > 1:
                    n_embed = random.randint(1, max_passages)
                    embed_passage = data["passage"][:n_embed]
            else:
                embed_passage = [data["passage"]]

        elif "passages" in data.keys():
            assert isinstance(data["passages"], str) or isinstance(
                data["passages"], list
            )
            if isinstance(data["passages"], list):
                if max_passages <= -1:
                    embed_passage = data["passages"][:-max_passages]
                elif max_passages == 1:
                    embed_passage = data["passages"][:max_passages]
                elif max_passages > 1:
                    n_embed = random.randint(1, max_passages)
                    embed_passage = data["passages"][:n_embed]
            else:
                embed_passage = [data["passages"]]
        else:
            raise ValueError("No passage or passages key found in data")

        assert isinstance(question, str), question

        question = TEMPLATE_FOR_QA.format(question=question)

        q_tokens = llm_tokenizer.tokenizer.encode(question, bos=False, eos=False)
        a_tokens = llm_tokenizer.tokenizer.encode(answer, bos=False, eos=True)

        masks = [False] * len(q_tokens) + [True] * len(a_tokens)
        # masks = [True] * len(q_tokens) + [True] * len(a_tokens)

        passages = EmbedPassage(
            [
                embed_tokenizer.tokenizer.encode(passage_sample, bos=False, eos=False)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(q_tokens + a_tokens, masks, passages, data_type="instruct")

    else:
        sample = data["text"]
        embed_passage = [sample]

        assert isinstance(sample, str), sample

        tokens = llm_tokenizer.tokenizer.encode(sample, bos=True, eos=True)

        masks = [True] * len(tokens)

        passages = EmbedPassage(
            [
                embed_tokenizer.tokenizer.encode(passage_sample, bos=False, eos=False)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(tokens, masks, passages, data_type="reconstruction")
