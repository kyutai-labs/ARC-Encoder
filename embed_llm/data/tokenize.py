import logging
import random
from dataclasses import dataclass
from mistral_common.tokens.tokenizers.base import Tokenizer as MistralTokenizer
from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.gemma.tokenizer import Tokenizer as GemmaTokenizer
from embed_llm.data.utils import templates_for_qa

Tokenizer = MistralTokenizer | LlamaTokenizer | GemmaTokenizer

logger = logging.getLogger("tokenize")

Sequence = list[int]
Mask = list[bool]


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
    tokenizer: Tokenizer | None = None,
    data_path: str | None = None,
) -> TokenSample | None:

    return get_sample(data, data_path, tokenizer)


def get_sample(data: dict[str, object], data_path: str, tokenizer) -> str:
    if "instruct_data" in data_path:

        question = data["question"]

        assert isinstance(data["answer"], str) or isinstance(data["answer"], list)
        if isinstance(data["answer"], list):
            answer = random.choice(data["answer"])
        else:
            answer = data["answer"]

        if "passage" in data.keys():
            assert isinstance(data["passage"], str) or isinstance(data["passage"], list)
            if isinstance(data["passage"], list):
                embed_passage = [random.choice(data["passage"])]
            else:
                embed_passage = [data["passage"]]

        elif "passages" in data.keys():
            assert isinstance(data["passages"], str) or isinstance(
                data["passages"], list
            )
            if isinstance(data["passages"], list):
                embed_passage = [random.choice(data["passages"])]
            else:
                embed_passage = [data["passages"]]
        else:
            raise ValueError("No passage or passages key found in data")

        assert isinstance(question, str), question

        # Add question prompt
        if "QA" in data_path:
            question = random.choice(templates_for_qa).format(question=question)

        q_tokens = tokenizer.encode(question, bos=True, eos=False)
        a_tokens = tokenizer.encode(answer, bos=False, eos=True)

        masks = [False] * len(q_tokens) + [True] * len(a_tokens)

        passages = EmbedPassage(
            [
                tokenizer.encode(passage_sample, bos=True, eos=True)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(q_tokens + a_tokens, masks, passages, data_type = 'instruct')

    else:
        sample = data["text"]

        if data.get("passage") is not None:
            passages = data["passage"]
            embed_passage = passages if isinstance(passages, list) else [passages]
        else:
            embed_passage = [sample]

        assert isinstance(sample, str), sample

        tokens = tokenizer.encode(sample, bos=True, eos=True)

        masks = [True] * len(tokens)

        passages = EmbedPassage(
            [
                tokenizer.encode(passage_sample, bos=True, eos=True)
                for passage_sample in embed_passage
            ],
            embed_passage,
        )

        return TokenSample(tokens, masks, passages, data_type = 'reconstruction')
