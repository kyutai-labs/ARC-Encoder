import logging
from dataclasses import dataclass
from mistral_common.tokens.tokenizers.base import Tokenizer as MistralTokenizer
from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.gemma.tokenizer import Tokenizer as GemmaTokenizer


Tokenizer = MistralTokenizer | LlamaTokenizer | GemmaTokenizer

logger = logging.getLogger("tokenize")

Sequence = list[int]
Mask = list[bool]


@dataclass()
class EmbedPassage:
    tokens: Sequence | list[Sequence]
    text: str | list[str]
    
@dataclass()
class TokenSample:
    tokens: Sequence
    masks: Mask
    passage: EmbedPassage | None = None


def encode(
    data: dict[str, object],
    tokenizer: Tokenizer | None = None,
    continuation: bool = False,
) -> TokenSample | None:
    sample = get_sample(data)
    return tokenize(sample=sample, tokenizer=tokenizer)


def get_sample(data: dict[str, object]) -> str:
    content_keys = ["text", "content"]
    assert not all(
        k in data for k in content_keys
    ), "Make sure to have either 'text' or 'content' in your data. Not both."
    assert any(
        data.get(k) is not None for k in content_keys
    ), f"Must have one of 'text' or 'content' in your data. Only have {data.keys()}"

    # get first non-None value
    sample = None
    for key in content_keys:
        sample = data[key] if key in data else sample

    assert isinstance(sample, str), sample

    return sample


def tokenize(sample: str, tokenizer: Tokenizer) -> TokenSample:
    tokens = tokenizer.encode(sample, bos=True, eos=True)
    masks = [True] * len(tokens)
    return TokenSample(tokens, masks)
