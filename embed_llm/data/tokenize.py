import logging
import random
from dataclasses import dataclass
from mistral_common.tokens.tokenizers.base import Tokenizer as MistralTokenizer
from embed_llm.data.utils import templates_for_qa

Tokenizer = MistralTokenizer

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
    max_embed: int = 1,
) -> TokenSample | None:

    return get_sample(data, data_path, tokenizer, max_embed)


def get_sample(data: dict[str, object], data_path: str, tokenizer, max_embed: int = 1) -> str:
    if "instruct_data" in data_path.lower() or "qa" in data_path.lower():

        question = data["question"]

        assert isinstance(data["answer"], str) or isinstance(data["answer"], list)
        if isinstance(data["answer"], list):
            answer = random.choice(data["answer"])
        else:
            answer = data["answer"]

        if "passage" in data.keys():
            assert isinstance(data["passage"], str) or isinstance(data["passage"], list)
            if isinstance(data["passage"], list):
                if max_embed <= -1:
                    embed_passage = data["passage"][:-max_embed]
                elif max_embed == 1:
                    embed_passage = data["passage"][:max_embed]
                elif max_embed > 1:
                    n_embed = random.randint(1, max_embed)
                    embed_passage = data["passage"][:n_embed]
            else:
                embed_passage = [data["passage"]]

        elif "passages" in data.keys():
            assert isinstance(data["passages"], str) or isinstance(
                data["passages"], list
            )
            if isinstance(data["passages"], list):
                if max_embed <= -1:
                    embed_passage = data["passages"][:-max_embed]
                elif max_embed == 1:
                    embed_passage = data["passages"][:max_embed]
                elif max_embed > 1:
                    n_embed = random.randint(1, max_embed)
                    embed_passage = data["passages"][:n_embed]
            else:
                embed_passage = [data["passages"]]
        else:
            raise ValueError("No passage or passages key found in data")

        assert isinstance(question, str), question

        # Add question prompt
        if "qa" in data_path.lower() or "reading_comp" in data_path.lower():
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

        return TokenSample(q_tokens + a_tokens, masks, passages, data_type="instruct")

    else:
        sample = data["text"]
        if data.get("passage") is not None:
            passages = data["passage"]
            
            if max_embed <= -1:
                n_embed = -max_embed
            elif max_embed == 1:
                n_embed = max_embed
            elif max_embed > 1:
                n_embed = random.randint(1, max_embed)
        
                
            embed_passage = [passages] if not isinstance(passages, list) else passages[:n_embed]
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

        return TokenSample(tokens, masks, passages, data_type="reconstruction")
