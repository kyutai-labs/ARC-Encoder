import dataclasses
import itertools
import json
import logging
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import torch.distributed as dist
from embed_llm.training.distributed import get_rank
from embed_llm.training.args import HybridTask
from embed_llm.data.args import DataArgs
from embed_llm.data.tokenize import Mask, TokenSample, encode, Tokenizer

logger = logging.getLogger("dataset")


_LOADED_DATASETS: dict[Path, list[TokenSample]] = {}


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path,
    rank: int,
    world_size: int,
    tokenizer: Tokenizer | None = None,
) -> list[TokenSample]:
    global _LOADED_DATASETS

    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    data_list: list[TokenSample] = []
    for line in lines:
        data = json.loads(line)
        
        if "rand" in data.keys() and float(data["rand"]) >= 0.8:
            continue

        data_sample: TokenSample = encode(
            data, tokenizer=tokenizer, data_path=str(path)
        )
        data_list.append(data_sample)

    main_logger_info(f"{path} loaded and tokenized.")
    _LOADED_DATASETS[path] = data_list

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert (
            len(jsonl_files) > 0
        ), f"{self.path} does not seem to have any files ending with '.jsonl'"
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []
    for source in pretrain_data.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source as <path/to/data>:<weight> or just <path/to/data>"
            )

        assert (
            path_ not in seen
        ), f"{path_} seems to be duplicated. Make sure to only add it once."
        assert (
            weight > 0
        ), f"Make sure to define strictly positive data sampling weights, not {weight}"

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or directory that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert (
        abs(1 - sum(n_weights)) < 1e-8
    ), f"Defined data sampling weights {weights} must sum to 1."
    return sources, n_weights


@dataclasses.dataclass()
class SequenceEmbedMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: list[int]
    y: list[int]
    to_embed: list[dict[str, str | int | list[int] | list[str]]]
    mask: Mask
    sizes: list[int]
    data_type: str
    n_prefixes: list[int] | None = None

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask)
        assert len(self.to_embed) == len(self.sizes)
        if self.n_prefixes is not None:
            assert len(self.n_prefixes) == len(self.sizes)


def sequence_iterator_hybrid(
    sample: TokenSample,
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    n_prefixes: list[int],
    mask_buffer: Mask,
    sizes: list[int],
    seq_len: int,
    tokenizer: Tokenizer,
    max_n_prefixes: int = 1,
    min_n_prefixes: int = 0,
    prop_continuation: float = 0.5,
    prop_uselessembed_continuation: float = 0.0,
    useless_embed_continuation: bool = False,
) -> SequenceEmbedMaskAndSizes:
    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)

    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]

    # Embed passage should be the same as x but might be divided in several sequence !
    embed_tokens = sample.passages.tokens

    if min_n_prefixes == 0 and max_n_prefixes == 0:
        n_prefix_tokens = 0
    else:
        n_prefix_tokens = np.random.randint(min_n_prefixes, max_n_prefixes + 1)

    continuation = False

    if not useless_embed_continuation and np.random.rand() < prop_continuation:
        
        if np.random.rand() < prop_uselessembed_continuation:
            # Truncate such that all the passages represent just [0,start_continuation] tokens
            new_embed = []
            n_emb_toks = 0

            # Truncate such that all the passages represent just [0,seq_len] tokens
            for passage in embed_tokens:
                if n_emb_toks + len(passage) <= seq_len:
                    n_emb_toks += len(passage)
                    new_embed.append(passage)
                    continue
                else:
                    new_embed.append(passage[: seq_len - n_emb_toks])
                    break

            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
            return to_embed_buffer, True
        else:
            # If the passage is too short, we can't continue from it
            if len(x) // 4 >= min((len(x) - 1) - 10, seq_len) or len(x) // 4 - n_prefix_tokens < 0:
                return None

            # Can't embed more than seq_len tokens
            # Arbitrary, to have enough context to continue from and to continue at least 10 tokens.
            start_lm = np.random.randint(len(x) // 4, min((len(x) - 1) - 10, seq_len))

            new_embed = []
            n_emb_toks = 0

            # Truncate such that all the passages represent just [0,start_continuation] tokens
            for passage in embed_tokens:
                if n_emb_toks + len(passage) <= start_lm:
                    n_emb_toks += len(passage)
                    new_embed.append(passage)
                    continue
                else:
                    new_embed.append(passage[: start_lm - n_emb_toks])
                    break

            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
            continuation = True
    elif useless_embed_continuation:
        n_prefix_tokens = 0
        start_lm = min(len(x)//2,8192-seq_len) #To not exceed Mistral7B context window
        x_buffer.extend(x[:start_lm])
        y_buffer.extend(y[:start_lm])
        mask_buffer.extend([False]*start_lm)
        
    else:
        # If the passage is too short, we can't reconstruct at least 10 tokens
        if n_prefix_tokens >= (len(x) - 1) - 10:
            return None

        # Can't embed more than seq_len tokens
        # Reconstruct at least 10 tokens
        start_lm = np.random.randint(n_prefix_tokens, min((len(x) - 1) - 10, seq_len))

        new_embed = []
        n_emb_toks = 0

        # Truncate such that all the passages represent just [0,seq_len] tokens
        for passage in embed_tokens:
            if n_emb_toks + len(passage) <= seq_len:
                n_emb_toks += len(passage)
                new_embed.append(passage)
                continue
            else:
                new_embed.append(passage[: seq_len - n_emb_toks])
                break

        to_embed_buffer.append(
            {
                "text": [tokenizer.decode(toks) for toks in new_embed],
                "tokens": new_embed,
            }
        )

    # Continue/Reconstruct maximum seq_len tokens
    x_buffer.extend(
        x[start_lm - n_prefix_tokens : start_lm - n_prefix_tokens + seq_len]
    )
    y_buffer.extend(
        y[start_lm - n_prefix_tokens : start_lm - n_prefix_tokens + seq_len]
    )
    mask_buffer.extend(
        n_prefix_tokens * [False]
        + mask[start_lm : start_lm - n_prefix_tokens + seq_len]
    )
    
    if not useless_embed_continuation:
        sizes.append(
            min(len(x), start_lm + seq_len - n_prefix_tokens) - start_lm + n_prefix_tokens
        )
    else:
        sizes.append(min(start_lm+seq_len,len(x)))
        
    n_prefixes.append(n_prefix_tokens)

    assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
    assert len(to_embed_buffer) == len(sizes)

    # we don't want to yield sequences with a mask filled with False
    if any(mask_buffer):
        if continuation:
            data_type = "continuation"
        elif useless_embed_continuation:
            data_type = "uselessembed_continuation"
        else:
            data_type = "reconstruction"
            
        return SequenceEmbedMaskAndSizes(
            x=x_buffer,
            y=y_buffer,
            to_embed=to_embed_buffer,
            mask=mask_buffer,
            sizes=sizes,
            data_type=data_type,
            n_prefixes = n_prefixes,
        )
    else:
        return None


def sequence_iterator_one_task_4_all(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    mask_buffer: Mask,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    n_prefixes: list[int],
    max_embeds: int = 1,
) -> SequenceEmbedMaskAndSizes:
    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)

    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]

    cur_pos = 0

    while cur_pos < len(x):
        
        if len(x) - cur_pos >= (max_embeds + 1) * seq_len - 10:
            if max_embeds == 1:
                to_embed_buffer.append(
                    {
                        "text": [tokenizer.decode(tokens[cur_pos:cur_pos + seq_len])],
                        "tokens": [tokens[cur_pos:cur_pos + seq_len]],
                    }
                )
                end_embed = seq_len
            else:
                nb_embed = np.random.randint(1, max_embeds + 1)
                new_embed = []
                n_embed_toks = 0
                
                for i in range(nb_embed):
                    new_embed.append(tokens[cur_pos + i*seq_len: cur_pos + (i + 1)*seq_len])
                    n_embed_toks += len(tokens[cur_pos + i*seq_len: cur_pos + (i + 1)*seq_len])
    
                    
                to_embed_buffer.append(
                    {
                        "text": [tokenizer.decode(toks) for toks in new_embed],
                        "tokens": new_embed,
                    }
                )
                end_embed = n_embed_toks
        

            start_lm = np.random.randint(1, end_embed - 10)
            n_prefixes.append(end_embed - start_lm)
            x_buffer.extend(x[start_lm + cur_pos: start_lm + cur_pos + seq_len])   
            y_buffer.extend(y[start_lm + cur_pos: start_lm + cur_pos + seq_len])
            mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
            size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
            sizes.append(size)
            cur_pos += start_lm + size
            
        # If not enought to put seqlen in both embedding and x, split the rest in two parts
        elif max_embeds == 1 and (len(x) - cur_pos)//2 > 10 + 1: 
            end_embed = (len(x) - cur_pos)//2
            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(tokens[cur_pos:cur_pos + end_embed])],
                    "tokens": [tokens[cur_pos:cur_pos + end_embed]],
                }
            )
            
            start_lm = np.random.randint(1, end_embed - 10)
            n_prefixes.append(end_embed - start_lm)
            x_buffer.extend(x[start_lm + cur_pos:])   
            y_buffer.extend(y[start_lm + cur_pos:])
            mask_buffer.extend(mask[start_lm + cur_pos :])
            size = len(x[start_lm + cur_pos:])
            sizes.append(size)
            cur_pos += start_lm + size

        elif max_embeds > 1 and (len(x) - cur_pos) > 12 + 1 and (len(x) - cur_pos - 2)//max_embeds > 0 : 
            # 10 of prefix + minimum 2 tokens to continue
            nb_embed = np.random.randint(1, max_embeds + 1)
            end_embed = len(x) - cur_pos - 2
            new_embed = []
            n_embed_toks = 0
            
            for i in range(nb_embed):
                new_embed.append(tokens[cur_pos + i*(end_embed//nb_embed): 
                    cur_pos + (i+1)*(end_embed//nb_embed)])
                n_embed_toks += len(tokens[cur_pos + i*(end_embed//nb_embed): 
                    cur_pos + (i+1)*(end_embed//nb_embed)])
    
                
            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
            # In case there is a rest to the euclidean div
            end_embed = n_embed_toks
        
            start_lm = np.random.randint(1, end_embed - 10)
            n_prefixes.append(end_embed - start_lm)
            x_buffer.extend(x[start_lm + cur_pos:start_lm + cur_pos + seq_len])   
            y_buffer.extend(y[start_lm + cur_pos:start_lm + cur_pos + seq_len])
            mask_buffer.extend(mask[start_lm + cur_pos :start_lm + cur_pos + seq_len])
            size = len(x[start_lm + cur_pos :start_lm + cur_pos + seq_len])
            sizes.append(size)
            cur_pos += start_lm + size

        else:
            return None
            


        assert len(mask_buffer) == len(x_buffer) == len(y_buffer), f'{len(mask_buffer)} == {len(x_buffer)} == {len(y_buffer)}'
        assert len(x_buffer) <= seq_len, f'{len(x_buffer)} <= {seq_len}'
        assert sum(sizes) <= seq_len, f'{sum(sizes)} <= {seq_len}'
        assert len(to_embed_buffer) == len(sizes), f'{len(to_embed_buffer)} == {len(sizes)}'
        
        # we don't want to yield sequences with a mask filled with False
        if any(mask_buffer):
            return SequenceEmbedMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                to_embed=to_embed_buffer,
                mask=mask_buffer,
                sizes=sizes,
                data_type='one_4_all',
                n_prefixes = n_prefixes,
            )
        else:
            return None 

   


def sequence_iterator_reconstruction(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    mask_buffer: Mask,
    n_missing: int,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
) -> SequenceEmbedMaskAndSizes:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    if not adapt_seq_len:
        assert n_missing == seq_len - len(
            x_buffer
        ), f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"

    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    embed_text = sample.passages.text
    data_type = sample.data_type
    cur_pos = 0

    while cur_pos < len(x):
        size = len(x[cur_pos : cur_pos + n_missing])

        curr_mask = mask[cur_pos : cur_pos + n_missing]
        if not any(curr_mask):
            cur_pos += size
            # we have a sequence with a mask filled with False
            continue

        x_buffer.extend(x[cur_pos : cur_pos + n_missing])
        y_buffer.extend(y[cur_pos : cur_pos + n_missing])

        # Because regeneration
        if len(embed_tokens) == 1 and data_type == "reconstruction":
            to_embed_buffer.append(
                {
                    "text": [
                        tokenizer.decode(embed_tokens[0][cur_pos : cur_pos + n_missing])
                    ],
                    "tokens": [embed_tokens[0][cur_pos : cur_pos + n_missing]],
                }
            )
        else:
            # If we want to reconstruct from several chunks of embedded text, we need to be able to reconstruct the full passage
            # To implement, prevent reconstruction of too long sequences
            assert adapt_seq_len
            to_embed_buffer.append({"text": embed_text, "tokens": embed_tokens})

        mask_buffer.extend(curr_mask)
        n_missing -= size

        sizes.append(size)

        cur_pos += size

        if n_missing == 0 or (adapt_seq_len and cur_pos == len(x)):
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(x_buffer) <= seq_len

            if not adapt_seq_len:
                assert sum(sizes) == seq_len
                assert seq_len == len(x_buffer)

            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type,
                )

            if adapt_seq_len:
                break
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes


def sequence_iterator_continuation(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    n_missing: int,
    mask_buffer: Mask,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
    data_type: str = "continuation",
) -> SequenceEmbedMaskAndSizes:

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens

    assert (
        len(embed_tokens) == 1
    ), "Continuation training only supports one passage per sample"

    cur_pos = 0

    while cur_pos < len(x):

        overall_size = len(x[cur_pos : cur_pos + n_missing])

        curr_mask = mask[cur_pos : cur_pos + n_missing]
        if not any(curr_mask):
            cur_pos += overall_size
            # we have a sequence with a mask filled with False
            continue

        if overall_size < 4:
            assert len(mask_buffer) == len(x_buffer) == sum(sizes) == len(y_buffer)
            assert len(to_embed_buffer) == len(sizes), (
                len(to_embed_buffer),
                len(sizes),
            )

            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type,
                )

        upper_bound = min(cur_pos + n_missing, len(x))

        x_buffer.extend(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

        y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

        to_embed_buffer.append(
            {
                "text": [
                    tokenizer.decode(
                        embed_tokens[0][
                            cur_pos : cur_pos + (upper_bound - cur_pos) // 2
                        ]
                    )
                ],
                "tokens": [
                    embed_tokens[0][cur_pos : cur_pos + (upper_bound - cur_pos) // 2]
                ],
            }
        )

        mask_buffer.extend(mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
        n_missing -= overall_size

        size = len(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

        sizes.append(size)

        cur_pos += overall_size

        if n_missing == 0 or (adapt_seq_len and cur_pos == len(x)):
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(x_buffer) <= seq_len
            assert len(to_embed_buffer) == len(sizes)

            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type,
                )

            if adapt_seq_len:
                break
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    tokenizer: Tokenizer,
    is_finite: bool,
    adapt_seq_len: bool = False,
    continuation: float = 0.0,
    hybrid_task: HybridTask | None = None,
) -> Iterator[SequenceEmbedMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: list[int] = []
    y_buffer: list[int] = []
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]] = []
    mask_buffer: Mask = []
    sizes: list[int] = []
    n_prefixes: list[int] = []
    n_missing = seq_len
    useless_embed_continuation = False
    for sample in ds_it:
        # Ensure that all batches have the same type to avoid gradient gathering errors
        if hybrid_task is None or not hybrid_task.do:
            rand_continue = np.random.rand()
            if (is_finite and continuation > 0) or continuation >= 1.0:
                do_continuation = True
            elif continuation == 0.0:
                do_continuation = False
            else:
                do_continuation = rand_continue < continuation

            if do_continuation:
                res = sequence_iterator_continuation(
                    sample=sample,
                    x_buffer=x_buffer,
                    y_buffer=y_buffer,
                    mask_buffer=mask_buffer,
                    to_embed_buffer=to_embed_buffer,
                    sizes=sizes,
                    seq_len=seq_len,
                    tokenizer=tokenizer,
                    adapt_seq_len=adapt_seq_len,
                    n_missing=n_missing,
                    data_type="continuation",
                )
                if isinstance(res, SequenceEmbedMaskAndSizes):
                    yield res

                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    sizes = []
                    n_missing = seq_len
                else:
                    (
                        x_buffer,
                        y_buffer,
                        to_embed_buffer,
                        mask_buffer,
                        n_missing,
                        sizes,
                    ) = res
                    continue
            else:
                res = sequence_iterator_reconstruction(
                    sample=sample,
                    x_buffer=x_buffer,
                    y_buffer=y_buffer,
                    mask_buffer=mask_buffer,
                    to_embed_buffer=to_embed_buffer,
                    sizes=sizes,
                    seq_len=seq_len,
                    tokenizer=tokenizer,
                    adapt_seq_len=adapt_seq_len,
                    n_missing=n_missing,
                )

                if isinstance(res, SequenceEmbedMaskAndSizes):
                    yield res

                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    sizes = []
                    n_missing = seq_len
                else:
                    (
                        x_buffer,
                        y_buffer,
                        to_embed_buffer,
                        mask_buffer,
                        n_missing,
                        sizes,
                    ) = res
                    continue
        else:
            if not hybrid_task.one_task_4_all:
                assert adapt_seq_len, "Hybrid task only works with adapt_seq_len=True"
                res = sequence_iterator_hybrid(
                    sample=sample,
                    x_buffer=x_buffer,
                    y_buffer=y_buffer,
                    mask_buffer=mask_buffer,
                    to_embed_buffer=to_embed_buffer,
                    n_prefixes=n_prefixes,
                    sizes=sizes,
                    seq_len=seq_len,
                    tokenizer=tokenizer,
                    max_n_prefixes=hybrid_task.max_n_prefixes,
                    min_n_prefixes=hybrid_task.min_n_prefixes,
                    prop_continuation=hybrid_task.prop_continuation,
                    prop_uselessembed_continuation = hybrid_task.prop_uselessembed_continuation,
                    useless_embed_continuation = useless_embed_continuation,
                    
                )

                if isinstance(res, SequenceEmbedMaskAndSizes):
                    yield res
                    n_prefixes = []
                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    sizes = []
                    useless_embed_continuation = False
                elif res is None:
                    useless_embed_continuation = False
                    continue
                else:
                    to_embed_buffer, useless_embed_continuation = res
                    continue
            else:
                res = sequence_iterator_one_task_4_all(
                    sample=sample,
                    x_buffer=x_buffer,
                    y_buffer=y_buffer,
                    mask_buffer=mask_buffer,
                    to_embed_buffer=to_embed_buffer,
                    sizes=sizes,
                    seq_len=seq_len,
                    tokenizer=tokenizer,
                    n_prefixes = n_prefixes,
                    max_embeds=hybrid_task.max_embeds,
                )

                if isinstance(res, SequenceEmbedMaskAndSizes):
                    yield res
                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    sizes = []
                    n_prefixes = []
                else:
                    continue


    if is_finite:
        # if dataloader is in eval, pad to seq length
        if any(mask_buffer):
            mask_buffer.extend(n_missing * [False])
            x_buffer.extend(n_missing * [0])
            y_buffer.extend(n_missing * [0])
            sizes.append(n_missing)
            to_embed_buffer.append({"text": "", "tokens": []})

            yield SequenceEmbedMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                to_embed=to_embed_buffer,
                mask=mask_buffer,
                sizes=sizes,
                data_type=(
                    "continuation"
                    if int(continuation) == 1 or isinstance(continuation, float)
                    else "reconstruction"
                ),
                n_prefixes = n_prefixes,
            )


def build_dataset(
    args: DataArgs,
    tokenizer: Tokenizer,
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    hybrid_task: HybridTask | None,
    seed: int | None = None,
    shuffle: bool = False,
    continuation: float = 0.0,
) -> Iterator[SequenceEmbedMaskAndSizes]:

    data = args.train_data if not is_eval else args.eval_data
    sources, probabilities = parse_data_sources(data)

    dataset_iterators = [
        get_dataset_iterator(
            source=source,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=not is_eval and shuffle,
        )
        for source in sources
    ]

    # Possible to iterate on zip(...,args.data_types) to force data types from config
    sequence_iterators = [
        sequence_iterator(
            ds_it=it,
            seq_len=seq_len,
            is_finite=is_eval,
            tokenizer=tokenizer,
            adapt_seq_len=args.adapt_seq_len,
            continuation=continuation,
            hybrid_task=hybrid_task,
        )
        for it in dataset_iterators
    ]

    # make sure random_seed is different per rank and original seed
    if not is_eval:
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            sequence_iterators, probabilities=probabilities, rng=rng
        )
    else:
        combined_iterator = itertools.chain.from_iterable(sequence_iterators)

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng


def get_dataset_iterator(
    source: DataDir | DataFile,
    rank: int,
    world_size: int,
    is_finite: bool,
    shuffle_at_epoch: bool,
    tokenizer: Tokenizer,
    seed: int | None = None,
) -> Iterator[TokenSample]:
    jsonl_files = source.jsonl_files
    rng: np.random.RandomState | None = (
        get_rng(seed, rank) if seed is not None else None
    )

    if not is_finite:
        # train mode
        while True:
            for jsonl_file in jsonl_files:
                if shuffle_at_epoch:
                    assert rng is not None, "`seed` has to be passed when shuffling"
                    # will preload all data into RAM, shuffle and yield
                    yield from preload_and_yield(
                        jsonl_file,
                        rank=rank,
                        world_size=world_size,
                        rng=rng,
                        tokenizer=tokenizer,
                    )
                else:
                    # will read data on-the-fly and yield
                    main_logger_info(f"Lazily loading {jsonl_file} ...")
                    yield from lazy_load_and_yield(
                        jsonl_file,
                        rank=rank,
                        world_size=world_size,
                        tokenizer=tokenizer,
                    )
    else:
        # eval mode
        for jsonl_file in jsonl_files:
            # No need to shuffle for eval
            yield from lazy_load_and_yield(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                tokenizer=tokenizer,
            )


def preload_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    rng: np.random.RandomState,
    tokenizer: Tokenizer | None = None,
) -> Iterator[TokenSample] | Iterator[str]:
    # only instruct data has to be chunked
    # load dataset if not already loaded. Make sure to only load 1/world_size dataset
    data_list = maybe_load_local_dataset(
        jsonl_file,
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
    )

    main_logger_info(f"Shuffling {jsonl_file} ...")
    rng.shuffle(data_list)  # type: ignore

    for data_sample in data_list:
        yield data_sample


def lazy_load_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    tokenizer: Tokenizer | None = None,
):
    with jsonl_file.open() as file_handle:
        for idx, line in enumerate(file_handle):
            if not idx % world_size == rank:
                continue

            data = json.loads(line)
            
            if "rand" in data.keys() and float(data["rand"]) >= 0.8:
                continue
            
            yield encode(
                data,
                tokenizer=tokenizer,
                data_path=str(jsonl_file),
            )


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
