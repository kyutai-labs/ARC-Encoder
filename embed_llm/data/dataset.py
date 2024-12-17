import dataclasses
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import torch.distributed as dist
from embed_llm.training.distributed import get_rank

from embed_llm.data.tokenize import Mask, TokenSample, Sequence, encode, Tokenizer

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
    path: Path, rank: int, world_size: int, tokenizer: Tokenizer | None = None
) -> list[TokenSample]:
    global _LOADED_DATASETS

    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    data_list: list[TokenSample] = []
    for line in lines:
        data = json.loads(line)

        data_sample: TokenSample = encode(
            data,
            tokenizer=tokenizer,
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
class SequenceTextMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: list[int]
    y: list[int]
    texts: list[str]
    mask: Mask
    sizes: list[int]

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask)
        assert len(self.texts) == len(self.sizes)


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    tokenizer: Tokenizer,
    is_finite: bool,
) -> Iterator[SequenceTextMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: list[int] = []
    y_buffer: list[int] = []
    text_buffer: list[str] = []
    mask_buffer: Mask = []

    sizes: list[int] = []
    n_missing = seq_len
    for sample in ds_it:
        assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
        assert n_missing == seq_len - len(
            x_buffer
        ), f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"

        tokens, mask = sample.tokens, sample.masks[1:]
        x, y = tokens[:-1], tokens[1:]
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
            text_buffer.append(
                tokenizer.decode([x[cur_pos]] + y[cur_pos : cur_pos + n_missing])
            )
            mask_buffer.extend(curr_mask)
            n_missing -= size

            sizes.append(size)

            cur_pos += size

            if n_missing == 0:
                assert len(mask_buffer) == len(x_buffer) == seq_len == len(y_buffer)
                assert sum(sizes) == seq_len
                assert len(text_buffer) == len(sizes)
                # we don't want to yield sequences with a mask filled with False
                if any(mask_buffer):
                    yield SequenceTextMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        texts=text_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                    )
                x_buffer, y_buffer = [], []
                mask_buffer = []
                text_buffer = []
                sizes = []
                n_missing = seq_len

    if is_finite:
        # if dataloader is in eval, pad to seq length
        if any(mask_buffer):
            mask_buffer.extend(n_missing * [False])
            x_buffer.extend(n_missing * [0])
            y_buffer.extend(n_missing * [0])
            sizes.append(n_missing)
            text_buffer.append("")

            yield SequenceTextMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                texts=text_buffer,
                mask=mask_buffer,
                sizes=sizes,
            )


def sequence_iterator_for_continuation(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    tokenizer: Tokenizer,
    is_finite: bool,
) -> Iterator[SequenceTextMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: list[int] = []
    y_buffer: list[int] = []
    text_buffer: list[str] = []
    mask_buffer: Mask = []

    sizes: list[int] = []
    embed_tokens_size: list[int] = []
    n_missing = seq_len
    for sample in ds_it:
        assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
        assert n_missing == seq_len - len(x_buffer) - sum(
            embed_tokens_size
        ), f"n_missing: {n_missing} | seq_len - len(x_buffer) - len(tokens as text) {seq_len - len(x_buffer) - sum(embed_tokens_size)}"

        tokens, mask = sample.tokens, sample.masks[1:]
        x, y = tokens[:-1], tokens[1:]
        cur_pos = 0

        while cur_pos < len(x):
            overall_size = len(x[cur_pos : cur_pos + n_missing])

            curr_mask = mask[cur_pos : cur_pos + n_missing]
            if not any(curr_mask):
                cur_pos += overall_size
                # we have a sequence with a mask filled with False
                continue

            upper_bound = min(cur_pos + n_missing, len(x))

            toks = x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound]
            x_buffer.extend(toks)
            y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            mask_buffer.extend(
                mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound]
            )
            text_buffer.append(
                tokenizer.decode(x[cur_pos : cur_pos + (upper_bound - cur_pos) // 2])
            )

            sizes.append(len(toks))
            embed_tokens_size.append(
                len(x[cur_pos : cur_pos + (upper_bound - cur_pos) // 2])
            )

            n_missing -= overall_size
            cur_pos += overall_size

            if n_missing == 0:
                assert len(mask_buffer) == len(x_buffer) == sum(sizes) == len(y_buffer)
                assert sum(sizes) + sum(embed_tokens_size) == seq_len
                assert len(text_buffer) == len(sizes)
                # we don't want to yield sequences with a mask filled with False
                if any(mask_buffer):
                    yield SequenceTextMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        texts=text_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                    )
                x_buffer, y_buffer = [], []
                mask_buffer = []
                text_buffer = []
                sizes = []
                embed_tokens_size = []
                n_missing = seq_len

    if is_finite:
        # if dataloader is in eval, pad to seq length
        if any(mask_buffer):
            mask_buffer.extend(n_missing * [False])
            x_buffer.extend(n_missing * [0])
            y_buffer.extend(n_missing * [0])
            sizes.append(n_missing)
            text_buffer.append("")

            yield SequenceTextMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                texts=text_buffer,
                mask=mask_buffer,
                sizes=sizes,
            )


def build_dataset(
    pretrain_data: str,
    tokenizer: Tokenizer,
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    shuffle: bool = False,
    continuation: bool = False,
) -> Iterator[SequenceTextMaskAndSizes]:
    sources, probabilities = parse_data_sources(pretrain_data)

    dataset_iterators = [
        get_dataset_iterator(
            source,
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=not is_eval and shuffle,
        )
        for source in sources
    ]

    if not continuation:
        sequence_iterators = [
            sequence_iterator(
                ds_it=it,
                seq_len=seq_len,
                is_finite=is_eval,
                tokenizer=tokenizer,
            )
            for it in dataset_iterators
        ]
    else:
        sequence_iterators = [
            sequence_iterator_for_continuation(
                ds_it=it,
                seq_len=seq_len,
                is_finite=is_eval,
                tokenizer=tokenizer,
            )
            for it in dataset_iterators
        ]

    if is_eval:
        combined_iterator = itertools.chain.from_iterable(sequence_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            sequence_iterators, probabilities=probabilities, rng=rng
        )

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
        jsonl_file, rank=rank, world_size=world_size, tokenizer=tokenizer
    )

    main_logger_info(f"Shuffling {jsonl_file} ...")
    rng.shuffle(tokens_list)  # type: ignore

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
            yield encode(
                data,
                tokenizer=tokenizer,
            )


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
