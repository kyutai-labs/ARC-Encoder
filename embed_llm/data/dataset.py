import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import torch.distributed as dist
from embed_llm.training.distributed import get_rank
from embed_llm.data.args import DataArgs
from embed_llm.data.tokenize import Mask, TokenSample, encode, Tokenizer
from embed_llm.data.sequence_iterators import (
    sequence_iterator_continuation,
    sequence_iterator_reconstruction,
    sequence_iterator_inserted_embed_continuation,
    SequenceEmbedMaskAndSizes,
)

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
    tokenizer: Tokenizer | None = None,  # type: ignore
    max_embeds: int = 1,
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
            data, tokenizer=tokenizer, data_path=str(path), max_embed=max_embeds
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
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
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
                f"{source} is not correctly formatted. Make sure to format each data source as \
                    <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        assert weight > 0, (
            f"Make sure to define strictly positive data sampling weights, not {weight}"
        )

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or \
                    directory that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    tokenizer: Tokenizer,  # type: ignore
    is_finite: bool,
    adapt_seq_len: bool = False,
    continuation: float = 0.0,
    insert_embeddings: bool = False,
) -> Iterator[SequenceEmbedMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """

    x_buffer: list[int] = []
    y_buffer: list[int] = []
    to_embed_buffer: list[dict[str, str | int]] = []
    mask_buffer: Mask = []
    sizes: list[int] = []
    n_missing_cont = seq_len * 2

    x_buffer_cont: list[int] = []
    y_buffer_cont: list[int] = []
    insert_embed_cont_list: list[list[int]] = []
    to_embed_buffer_cont: list[dict[str, str | int]] = []
    mask_buffer_cont: Mask = []
    sizes_cont: list[int] = []

    cur_pos = 0
    n_missing = seq_len
    for sample in ds_it:
        # Ensure that all batches have the same type to avoid gradient gathering errors

        rand_continue = np.random.rand()
        if (is_finite and continuation > 0) or continuation >= 1.0:
            do_continuation = True
        elif continuation == 0.0:
            do_continuation = False
        else:
            do_continuation = rand_continue < continuation

        if do_continuation:
            if insert_embeddings and not is_finite:
                while True:
                    res = sequence_iterator_inserted_embed_continuation(
                        sample=sample,
                        x_buffer=x_buffer_cont,
                        y_buffer=y_buffer_cont,
                        mask_buffer=mask_buffer_cont,
                        to_embed_buffer=to_embed_buffer_cont,
                        insert_embed_list=insert_embed_cont_list,
                        sizes=sizes_cont,
                        seq_len=seq_len,
                        tokenizer=tokenizer,
                        n_missing=n_missing_cont,
                        data_type="continuation",
                        cur_pos=cur_pos,
                    )

                    if len(res) == 2 and isinstance(res[0], SequenceEmbedMaskAndSizes):
                        yield res[0]

                        x_buffer_cont, y_buffer_cont = [], []
                        mask_buffer_cont = []
                        to_embed_buffer_cont = []
                        insert_embed_cont_list = []
                        sizes_cont = []
                        n_missing_cont = seq_len * 3
                        cur_pos = res[1]
                    else:
                        (
                            x_buffer_cont,
                            y_buffer_cont,
                            to_embed_buffer_cont,
                            insert_embed_cont_list,
                            mask_buffer_cont,
                            n_missing_cont,
                            sizes_cont,
                        ) = res
                        cur_pos = 0
                        break
            else:
                while True:
                    res = sequence_iterator_continuation(
                        sample=sample,
                        x_buffer=x_buffer_cont,
                        y_buffer=y_buffer_cont,
                        mask_buffer=mask_buffer_cont,
                        to_embed_buffer=to_embed_buffer_cont,
                        sizes=sizes_cont,
                        seq_len=seq_len
                        * 2,  # To ensure max seq len to generate and max seq len to embed
                        tokenizer=tokenizer,
                        n_missing=n_missing_cont,
                        data_type="continuation",
                        cur_pos=cur_pos,
                    )

                    if len(res) == 2 and isinstance(res[0], SequenceEmbedMaskAndSizes):
                        yield res[0]

                        x_buffer_cont, y_buffer_cont = [], []
                        mask_buffer_cont = []
                        to_embed_buffer_cont = []
                        sizes_cont = []
                        n_missing_cont = seq_len * 2
                        cur_pos = res[1]
                    else:
                        (
                            x_buffer_cont,
                            y_buffer_cont,
                            to_embed_buffer_cont,
                            mask_buffer_cont,
                            n_missing_cont,
                            sizes_cont,
                        ) = res
                        cur_pos = 0
                        break
        else:
            while True:
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
                    cur_pos=cur_pos,
                )

                if len(res) == 2 and isinstance(res[0], SequenceEmbedMaskAndSizes):
                    yield res[0]

                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    sizes = []
                    n_missing = seq_len
                    cur_pos = res[1]
                else:
                    (
                        x_buffer,
                        y_buffer,
                        to_embed_buffer,
                        mask_buffer,
                        n_missing,
                        sizes,
                    ) = res
                    cur_pos = 0
                    break

    if is_finite:
        # if dataloader is in eval, pad to seq length
        if any(mask_buffer):
            mask_buffer.extend(n_missing * [False])
            x_buffer.extend(n_missing * [0])
            y_buffer.extend(n_missing * [0])
            sizes.append(n_missing)
            to_embed_buffer.append({"text": "", "tokens": []})
            if len(insert_embed_cont_list) > 0:
                insert_embed_cont_list.append([0])

            yield SequenceEmbedMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                to_embed=to_embed_buffer,
                mask=mask_buffer,
                sizes=sizes,
                insert_embed_list=insert_embed_cont_list
                if int(continuation) == 1 or isinstance(continuation, float)
                else [],
                data_type=(
                    "continuation"
                    if int(continuation) == 1 or isinstance(continuation, float)
                    else "reconstruction"
                ),
            )


def build_dataset(
    args: DataArgs,
    tokenizer: Tokenizer,  # type: ignore
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    shuffle: bool = False,
    continuation: float = 0.0,
    max_embeds: int = 1,
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
            max_embeds=max_embeds,
        )
        for source in sources
    ]

    sequence_iterators = [
        sequence_iterator(
            ds_it=it,
            seq_len=seq_len,
            is_finite=is_eval,
            tokenizer=tokenizer,
            adapt_seq_len=args.adapt_seq_len,
            continuation=continuation,
            insert_embeddings=args.insert_embeddings,
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
    tokenizer: Tokenizer,  # type: ignore
    seed: int | None = None,
    max_embeds: int = 1,
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
                        max_embeds=max_embeds,
                    )
                else:
                    # will read data on-the-fly and yield
                    main_logger_info(f"Lazily loading {jsonl_file} ...")
                    yield from lazy_load_and_yield(
                        jsonl_file,
                        rank=rank,
                        world_size=world_size,
                        tokenizer=tokenizer,
                        max_embeds=max_embeds,
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
                max_embeds=max_embeds,
            )


def preload_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    rng: np.random.RandomState,
    tokenizer: Tokenizer | None = None,  # type: ignore
    max_embeds: int = 1,
) -> Iterator[TokenSample] | Iterator[str]:
    # only instruct data has to be chunked
    # load dataset if not already loaded. Make sure to only load 1/world_size dataset
    data_list = maybe_load_local_dataset(
        jsonl_file,
        rank=rank,
        world_size=world_size,
        tokenizer=tokenizer,
        max_embeds=max_embeds,
    )

    main_logger_info(f"Shuffling {jsonl_file} ...")
    rng.shuffle(data_list)  # type: ignore

    for data_sample in data_list:
        yield data_sample


def lazy_load_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    tokenizer: Tokenizer | None = None,  # type: ignore
    max_embeds: int = 1,
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
                max_embed=max_embeds,
            )


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
