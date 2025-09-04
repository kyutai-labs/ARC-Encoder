import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch.distributed as dist

from embed_llm.data.args import DataArgs
from embed_llm.data.sequence_iterators import (
    SequenceEmbedMaskAndSizes,
    sequence_iterator_continuation,
    sequence_iterator_reconstruction,
)
from embed_llm.data.tokenize import Mask, Tokenizer, TokenSample, encode
from embed_llm.training.distributed import get_rank

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
    few_shots: list[int] = []
    for source in pretrain_data.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
            few_shot = 0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
            few_shot = 0
        elif len(source_items) == 3:
            path_, weight_, few_shot_ = source_items
            weight = float(weight_)
            few_shot = int(few_shot_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source as \
                    <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        # assert weight > 0, (
        #     f"Make sure to define strictly positive data sampling weights, not {weight}"
        # )
        assert few_shot >= 0, (
            f"Make sure to define non-negative few-shot value, not {few_shot}"
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
        if weight <= 0:
            seen.add(path_)
            continue
        else:
            sources.append(data)
            weights.append(weight)
            few_shots.append(few_shot)
            seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights, few_shots


def sequence_iterator(
    ds_it: Iterator[TokenSample],
    seq_len: int,
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    is_finite: bool,
    adapt_seq_len: bool = False,
    continuation: float = 0.0,
    few_shot: int = 0,
    interleave: bool = False,
    loss_last_cont_only: bool = False,  # If True, the loss will be computed only on the last continuation token.
    sep_passages: bool = False,  # If True, passages will be separated by a special token in the input sequence.
    chunk_to: int | None = None,
    instruct_decoder: bool = False,  # If True, the decoder will be used for instruction data.
) -> Iterator[SequenceEmbedMaskAndSizes]:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    x_buffer: list[int] = []
    y_buffer: list[int] = []
    to_embed_buffer: list[dict[list[str], list[list[int]]]] = []
    insert_embed_list: list[list[int]] = []
    mask_buffer: Mask = []
    sizes: list[int] = []
    n_missing_cont = (
        seq_len * 2 + int(interleave) * seq_len 
    )
    instruct_prompt: list[str] = []
    instruct_prompt_cont: list[str] = []

    x_buffer_cont: list[int] = []
    y_buffer_cont: list[int] = []
    insert_embed_cont_list: list[list[int]] = []
    to_embed_buffer_cont: list[dict[list[str], list[list[int]]]] = []
    mask_buffer_cont: Mask = []
    sizes_cont: list[int] = []

    few_shot_instruct = []
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
            while True:
                res = (
                    sequence_iterator_continuation(
                        sample=sample,
                        x_buffer=x_buffer_cont,
                        y_buffer=y_buffer_cont,
                        mask_buffer=mask_buffer_cont,
                        to_embed_buffer=to_embed_buffer_cont,
                        insert_embed_list=insert_embed_cont_list,
                        sizes=sizes_cont,
                        seq_len=seq_len,
                        llm_tokenizer=llm_tokenizer,
                        embed_tokenizer=embed_tokenizer,
                        n_missing=n_missing_cont,
                        data_type="continuation",
                        cur_pos=cur_pos,
                        interleave=interleave,
                        instruct_prompt=instruct_prompt_cont if instruct_decoder else None,
                    )
                    )
  

                if len(res) == 2 and isinstance(res[0], SequenceEmbedMaskAndSizes):
                    yield res[0]

                    x_buffer_cont, y_buffer_cont = [], []
                    mask_buffer_cont = []
                    to_embed_buffer_cont = []
                    insert_embed_cont_list = []
                    sizes_cont = []
                    instruct_prompt_cont = []
                    n_missing_cont = (
                        seq_len * 2 + int(interleave) * seq_len
                    )  # 2*seq_len for compressed tokens and contionuation, + the ones for text before compressed tokens
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
                        instruct_prompt_cont,
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
                    llm_tokenizer=llm_tokenizer,
                    embed_tokenizer=embed_tokenizer,
                    adapt_seq_len=adapt_seq_len,
                    n_missing=n_missing,
                    cur_pos=cur_pos,
                    insert_embed_list=insert_embed_list,
                    few_shot_instruct=few_shot_instruct if few_shot > 0 else None,
                    few_shot=few_shot,
                    interleave=interleave,
                    loss_last_cont_only=loss_last_cont_only,  # If True, the loss will be computed only on the last continuation token.
                    sep_passages=sep_passages,
                    chunk_to=chunk_to,
                    instruct_prompt=instruct_prompt if instruct_decoder else None,
                )

                if len(res) == 2 and isinstance(res[0], SequenceEmbedMaskAndSizes):
                    yield res[0]

                    x_buffer, y_buffer = [], []
                    mask_buffer = []
                    to_embed_buffer = []
                    insert_embed_list = []
                    sizes = []
                    instruct_prompt = []
                    n_missing = seq_len
                    cur_pos = res[1]
                else:
                    (
                        x_buffer,
                        y_buffer,
                        to_embed_buffer,
                        insert_embed_list,
                        mask_buffer,
                        n_missing,
                        sizes,
                        few_shot_instruct,
                        instruct_prompt,
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
            to_embed_buffer.append({"text": [""], "tokens": [[]]})
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
                instruct_prompt=instruct_prompt if instruct_decoder else None,
            )


def build_dataset(
    args: DataArgs,
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    continuation: float = 0.0,
) -> Iterator[SequenceEmbedMaskAndSizes]:
    data = args.train_data if not is_eval else args.eval_data
    sources, probabilities, few_shots = parse_data_sources(data)

    dataset_iterators = [
        get_dataset_iterator(
            source=source,
            llm_tokenizer=llm_tokenizer,
            embed_tokenizer=embed_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            max_passages=args.max_passages,
            instruct_decoder=args.instruct_decoder,
            instruct=args.instruct,
        )
        for source in sources
    ]

    sequence_iterators = [
        sequence_iterator(
            ds_it=it,
            seq_len=seq_len,
            is_finite=is_eval,
            llm_tokenizer=llm_tokenizer,
            embed_tokenizer=embed_tokenizer,
            adapt_seq_len=args.adapt_seq_len,
            continuation=continuation,
            few_shot=fs,
            interleaved=args.interleave,
            loss_last_cont_only=args.loss_last_cont_only,
            sep_passages=args.sep_passages,
            chunk_to=args.chunk_to,
            instruct_decoder=args.instruct_decoder,
        )
        for fs, it in zip(few_shots, dataset_iterators)
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
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    seed: int | None = None,
    max_passages: int = 1,
    instruct: bool = False,
    instruct_decoder: bool = False,  # If True, the decoder will be used for instruction data
) -> Iterator[TokenSample]:
    jsonl_files = source.jsonl_files
    rng: np.random.RandomState | None = (
        get_rng(seed, rank) if seed is not None else None
    )

    if not is_finite:
        # train mode
        while True:
            for jsonl_file in jsonl_files:
                # will read data on-the-fly and yield
                main_logger_info(f"Lazily loading {jsonl_file} ...")
                yield from lazy_load_and_yield(
                    jsonl_file,
                    rank=rank,
                    world_size=world_size,
                    llm_tokenizer=llm_tokenizer,
                    embed_tokenizer=embed_tokenizer,
                    max_passages=max_passages,
                    instruct_decoder=instruct_decoder,
                    instruct=instruct,
                )
    else:
        # eval mode
        for jsonl_file in jsonl_files:
            yield from lazy_load_and_yield(
                jsonl_file,
                rank=rank,
                world_size=world_size,
                llm_tokenizer=llm_tokenizer,
                embed_tokenizer=embed_tokenizer,
                max_passages=max_passages,
                instruct=instruct,
                instruct_decoder=instruct_decoder,
            )


def lazy_load_and_yield(
    jsonl_file: Path,
    rank: int,
    world_size: int,
    llm_tokenizer: Tokenizer | None = None,  # type: ignore
    embed_tokenizer: Tokenizer | None = None,  # type: ignore
    max_passages: int = 1,
    instruct: bool = False,
    instruct_decoder: bool = False,  # If True, the decoder will be used for instruction data
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
                llm_tokenizer=llm_tokenizer,
                embed_tokenizer=embed_tokenizer,
                max_passages=max_passages,
                instruct=instruct,
                instruct_decoder=instruct_decoder,
            )

def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        try:
            yield next(iterators[it_id])
        except (OSError, StopIteration) as e:
            # If the iterator is exhausted, we remove it from the list
            # and continue with the next one.
            main_logger_info(
                f"Iterator {it_id} exhausted. Removing it from the list."
            )
            del iterators[it_id]
            if not isinstance(probabilities, list):
                probabilities = list(probabilities)
            del probabilities[it_id]
            probabilities = [prob / np.sum(probabilities) for prob in probabilities]  # re-normalize probabilities
            if len(iterators) == 0:
                raise e
            continue
