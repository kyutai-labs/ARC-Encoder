import dataclasses
from typing import Iterator
import numpy as np

from embed_llm.training.args import HybridTask
from embed_llm.data.args import DataArgs
from embed_llm.data.dataset import build_dataset
from embed_llm.data.tokenize import Tokenizer


@dataclasses.dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    to_embed: list[dict[str, list[list[int]] | list[str]]]
    sizes: list[int]
    y_mask: np.ndarray | None = None
    is_pad_only: bool = False
    data_type: str = "reconstruction"
    n_prefixes: list[int] | None = None

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert isinstance(self.sizes, list)
        assert isinstance(self.to_embed, list)
        assert sum(self.sizes) == self.x.size == self.y.size
        assert len(self.to_embed) == len(self.sizes)

        if self.y_mask is not None:
            assert self.y_mask.size == self.y.size, (self.y_mask.shape, self.y.shape)
            assert self.y_mask.dtype == bool
            assert sum(self.sizes) == self.y_mask.size
            assert not self.y_mask.all()
            assert self.y_mask.any()

        if self.is_pad_only:
            assert np.sum(np.abs(self.y)) == 0
            assert np.sum(np.abs(self.x)) == 0
            assert self.y_mask is None
            # create all 0's mask for pad samples
            self.y_mask = np.zeros_like(self.x)
            self.to_embed = [{"text": [""], "tokens": [[0]]} for _ in self.to_embed]
        if self.n_prefixes is not None:
            assert len(self.n_prefixes) == len(self.to_embed)


@dataclasses.dataclass
class Batchlist:
    x: list[list[int]] = dataclasses.field(default_factory=list)
    y: list[list[int]] = dataclasses.field(default_factory=list)
    to_embed: list[list[dict[str, list[list[int]] | list[str]]]] = dataclasses.field(
        default_factory=list
    )
    sizes: list[list[int]] = dataclasses.field(default_factory=list)
    y_mask: list[list[bool]] = dataclasses.field(default_factory=list)
    data_type: str = None
    n_prefixes: list[list[int]] | None = None

    def __post_init__(self):
        assert self.x == [], "`Batchlist` has to be empty at init."
        assert self.y == [], "`Batchlist` has to be empty at init."
        assert self.to_embed == [], "`Batchlist` has to be empty at init."
        assert self.sizes == [], "`Batchlist` has to be empty at init."
        assert self.y_mask == [], "`Batchlist` has to be empty at init."

    def __len__(self) -> int:
        return len(self.x)

    def add(
        self,
        x: list[int],
        y: list[int],
        to_embed: list[dict[str, list[list[int]] | list[str]]],
        sizes: list[int],
        y_mask: list[bool],
        data_type: str,
        n_prefixes: list[int] | None = None,
    ):
        self.x.append(x)
        self.y.append(y)
        self.to_embed.append(to_embed)
        self.sizes.append(sizes)
        self.y_mask.append(y_mask)
        if self.data_type is None:
            self.data_type = data_type

        if n_prefixes is not None:
            if self.n_prefixes is None:
                self.n_prefixes = []
            self.n_prefixes.append(n_prefixes)

        assert self.data_type == data_type

    def empty(self):
        self.x = []
        self.y = []
        self.to_embed = []
        self.sizes = []
        self.y_mask = []
        self.data_type = None
        self.n_prefixes = None

    @staticmethod
    def flatten_to_numpy(list_of_lists: list[list[object]], dtype) -> np.ndarray:
        return np.array(
            [el for sublist in list_of_lists for el in sublist], dtype=dtype
        )

    def create_batch(self) -> Batch:
        x_np: np.ndarray = self.flatten_to_numpy(self.x, dtype=np.int64)
        y_np: np.ndarray = self.flatten_to_numpy(self.y, dtype=np.int64)
        sizes = sum(self.sizes, [])  # noqa
        to_embed = sum(self.to_embed, [])  # noqa

        y_mask_flatten = self.flatten_to_numpy(self.y_mask, dtype=bool)
        y_mask_np: np.ndarray | None = None if y_mask_flatten.all() else y_mask_flatten

        if self.n_prefixes is not None:
            n_prefixes = sum(self.n_prefixes, [])
        else:
            n_prefixes = None

        return Batch(
            x_np,
            y_np,
            to_embed,
            sizes,
            y_mask_np,
            data_type=self.data_type,
            n_prefixes=n_prefixes,
        )


def build_data_loader(
    tokenizer: Tokenizer,
    args: DataArgs,
    batch_size: int,
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    continuation: float = 0.0,
    hybrid_task: HybridTask | None = None,
) -> Iterator[Batch]:

    dataset = build_dataset(
        args=args,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        continuation=continuation,
        hybrid_task=hybrid_task,
    )

    batch_list_dict = {}
    for sample in dataset:
        assert all(s >= 0 for s in sample.sizes)

        # Avoid empty samples
        if any(
            [
                len(l_tokens) <= 1
                for embed in sample.to_embed
                for l_tokens in embed["tokens"]
            ]
        ):
            continue

        if sample.data_type not in batch_list_dict:
            batch_list_dict[sample.data_type] = Batchlist()

        batch_list = batch_list_dict[sample.data_type]

        batch_list.add(
            sample.x,
            sample.y,
            sample.to_embed,
            sample.sizes,
            sample.mask,
            data_type=sample.data_type,
            n_prefixes=getattr(sample, "n_prefixes", None),
        )

        if len(batch_list) == batch_size:

            batch: Batch = batch_list.create_batch()
            yield batch

            batch_list.empty()
            batch_list_dict[sample.data_type] = batch_list
