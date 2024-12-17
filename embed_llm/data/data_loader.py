import dataclasses
from typing import Iterator
import numpy as np

from embed_llm.data.args import DataArgs
from embed_llm.data.dataset import build_dataset
from embed_llm.data.tokenize import Tokenizer


@dataclasses.dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    texts: list[str]
    sizes: list[int]
    y_mask: np.ndarray | None = None
    is_pad_only: bool = False

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert isinstance(self.sizes, list)
        assert isinstance(self.texts, list)
        assert sum(self.sizes) == self.x.size == self.y.size
        assert len(self.texts) == len(self.sizes)

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
            self.texts = [""] * len(self.sizes)


@dataclasses.dataclass
class Batchlist:
    x: list[list[int]] = dataclasses.field(default_factory=list)
    y: list[list[int]] = dataclasses.field(default_factory=list)
    texts: list[list[str]] = dataclasses.field(default_factory=list)
    sizes: list[list[int]] = dataclasses.field(default_factory=list)
    y_mask: list[list[bool]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        assert self.x == [], "`Batchlist` has to be empty at init."
        assert self.y == [], "`Batchlist` has to be empty at init."
        assert self.texts == [], "`Batchlist` has to be empty at init."
        assert self.sizes == [], "`Batchlist` has to be empty at init."
        assert self.y_mask == [], "`Batchlist` has to be empty at init."

    def __len__(self) -> int:
        return len(self.x)

    def add(
        self,
        x: list[int],
        y: list[int],
        texts: list[str],
        sizes: list[int],
        y_mask: list[bool],
    ):
        self.x.append(x)
        self.y.append(y)
        self.texts.append(texts)
        self.sizes.append(sizes)
        self.y_mask.append(y_mask)

    def empty(self):
        self.x = []
        self.y = []
        self.texts = []
        self.sizes = []
        self.y_mask = []

    @staticmethod
    def flatten_to_numpy(list_of_lists: list[list[object]], dtype: type) -> np.ndarray:
        return np.array(
            [el for sublist in list_of_lists for el in sublist], dtype=dtype
        )

    def create_batch(self) -> Batch:
        x_np: np.ndarray = self.flatten_to_numpy(self.x, dtype=np.int64)
        y_np: np.ndarray = self.flatten_to_numpy(self.y, dtype=np.int64)
        sizes = sum(self.sizes, [])  # noqa
        texts = sum(self.texts, [])  # noqa

        y_mask_flatten = self.flatten_to_numpy(self.y_mask, dtype=bool)
        y_mask_np: np.ndarray | None = None if y_mask_flatten.all() else y_mask_flatten

        return Batch(x_np, y_np, texts, sizes, y_mask_np)


def build_data_loader(
    tokenizer: Tokenizer,
    args: DataArgs,
    batch_size: int,
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    continuation: bool = False,
) -> Iterator[Batch]:
    data = args.train_data if not is_eval else args.eval_data

    dataset = build_dataset(
        pretrain_data=data,
        tokenizer=tokenizer,
        seq_len=seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle=args.shuffle,
        continuation=continuation
    )

    batch_list = Batchlist()
    for sample in dataset:
        assert all(s >= 0 for s in sample.sizes)

        batch_list.add(sample.x, sample.y, sample.texts, sample.sizes, sample.mask)

        if len(batch_list) == batch_size:
            batch: Batch = batch_list.create_batch()
            yield batch

            batch_list.empty()
