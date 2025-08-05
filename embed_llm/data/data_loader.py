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
    to_embed: list[dict[list[str], list[list[int]]]] # A batch is a list of dicts within each dict: tokens and text, tokens are a list of lists 
    sizes: list[int]
    batch_size: int
    y_mask: np.ndarray | None = None
    is_pad_only: bool = False
    data_type: str = "reconstruction"
    insert_embed_list: list[list[int]] | None = None # List of lists, each list contains the indices where to insert embeddings tokens in the text stream (x and y)

    def __post_init__(self):
        assert self.x.ndim == 1
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert self.insert_embed_list is None or len(self.insert_embed_list) == len(
            self.sizes
        ), f"{self.insert_embed_list}, {self.sizes}"
        assert isinstance(self.sizes, list)
        assert isinstance(self.to_embed, list)
        assert sum(self.sizes) == self.x.size == self.y.size, (
            f"{self.sizes}, {self.x.shape}, {self.y.shape}"
        )
        assert len(self.to_embed) == len(self.sizes), (
            f"{len(self.to_embed)}, {len(self.sizes)}"
        )

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


@dataclasses.dataclass
class Batchlist:
    x: list[list[int]] = dataclasses.field(default_factory=list)
    y: list[list[int]] = dataclasses.field(default_factory=list)
    to_embed: list[list[dict[list[str], list[list[int]]]]] = dataclasses.field(
        default_factory=list
    )
    insert_embed_list: list[list[list[int]]] | None = None
    sizes: list[list[int]] = dataclasses.field(default_factory=list)
    y_mask: list[list[bool]] = dataclasses.field(default_factory=list)
    data_type: str = None

    def __post_init__(self):
        assert self.x == [], "`Batchlist` has to be empty at init."
        assert self.y == [], "`Batchlist` has to be empty at init."
        assert self.to_embed == [], "`Batchlist` has to be empty at init."
        assert self.insert_embed_list is None or self.insert_embed_list == [], (
            "`Batchlist` has to be empty at init."
        )
        assert self.sizes == [], "`Batchlist` has to be empty at init."
        assert self.y_mask == [], "`Batchlist` has to be empty at init."

    def __len__(self) -> int:
        return len(self.x)

    def add(
        self,
        x: list[int],
        y: list[int],
        to_embed: list[list[dict[list[str], list[list[int]]]]],
        sizes: list[int],
        y_mask: list[bool],
        data_type: str,
        insert_embed_list: list[list[int]] | None = None,
    ):
        self.x.append(x)
        self.y.append(y)
        self.to_embed.append(to_embed)
        self.sizes.append(sizes)
        self.y_mask.append(y_mask)

        if self.data_type is None:
            self.data_type = data_type

        if insert_embed_list is not None:
            if self.insert_embed_list is None:
                self.insert_embed_list = []
            self.insert_embed_list.append(insert_embed_list)

        assert self.data_type == data_type

    def empty(self):
        self.x = []
        self.y = []
        self.to_embed = []
        self.insert_embed_list = None
        self.sizes = []
        self.y_mask = []
        self.data_type = None

    @staticmethod
    def flatten_to_numpy(list_of_lists: list[list[object]], dtype) -> np.ndarray:
        return np.array(
            [el for sublist in list_of_lists for el in sublist], dtype=dtype
        )

    def create_batch(self, batch_size: int) -> Batch:
        x_np: np.ndarray = self.flatten_to_numpy(self.x, dtype=np.int64)
        y_np: np.ndarray = self.flatten_to_numpy(self.y, dtype=np.int64)
        sizes = sum(self.sizes, [])  # noqa
        if self.insert_embed_list is not None:
            insert_embed_list = sum(self.insert_embed_list, [])  # noqa
        else:
            insert_embed_list = None
        to_embed = sum(self.to_embed, [])  # noqa

        y_mask_flatten = self.flatten_to_numpy(self.y_mask, dtype=bool)
        y_mask_np: np.ndarray | None = None if y_mask_flatten.all() else y_mask_flatten

        return Batch(
            x_np,
            y_np,
            to_embed,
            sizes,
            batch_size=batch_size,
            y_mask=y_mask_np,
            data_type=self.data_type,
            insert_embed_list=insert_embed_list,
        )


def build_data_loader(
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    args: DataArgs,
    batch_size: int,
    seq_len: int,
    rank: int,
    world_size: int,
    is_eval: bool,
    seed: int | None = None,
    continuation: float = 0.0,
    max_embeds: int = 1,
) -> Iterator[Batch]:
    dataset = build_dataset(
        args=args,
        llm_tokenizer=llm_tokenizer,
        embed_tokenizer=embed_tokenizer,
        seq_len=seq_len,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        continuation=continuation,
        max_embeds=max_embeds,
    )

    batch_list_dict = {}
    for sample in dataset:
        assert all(s >= 0 for s in sample.sizes)

        # Avoid empty samples
        if any(
            [
                len(l_toks) <= 1
                for embed in sample.to_embed
                for l_toks in embed["tokens"]
            ]
        ) or any([s == 0 for s in sample.sizes]):
            continue

        # Store in different batch lists based on data type
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
            insert_embed_list=sample.insert_embed_list,
        )

        if len(batch_list) == batch_size:
            batch: Batch = batch_list.create_batch(batch_size)
            yield batch

            batch_list.empty()
            batch_list_dict[sample.data_type] = batch_list
