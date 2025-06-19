import logging
from dataclasses import dataclass

from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    # The data arguments `data` and `instruct_data` are a string in the format
    # "data_source_dir_1:weight_1,data_source_dir_2:weight_2,...". The weight
    # will be used to sample the data sources. If the sum of the weights is
    # not 1 when concatenating the two arguments `data` and `instruct_data`,
    # it will be normalized. The data sources folders must contain jsonl files.
    # If the value is an empty string, no data will be used for the corresponding
    # data type.
    train_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key.
        # See Readme for more details. Can be left empty.
    )
    eval_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key.
        # See Readme for more details. Can be left empty.
    )
    shuffle: bool = False
    adapt_seq_len: bool = False
    n_times_sl_insertion: int = 1
    n_interleaved: int = 1  # Number of interleaved sequences to use for training. If > 0, the data will be interleaved.
    loss_last_cont_only: bool | None = None  # If True, the loss will be computed only on the last continuation token.
    rec_seq_len_factor: float = 1.0  # If > 1.0, the seqlen will be increased for reconstruction and it will shorten continuation (fixed seqlen for embedding but shorter text to continue)
    few_shot: int = 0
    prefix: str | None = None  # If set, the prefix will be prepended to each datapath.

    def __post_init__(self) -> None:
        if self.prefix is not None:
            self.train_data = ",".join(
                [self.prefix + train_path for train_path in self.train_data.split(",")]
            )
            self.eval_data = ",".join(
                [self.prefix + eval_path for eval_path in self.eval_data.split(",")]
            )
        if self.adapt_seq_len:
           self.loss_last_cont_only = self.loss_last_cont_only if self.loss_last_cont_only is not None else True
        else:
            self.loss_last_cont_only = self.loss_last_cont_only if self.loss_last_cont_only is not None else False