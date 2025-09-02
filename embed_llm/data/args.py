import logging
from dataclasses import dataclass
from pathlib import Path
from simple_parsing.helpers import Serializable

logger = logging.getLogger("data")


@dataclass()
class DataArgs(Serializable):
    # The data arguments are a string in the format
    # "data_source_dir_1:weight_1,data_source_dir_2:weight_2,...". The weight
    # will be used to sample the data sources. If the sum of the weights is
    # not 1 it will be normalized. The data sources folders must contain jsonl files.
    # If the value is an empty string, no data will be used for the corresponding data type.
    train_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key.
        # See Readme for more details. Can be left empty.
    )
    eval_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key.
        # See Readme for more details. Can be left empty.
    )
    
    adapt_seq_len: bool = False
    interleave: bool = False  # Number of interleaved sequences to use for training. If > 0, the data will be interleaved.
    loss_last_cont_only: bool | None = (
        None  # If True, the loss will be computed only on the last continuation token.
    )
    prefix: str | None = None  # If set, the prefix will be prepended to each datapath.
    sep_passages: bool = False  # If True, passages will be separated by a special token in the input sequence.
    chunk_to: int | None = None
    max_passages: int = 1  # Maximum number of passages to use per loaded sample (if several retrieved passages in the dataset).
    n_eval_batchs: int = 40
    
    def __post_init__(self) -> None:
        if self.prefix is not None:
            self.train_data = ",".join(
                [self.prefix + train_path for train_path in self.train_data.split(",")]
            )
            if self.eval_data != '' and not Path(self.eval_data).exists(): 
                self.eval_data = ",".join(
                    [self.prefix + eval_path for eval_path in self.eval_data.split(",")]
                )
        if self.adapt_seq_len:
            self.loss_last_cont_only = (
                self.loss_last_cont_only
                if self.loss_last_cont_only is not None
                else True
            )
        else:
            self.loss_last_cont_only = (
                self.loss_last_cont_only
                if self.loss_last_cont_only is not None
                else False
            )
