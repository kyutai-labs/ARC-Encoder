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
    rec_seq_len_factor: float = 1.0  # If > 1.0, the seqlen will be increased for reconstruction and it will shorten continuation (fixed seqlen for embedding but shorter text to continue)
