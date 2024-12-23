import logging
from dataclasses import dataclass, field

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
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key. See Readme for more details. Can be left empty.
    )
    eval_data: str = (
        ""  # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key. See Readme for more details. Can be left empty.
    )
    shuffle: bool = False
    adapt_seq_len: bool = False
    continuation: bool = False
    data_types: list[str] = field(default_factory=lambda: ["reconstruction"])

    def __post_init__(self) -> None:
        assert len(self.train_data.strip().split(",")) == len(self.data_types), "Number of data sources must match number of types."
        pass
