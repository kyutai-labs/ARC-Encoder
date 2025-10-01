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
    instruct: bool = False
    interleave: bool = False  # Number of interleaved sequences to use for training. If True, the data will be interleaved.
    prefix_path: str | None = None  # If set, the prefix path will be prepended to each datapath.
    sep_passages: bool = False  # If True, passages will be separated by a special token in the input sequence.
    max_passages: int = 1  # Maximum number of passages to use per loaded sample (if several retrieved passages for each samples in the dataset).
    n_eval_batches: int = 40 # To reduce if long context
    instruct_decoder: bool = False  # If True, only the decoder will be instructed (for encoder-decoder models).
    
    chunk_to: int | None = None # Whether to chunk the sequences to a maximum length (for long documents) and process them in parallel.
    max_chunks: int = 5  # Maximum number of chunks to use if split context in several chunks.
    
    def __post_init__(self) -> None:
        if self.prefix_path is not None:
            self.train_data = ",".join(
                [self.prefix_path + train_path for train_path in self.train_data.split(",")]
            )
            if self.eval_data != '' and not Path(self.eval_data).exists(): 
                self.eval_data = ",".join(
                    [self.prefix_path + eval_path for eval_path in self.eval_data.split(",")]
                )
