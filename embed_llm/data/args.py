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
        ""
        # Each line in the jsonl files inside the data source directories must be a dictionary with a "text" key for pretraining.
        # For fine-tuning, the dictionary must contain a "question" key, an "answer" key and a "passages" key (list of strings).
        # See Readme for more details. Can be left empty.
    )
    eval_data: str = (
        ""
        # Same as above
    )
    instruct: bool = False  # Precise if the data contains instructions (question, answer, passages) or is for pretraining (text).
    interleave: bool = False  # If True, compressed sequences and text will be interleaved during fine-tuning (in the few-shot samples case)
    # or seq_len text will be inserted before compressed sequence for continuation pretraining.
    prefix_path: str | None = (
        None  # If set, the prefix path will be prepended to each datapath.
    )
    max_passages: int = 1  # Maximum number of passages to use per loaded sample (if several retrieved passages for each samples in the dataset, <= len(data['passages'])).
    n_eval_batches: int = 40  # Number of batches from eval_data to evaluate on (for quick evals during training).
    instruct_decoder: bool = False  # Precise if you use an instruct decoder which requires a precise chat format.
    chunk_to: int | None = (
        None  # Whether to chunk the sequences to a maximum length (for long documents) and process them in parallel.
    )
    max_chunks: int = 5  # Maximum number of chunks to use if split context in several chunks (used only if chunk_to is not None).

    def __post_init__(self) -> None:
        if self.prefix_path is not None:
            self.train_data = ",".join(
                [
                    self.prefix_path + train_path
                    for train_path in self.train_data.split(",")
                ]
            )
            if self.eval_data != "" and not Path(self.eval_data).exists():
                self.eval_data = ",".join(
                    [
                        self.prefix_path + eval_path
                        for eval_path in self.eval_data.split(",")
                    ]
                )
