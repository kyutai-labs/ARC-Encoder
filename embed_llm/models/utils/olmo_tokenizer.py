from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from tokenizers import Tokenizer as BaseTokenizer


__all__ = ["Tokenizer"]


class Tokenizer:
    """
    A :class:`Tokenizer` is a light-weight wrapper around a HuggingFace :class:`tokenizers.Tokenizer`.

    :param base_tokenizer: The :class:`tokenizers.Tokenizer` to use.
    :param eos_id: The token ID corresponding to the "end-of-sentence" token.
    :param truncate_to: Truncate when tokenizing to this number of token IDs.
    :param truncate_direction: The direction to truncate in. "right" means truncate the tokens
        on the right. "left" means truncate the tokens on the left. If ``truncate_to`` is null,
        this setting has no effect.
    """

    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        eos_token_id: int,
        pad_token_id: Optional[int] = None,
    ):
        self.base_tokenizer = base_tokenizer
        self.eos_id = eos_token_id
        self.bos_id = pad_token_id if pad_token_id is not None else 1

    @property
    def vocab_size(self) -> int:
        return self.base_tokenizer.get_vocab_size()

    @property
    def eos_token(self) -> str:
        return self.decode([self.eos_id], skip_special_tokens=False)

    @property
    def pad_token(self) -> str:
        return self.decode([self.bos_id], skip_special_tokens=False)

    @classmethod
    def from_pretrained(cls, identifier: str, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a pretrained tokenizer on the HuggingFace Hub.

        :param identifier: The identifier of a model on the Hub that contains a
            ``tokenizer.json`` file.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_pretrained(identifier)
        eos_id = kwargs.pop("eos_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_id, **kwargs)
    
    @classmethod
    def from_file(cls, filename: Path | str, **kwargs) -> Tokenizer:
        """
        Initialize a tokenizer from a file.

        You can create those files with ``BaseTokenizer.save()``.

        :param filename: The name of a file containing a tokenizer specification.
        :param kwargs: Other key word arguments passed to :class:`Tokenizer`.
        """
        base_tokenizer = BaseTokenizer.from_file(str(filename))
        eos_id = kwargs.pop("eos_id", base_tokenizer.get_vocab_size() - 1)
        return cls(base_tokenizer, eos_id, **kwargs)

    def add_special_tokens(self, input_ids: List[int], eos: bool = False, bos: bool = False) -> List[int]:
        """
        Add special tokens in-place (if not already present) to the given token IDs.
        """
        if eos and (not input_ids or input_ids[-1] != self.eos_id):
            input_ids.append(self.eos_id)
        # if bos and (not input_ids or input_ids[0] != self.bos_id):
        #     input_ids.insert(0, self.bos_id)
        return input_ids

    def num_special_tokens_to_add(self, is_pair: bool = False) -> int:
        return 2 if is_pair else 1


    def encode(self, input: str, eos = False, bos = False) -> List[int]:
        """
        Encode a string into token IDs.
        """
        return self.encode_batch([input], eos=eos, bos=bos)[0]

    def encode_batch(self, inputs: List[str], eos = False, bos = False) -> List[List[int]]:
        """
        Encode a batch of strings into token IDs.
        """
 

        batch_encoding = self.base_tokenizer.encode_batch(inputs)

        all_input_ids = []
        for encoding in batch_encoding:
            input_ids = encoding.ids
            input_ids = self.add_special_tokens(input_ids, eos=eos, bos=bos)
            all_input_ids.append(input_ids)

        return all_input_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs to a string.
        """
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)