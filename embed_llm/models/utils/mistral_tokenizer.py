import logging
import os
from pathlib import Path
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.sentencepiece import is_sentencepiece
from mistral_common.tokens.tokenizers.tekken import (
    SpecialTokenPolicy,
    Tekkenizer,
    is_tekken,
)


def load_tokenizer(model_path: Path) -> MistralTokenizer:
    tokenizer = [
        f
        for f in os.listdir(model_path)
        if is_tekken(model_path / f) or is_sentencepiece(model_path / f)
    ]
    assert len(tokenizer) > 0, (
        f"No tokenizer in {model_path}, place a `tokenizer.model.[v1,v2,v3]` or `tekken.json` file in {model_path}."
    )
    assert len(tokenizer) == 1, (
        f"Multiple tokenizers {', '.join(tokenizer)} found in `model_path`, make sure to only have one tokenizer"
    )

    mistral_tokenizer = MistralTokenizer.from_file(str(model_path / tokenizer[0]))

    if isinstance(mistral_tokenizer.instruct_tokenizer.tokenizer, Tekkenizer):
        mistral_tokenizer.instruct_tokenizer.tokenizer.special_token_policy = (
            SpecialTokenPolicy.KEEP
        )

    logging.info(
        f"Loaded tokenizer of type {mistral_tokenizer.instruct_tokenizer.__class__}"
    )

    return mistral_tokenizer
