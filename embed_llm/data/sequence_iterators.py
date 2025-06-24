import dataclasses
import numpy as np
import re
from embed_llm.data.tokenize import Mask, TokenSample, Tokenizer


@dataclasses.dataclass()
class SequenceEmbedMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: list[int]
    y: list[int]
    to_embed: list[dict[list[str], list[list[int]]]]
    mask: Mask
    sizes: list[int]
    data_type: str
    insert_embed_list: list[list[int]] | None = None

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask), (
            sum(self.sizes),
            len(self.x),
            len(self.y),
            len(self.mask),
        )
        assert len(self.to_embed) == len(self.sizes)

        if self.insert_embed_list is not None:
            if len(self.insert_embed_list) == 0:
                self.insert_embed_list = None
            elif len(self.insert_embed_list) == 1:
                assert 1 == len(self.sizes)

            else:
                assert len(self.insert_embed_list) == len(self.sizes), (
                    f"{self.insert_embed_list}, {self.sizes}"
                )


def sequence_iterator_reconstruction(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[list[str], list[list[int]]]],
    insert_embed_list: list[list[int]],
    mask_buffer: Mask,
    cur_pos: int,
    n_missing: int,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    adapt_seq_len: bool = False,
    few_shot_instruct: list[str] | None = None,
    few_shot: int = 0,
    n_interleaved: int = 1,
    loss_last_cont_only: bool = False,
) -> SequenceEmbedMaskAndSizes:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    if not adapt_seq_len:
        assert n_missing == seq_len - len(x_buffer), (
            f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"
        )

    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    data_type = sample.data_type

    while cur_pos < len(x):
        size = len(x[cur_pos : cur_pos + n_missing])
        curr_mask = mask[cur_pos : cur_pos + n_missing]

        if not any(curr_mask):
            cur_pos += size
            # we have a sequence with a mask filled with False
            continue

        # If instruct data type do not split the passage into smaller embeddings
        if data_type == "reconstruction":
            new_embed = tokens[cur_pos : cur_pos + n_missing]
            if (
                llm_tokenizer.model_name == "llama"
                and embed_tokenizer.model_name == "mistral"
            ):
                new_text = llm_tokenizer.tokenizer.decode(new_embed)
                bos = "<|begin_of_text|>" in new_text
                eos = "<|end_of_text|>" in new_text
                for sp_tok in llm_tokenizer.tokenizer.special_tokens.keys():
                    new_text = new_text.replace(sp_tok, "")
                to_embed_buffer.append(
                    {
                        "text": [new_text],
                        "tokens": [
                            embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                        ],
                    }
                )

            elif (
                llm_tokenizer.model_name == "mistral"
                and embed_tokenizer.model_name == "llama"
            ):
                bos = llm_tokenizer.tokenizer.bos_id in new_embed
                eos = llm_tokenizer.tokenizer.eos_id in new_embed
                new_text = llm_tokenizer.tokenizer.decode(new_embed)
                to_embed_buffer.append(
                    {
                        "text": [new_text],
                        "tokens": [
                            embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                        ],
                    }
                )
            else:
                to_embed_buffer.append(
                    {
                        "text": [llm_tokenizer.tokenizer.decode(new_embed)],
                        "tokens": [new_embed],
                    }
                )

            # Each sample consists in: Embeddings + text (no text before the embeddings)
            insert_embed_list.append([0])

        else:
            assert adapt_seq_len
            seq_len = (
                seq_len * (2 * n_interleaved + 1) if n_interleaved > 1 else seq_len
            )
            # If we can use more embeddings and that one passage reaches the limit \
            # we split it in two embeddings and so on

            new_embed = []
            for i in range(len(embed_tokens)):
                new_embed.append(embed_tokens[i])

            new_embed_tokens = sum([toks[:seq_len] for toks in new_embed], [])
            new_embed_text = " ".join(
                [
                    embed_tokenizer.tokenizer.decode(toks[:seq_len]).strip()
                    for toks in new_embed
                ]
            )
            if embed_tokenizer.model_name == "llama":
                for sp_tok in embed_tokenizer.tokenizer.special_tokens.keys():
                    new_embed_text = new_embed_text.replace(sp_tok, "")

            if data_type == "instruct":
                if few_shot_instruct is None:
                    prefix = "Document: "
                else:
                    prefix = "\n\n".join(
                        few_shot_instruct[
                            : np.random.randint(0, len(few_shot_instruct) + 1)
                        ]
                    )

                    prefix = (
                        prefix + "\n\nDocument: " if len(prefix) > 0 else "Document: "
                    )
                added_prefix = 0
                splits = re.split(r"\n\nDocument:|\nQuestion:", prefix)
                if n_interleaved > 1 and len(splits) > 1:
                    ins_list = []
                    embed_toks = []
                    embed_text = []
                    doc_tokens = llm_tokenizer.tokenizer.encode(
                        "Document: ", bos=True, eos=False
                    )
                    ins_list.append(len(doc_tokens))
                    x_buffer.extend(doc_tokens)
                    y_buffer.extend(doc_tokens)
                    for i, split in enumerate(splits):
                        split = split.replace("Document: ", "").strip()
                        if split == "":
                            continue
                        if i % 2 == 0:  # First part is the document
                            embed_toks.append(
                                embed_tokenizer.tokenizer.encode(
                                    split, bos=False, eos=False
                                )
                            )
                            embed_text.append(split)
                        elif (
                            i < len(splits) - 1
                        ):  # Second part is the question and answer which should not be compressed
                            doc_tokens = llm_tokenizer.tokenizer.encode(
                                "\nQuestion: " + split.strip() + "\n\nDocument: ",
                                bos=False,
                                eos=False,
                            )
                            ins_list.append(len(doc_tokens))
                            x_buffer.extend(doc_tokens)
                            y_buffer.extend(doc_tokens)

                        else:
                            doc_tokens = llm_tokenizer.tokenizer.encode(
                                "\nQuestion: " + split.strip(), bos=False, eos=False
                            )
                            ins_list.append(len(doc_tokens))
                            x_buffer.extend(doc_tokens)
                            y_buffer.extend(doc_tokens)
                    insert_embed_list.append(ins_list)
                    added_prefix = sum(ins_list)
                    embed_toks.append(new_embed_tokens)
                    embed_text.append(new_embed_text)
                    if any([len(toks) <= 1 for toks in embed_toks]):
                        print("Embed text small", embed_text)

                    to_embed_buffer.append({"text": embed_text, "tokens": embed_toks})
                else:
                    doc_tokens = llm_tokenizer.tokenizer.encode(
                        prefix, bos=True, eos=False
                    )
                    insert_embed_list.append([len(doc_tokens)])
                    x_buffer.extend(doc_tokens)
                    y_buffer.extend(doc_tokens)
                    added_prefix = len(doc_tokens)
                    to_embed_buffer.append(
                        {"text": [new_embed_text], "tokens": [new_embed_tokens]}
                    )

                if few_shot_instruct is not None:
                    question = llm_tokenizer.tokenizer.decode(
                        [
                            int(tok)
                            for i, tok in enumerate(
                                tokens[cur_pos : cur_pos + len(curr_mask)]
                            )
                            if not curr_mask[i]
                        ]
                    )
                    answer = llm_tokenizer.tokenizer.decode(
                        [
                            int(tok)
                            for i, tok in enumerate(
                                tokens[cur_pos : cur_pos + len(curr_mask)]
                            )
                            if curr_mask[i]
                        ]
                    )
                    new_ex = "Document: " + new_embed_text + question + answer
                    if len(few_shot_instruct) < few_shot:
                        few_shot_instruct.append(new_ex)
                    else:
                        few_shot_instruct = [new_ex] + few_shot_instruct[:-1]
                        assert len(few_shot_instruct) == few_shot, (
                            f"size of the examples {len(few_shot_instruct)}"
                        )
                if loss_last_cont_only:
                    curr_mask = [False] * added_prefix + curr_mask
                else:
                    curr_mask = [True] * added_prefix + curr_mask
                size = added_prefix + size
                seq_len += added_prefix

        x_buffer.extend(x[cur_pos : cur_pos + n_missing])
        y_buffer.extend(y[cur_pos : cur_pos + n_missing])
        mask_buffer.extend(curr_mask)
        if not adapt_seq_len:
            n_missing -= size

        sizes.append(size)

        cur_pos += size
        if n_missing == 0 or (
            (adapt_seq_len and cur_pos >= len(x)) or len(x_buffer) == seq_len
        ):
            # With adapt seq len just do not cut a sequence in the middle to fill the empty space
            # But still upper bounded by max_seq_len

            try:
                assert len(mask_buffer) == len(x_buffer) == len(y_buffer), (
                    len(mask_buffer),
                    len(x_buffer),
                    len(y_buffer),
                )
                assert len(x_buffer) <= seq_len, f"Buffer to long {len(x_buffer)}"
            except AssertionError as e:
                print(e)
                return [], [], [], [], [], seq_len, [], few_shot_instruct

            if not adapt_seq_len:
                assert sum(sizes) == seq_len
                assert seq_len == len(x_buffer)

            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return (
                    SequenceEmbedMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        to_embed=to_embed_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                        data_type=data_type,
                        insert_embed_list=insert_embed_list,
                    ),
                    cur_pos,
                )

            if adapt_seq_len:
                break
    return (
        x_buffer,
        y_buffer,
        to_embed_buffer,
        insert_embed_list,
        mask_buffer,
        n_missing,
        sizes,
        few_shot_instruct,
    )


def sequence_iterator_inserted_embed_continuation(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[list[str], list[list[int]]]],
    insert_embed_list: list[list[int]],
    n_missing: int,
    mask_buffer: Mask,
    sizes: list[int],
    sample: TokenSample,
    cur_pos: int,
    seq_len: int,
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    data_type: str = "continuation",
    n_times_sl_insertion: int = 1,
    shorten_continuation: bool = False,
) -> SequenceEmbedMaskAndSizes:
    assert 0 <= len(x_buffer) < (1 + n_times_sl_insertion) * seq_len, len(x_buffer)
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    size = 0
    while cur_pos < len(x):
        overall_size = len(x[cur_pos : cur_pos + n_missing])
        curr_mask = mask[cur_pos : cur_pos + n_missing]
        if not any(curr_mask):
            cur_pos += overall_size
            # we have a sequence with a mask filled with False
            continue

        if overall_size < 6:
            assert len(mask_buffer) == len(x_buffer) == sum(sizes) == len(y_buffer)
            assert len(to_embed_buffer) == len(sizes), (
                len(to_embed_buffer),
                len(sizes),
            )
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return (
                    SequenceEmbedMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        to_embed=to_embed_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                        data_type=data_type,
                        insert_embed_list=insert_embed_list,
                    ),
                    len(x),
                )  # ensures that it does not come back to this sample
            else:
                break

        upper_bound_non_embed_prefix = max(0, overall_size - 2 * seq_len)
        # either you can continue 256 tokens of embeddings by 256 tokens then the spared ones are put before the embeddings

        x_buffer.extend(x[cur_pos : cur_pos + upper_bound_non_embed_prefix])
        y_buffer.extend(y[cur_pos : cur_pos + upper_bound_non_embed_prefix])
        mask_buffer.extend(
            [False] * len(mask[cur_pos : cur_pos + upper_bound_non_embed_prefix])
        )
        size += len(x[cur_pos : cur_pos + upper_bound_non_embed_prefix])
        cur_pos += len(x[cur_pos : cur_pos + upper_bound_non_embed_prefix])

        insert_embed_list.append([size])
        left_tokens = max(
            min(n_missing - upper_bound_non_embed_prefix, len(x) - cur_pos), 0
        )
        new_embed = x[cur_pos : cur_pos + left_tokens // 2]

        if (
            llm_tokenizer.model_name == "llama"
            and embed_tokenizer.model_name == "mistral"
        ):
            new_text = llm_tokenizer.tokenizer.decode(new_embed)
            bos = "<|begin_of_text|>" in new_text
            eos = "<|end_of_text|>" in new_text
            for sp_tok in llm_tokenizer.tokenizer.special_tokens.keys():
                new_text = new_text.replace(sp_tok, "")
            to_embed_buffer.append(
                {
                    "text": [new_text],
                    "tokens": [
                        embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                    ],
                }
            )

        elif (
            llm_tokenizer.model_name == "mistral"
            and embed_tokenizer.model_name == "llama"
        ):
            bos = llm_tokenizer.tokenizer.bos_id in new_embed
            eos = llm_tokenizer.tokenizer.eos_id in new_embed
            new_text = llm_tokenizer.tokenizer.decode(new_embed)
            to_embed_buffer.append(
                {
                    "text": [new_text],
                    "tokens": [
                        embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                    ],
                }
            )
        else:
            to_embed_buffer.append(
                {
                    "text": [llm_tokenizer.tokenizer.decode(new_embed)],
                    "tokens": [new_embed],
                }
            )

        cur_pos += len(x[cur_pos : cur_pos + left_tokens // 2])

        to_continue_tokens = (
            left_tokens // 2 if not shorten_continuation else min(32, left_tokens // 2)
        )
        if shorten_continuation:
            overall_size -= max((left_tokens // 2) - 32, 0)
        x_buffer.extend(x[cur_pos : cur_pos + to_continue_tokens])
        y_buffer.extend(y[cur_pos : cur_pos + to_continue_tokens])

        if np.unique(new_embed).size == 1:
            mask_buffer.extend(
                [False] * len(mask[cur_pos : cur_pos + to_continue_tokens])
            )
        else:
            mask_buffer.extend(
                [True] * len(mask[cur_pos : cur_pos + to_continue_tokens])
            )
        size += len(x[cur_pos : cur_pos + to_continue_tokens])
        cur_pos += len(x[cur_pos : cur_pos + to_continue_tokens])

        sizes.append(size)

        n_missing -= overall_size
        size = 0
        if n_missing == 0:
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(x_buffer) <= seq_len * (1 + n_times_sl_insertion), (
                f"Buffer to long {len(x_buffer)} | {seq_len * (1 + n_times_sl_insertion)}"
            )

            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return (
                    SequenceEmbedMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        to_embed=to_embed_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                        data_type=data_type,
                        insert_embed_list=insert_embed_list,
                    ),
                    cur_pos,
                )
            else:
                return (
                    [],
                    [],
                    [],
                    [],
                    [],
                    seq_len * 2 + n_times_sl_insertion * seq_len,
                    [],
                )
    return (
        x_buffer,
        y_buffer,
        to_embed_buffer,
        insert_embed_list,
        mask_buffer,
        n_missing,
        sizes,
    )


def sequence_iterator_interleaved_cont(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[list[str], list[list[int]]]],
    insert_embed_list: list[list[int]],
    n_missing: int,
    mask_buffer: Mask,
    sizes: list[int],
    sample: TokenSample,
    cur_pos: int,
    seq_len: int,
    llm_tokenizer: Tokenizer,  # type: ignore
    embed_tokenizer: Tokenizer,  # type: ignore
    data_type: str = "continuation",
    n_interleaved: int = 1,
    loss_last_cont_only: bool = False,
) -> SequenceEmbedMaskAndSizes:
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    size = 0
    while cur_pos < len(x):
        overall_size = len(x[cur_pos : cur_pos + n_missing])
        curr_mask = mask[cur_pos : cur_pos + n_missing]
        if not any(curr_mask):
            cur_pos += overall_size
            # we have a sequence with a mask filled with False
            continue

        if overall_size < 2 * (2 * n_interleaved + 1):
            assert len(mask_buffer) == len(x_buffer) == sum(sizes) == len(y_buffer)
            assert len(to_embed_buffer) == len(sizes), (
                len(to_embed_buffer),
                len(sizes),
            )

            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return (
                    SequenceEmbedMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        to_embed=to_embed_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                        data_type=data_type,
                        insert_embed_list=insert_embed_list,
                    ),
                    len(x),
                )  # ensures that it does not come back to this sample
            else:
                break

        one_chunk_size = overall_size // (2 * n_interleaved + 1)
        text_embed = []
        toks_embed = []
        ins_list = []
        for j in range(2 * n_interleaved + 1):
            if j % 2 == 0 and j < 2 * n_interleaved:
                x_buffer.extend(x[cur_pos : cur_pos + one_chunk_size])
                y_buffer.extend(y[cur_pos : cur_pos + one_chunk_size])
                if loss_last_cont_only:
                    mask_buffer.extend(
                        [False] * len(mask[cur_pos : cur_pos + one_chunk_size])
                    )
                else:
                    mask_buffer.extend(
                        [True] * len(mask[cur_pos : cur_pos + one_chunk_size])
                    )
                size += len(x[cur_pos : cur_pos + one_chunk_size])
                ins_list.append(len(x[cur_pos : cur_pos + one_chunk_size]))

                cur_pos += len(x[cur_pos : cur_pos + one_chunk_size])

            elif j % 2 == 1:
                new_embed = x[cur_pos : cur_pos + one_chunk_size]
                if (
                    llm_tokenizer.model_name == "llama"
                    and embed_tokenizer.model_name == "mistral"
                ):
                    new_text = llm_tokenizer.tokenizer.decode(new_embed)
                    bos = "<|begin_of_text|>" in new_text
                    eos = "<|end_of_text|>" in new_text
                    for sp_tok in llm_tokenizer.tokenizer.special_tokens.keys():
                        new_text = new_text.replace(sp_tok, "")
                    text_embed.append(new_text)
                    toks_embed.append(
                        embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                    )

                elif (
                    llm_tokenizer.model_name == "mistral"
                    and embed_tokenizer.model_name == "llama"
                ):
                    bos = llm_tokenizer.tokenizer.bos_id in new_embed
                    eos = llm_tokenizer.tokenizer.eos_id in new_embed
                    new_text = llm_tokenizer.tokenizer.decode(new_embed)
                    text_embed.append(new_text)
                    toks_embed.append(
                        embed_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                    )
                else:
                    text_embed.append(llm_tokenizer.tokenizer.decode(new_embed))
                    toks_embed.append(new_embed)

                cur_pos += len(x[cur_pos : cur_pos + one_chunk_size])
            else:
                to_continue_tokens = overall_size // (
                    2 * n_interleaved + 1
                ) + overall_size % (2 * n_interleaved + 1)

                x_buffer.extend(x[cur_pos : cur_pos + to_continue_tokens])
                y_buffer.extend(y[cur_pos : cur_pos + to_continue_tokens])
                size += len(x[cur_pos : cur_pos + to_continue_tokens])
                mask_buffer.extend(
                    [True] * len(mask[cur_pos : cur_pos + to_continue_tokens])
                )
                cur_pos += len(x[cur_pos : cur_pos + to_continue_tokens])
        insert_embed_list.append(ins_list)
        sizes.append(size)
        to_embed_buffer.append(
            {
                "text": text_embed,
                "tokens": toks_embed,
            }
        )
        n_missing -= overall_size
        size = 0
        if n_missing == 0:
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return (
                    SequenceEmbedMaskAndSizes(
                        x=x_buffer,
                        y=y_buffer,
                        to_embed=to_embed_buffer,
                        mask=mask_buffer,
                        sizes=sizes,
                        data_type=data_type,
                        insert_embed_list=insert_embed_list,
                    ),
                    cur_pos,
                )
            else:
                return (
                    [],
                    [],
                    [],
                    [],
                    [],
                    seq_len,
                    [],
                )
    return (
        x_buffer,
        y_buffer,
        to_embed_buffer,
        insert_embed_list,
        mask_buffer,
        n_missing,
        sizes,
    )
