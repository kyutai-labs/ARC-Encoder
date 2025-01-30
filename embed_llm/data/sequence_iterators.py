import dataclasses
from dataclasses import dataclass
import numpy as np
from embed_llm.training.distributed import get_rank
from embed_llm.data.tokenize import Mask, TokenSample, encode, Tokenizer


@dataclasses.dataclass()
class SequenceEmbedMaskAndSizes:
    """
    Concatenation of samples to reach a given size
    """

    x: list[int]
    y: list[int]
    to_embed: list[dict[str, str | int | list[int] | list[str]]]
    mask: Mask
    sizes: list[int]
    data_type: str
    n_prefixes: list[int] | None = None

    def __post_init__(self):
        assert sum(self.sizes) == len(self.x) == len(self.y) == len(self.mask)
        assert len(self.to_embed) == len(self.sizes)
        if self.n_prefixes is not None:
            assert len(self.n_prefixes) == len(self.sizes), (
                len(self.n_prefixes),
                len(self.sizes),
            )


def sequence_iterator_one_task_4_all(
    tokens: list[int],
    mask: Mask,
    cur_pos: int,
    seq_len: int,
    tokenizer: Tokenizer,
    max_embeds: int = 1,
    start_point: float = 0.0,
) -> SequenceEmbedMaskAndSizes:

    x_buffer, y_buffer, to_embed_buffer, mask_buffer, sizes, n_prefixes = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    x, y = tokens[:-1], tokens[1:]

    if len(x) - cur_pos >= (max_embeds + 1) * seq_len + 20:
        if max_embeds == 1:
            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(tokens[cur_pos : cur_pos + seq_len])],
                    "tokens": [tokens[cur_pos : cur_pos + seq_len]],
                }
            )
            end_embed = seq_len
        else:
            nb_embed = np.random.randint(1, max_embeds + 1)
            new_embed = []
            n_embed_toks = 0

            for i in range(nb_embed):
                new_embed.append(
                    tokens[cur_pos + i * seq_len : cur_pos + (i + 1) * seq_len]
                )
                n_embed_toks += len(
                    tokens[cur_pos + i * seq_len : cur_pos + (i + 1) * seq_len]
                )

            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
            end_embed = n_embed_toks

        min_rand = min(max(1, int(start_point * end_embed)), end_embed + 20)
        start_lm = np.random.randint(min_rand, end_embed + 20)

        n_prefixes.append(end_embed - start_lm)
        x_buffer.extend(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        y_buffer.extend(y[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        sizes.append(size)
        cur_pos += start_lm + size

    # If not enought to put seqlen in both embedding and x, split the rest in two parts
    elif max_embeds == 1 and (len(x) - cur_pos) // 2 > 20 + 1:
        end_embed = (len(x) - cur_pos) // 2 - 10
        to_embed_buffer.append(
            {
                "text": [tokenizer.decode(tokens[cur_pos : cur_pos + end_embed])],
                "tokens": [tokens[cur_pos : cur_pos + end_embed]],
            }
        )
        min_rand = min(max(1, int(start_point * end_embed)), end_embed + 1)
        start_lm = np.random.randint(min_rand, end_embed + 1)

        n_prefixes.append(end_embed - start_lm)
        x_buffer.extend(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        y_buffer.extend(y[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        sizes.append(size)
        cur_pos += start_lm + size

    elif (
        max_embeds > 1
        and (len(x) - cur_pos) > 20 + 1
        and (len(x) - cur_pos - 10) // max_embeds > 0
    ):
        # 10 of prefix + minimum 2 tokens to continue
        nb_embed = np.random.randint(1, max_embeds + 1)
        new_embed = []
        n_embed_toks = 0

        for i in range(nb_embed):
            add_token = min(len(x) - cur_pos - 10 // nb_embed, seq_len)
            new_embed.append(
                tokens[
                    cur_pos
                    + i * add_token : cur_pos
                    + (i + 1) * add_token
                ]
            )
            n_embed_toks += len(
                tokens[
                    cur_pos
                    + i * add_token : cur_pos
                    + (i + 1) * add_token
                ]
            )

        to_embed_buffer.append(
            {
                "text": [tokenizer.decode(toks) for toks in new_embed],
                "tokens": new_embed,
            }
        )
        # In case there is a rest to the euclidean div
        end_embed = n_embed_toks

        min_rand = min(max(1, int(start_point * end_embed)), end_embed + 10)
        start_lm = np.random.randint(min_rand, end_embed + 10)

        n_prefixes.append(end_embed - start_lm)
        x_buffer.extend(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        y_buffer.extend(y[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        sizes.append(size)
        cur_pos += start_lm + size

    else:
        return None

    assert (
        len(mask_buffer) == len(x_buffer) == len(y_buffer)
    ), f"{len(mask_buffer)} == {len(x_buffer)} == {len(y_buffer)}"
    assert sum(sizes) <= seq_len, f"{sum(sizes)} <= {seq_len}"
    assert len(to_embed_buffer) == len(sizes), f"{len(to_embed_buffer)} == {len(sizes)}"

    # we don't want to yield sequences with a mask filled with False
    if any(mask_buffer):
        return (
            SequenceEmbedMaskAndSizes(
                x=x_buffer,
                y=y_buffer,
                to_embed=to_embed_buffer,
                mask=mask_buffer,
                sizes=sizes,
                data_type="one_4_all",
                n_prefixes=n_prefixes
            ),
            cur_pos,
        )
    else:
        return None


def sequence_iterator_reconstruction(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    mask_buffer: Mask,
    n_missing: int,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
    is_eval: bool = False,
) -> SequenceEmbedMaskAndSizes:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    if not adapt_seq_len:
        assert n_missing == seq_len - len(
            x_buffer
        ), f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"

    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    data_type = sample.data_type
    cur_pos = 0

    while cur_pos < len(x):
        size = len(x[cur_pos : cur_pos + n_missing])
        curr_mask = mask[cur_pos : cur_pos + n_missing]
        
        if not any(curr_mask):
            cur_pos += size
            # we have a sequence with a mask filled with False
            continue

        x_buffer.extend(x[cur_pos : cur_pos + n_missing])
        y_buffer.extend(y[cur_pos : cur_pos + n_missing])

        # Because regeneration
        if len(embed_tokens) == 1 and data_type == "reconstruction":
            to_embed_buffer.append(
                {
                    "text": [
                        tokenizer.decode(embed_tokens[0][cur_pos : cur_pos + n_missing])
                    ],
                    "tokens": [embed_tokens[0][cur_pos : cur_pos + n_missing]],
                }
            )
        else:
            # If we want to reconstruct from several chunks of embedded text, we need to be able to reconstruct the full passage
            # To implement, prevent reconstruction of too long sequences
            assert adapt_seq_len
            new_embed_tokens =  [toks[:seq_len] for toks in embed_tokens]
            new_embed_text = [tokenizer.decode(toks[:seq_len]) for toks in embed_tokens]
            to_embed_buffer.append({"text": new_embed_text, "tokens": new_embed_tokens})

        if is_eval:
            curr_mask = [False] * (len(curr_mask)//10) + [True] * (len(curr_mask) - len(curr_mask)//10)
            
        mask_buffer.extend(curr_mask)
        
        if not adapt_seq_len:
            n_missing -= size

        sizes.append(size)

        cur_pos += size

        if n_missing == 0 or (adapt_seq_len and cur_pos == len(x)):
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(x_buffer) <= seq_len

            if not adapt_seq_len:
                assert sum(sizes) == seq_len
                assert seq_len == len(x_buffer)

            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type
                )

            if adapt_seq_len:
                break
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes


def sequence_iterator_continuation(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    n_missing: int,
    mask_buffer: Mask,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
    data_type: str = "continuation",
    is_eval: bool = False,
) -> SequenceEmbedMaskAndSizes:

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    assert (
        len(embed_tokens) == 1
    ), "Continuation training only supports one passage per sample"

    cur_pos = 0

    while cur_pos < len(x):

        overall_size = len(x[cur_pos : cur_pos + n_missing])
        # print('Size:', overall_size, 'N missing:', n_missing, 'Cur pos:', cur_pos)
        curr_mask = mask[cur_pos : cur_pos + n_missing]
        if not any(curr_mask):
            cur_pos += overall_size
            # we have a sequence with a mask filled with False
            continue

        if overall_size < 4:
            assert len(mask_buffer) == len(x_buffer) == sum(sizes) == len(y_buffer)
            assert len(to_embed_buffer) == len(sizes), (
                len(to_embed_buffer),
                len(sizes),
            )

            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type
                )
            else:
                break

        upper_bound = min(cur_pos + n_missing, len(x))

        if not is_eval:
            x_buffer.extend(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            mask_buffer.extend(mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            n_missing -= overall_size

            size = len(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

            sizes.append(size)

            
            to_embed_buffer.append(
                {
                    "text": [
                        tokenizer.decode(
                            embed_tokens[0][
                                cur_pos : cur_pos + (upper_bound - cur_pos) // 2
                            ]
                        )
                    ],
                    "tokens": [
                        embed_tokens[0][cur_pos : cur_pos + (upper_bound - cur_pos) // 2]
                    ],
                }
            )
        else:
            x_buffer.extend(x[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])
            y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])
            
            mask_buffer.extend([False]*len(mask[cur_pos + (upper_bound - cur_pos) // 3 : cur_pos + (upper_bound - cur_pos) // 2 ]) + 
                               mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            


            size = len(x[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])

            sizes.append(size)


            to_embed_buffer.append(
                {
                    "text": [
                        tokenizer.decode(
                            embed_tokens[0][
                                cur_pos : cur_pos + (upper_bound - cur_pos) // 2
                            ]
                        )
                    ],
                    "tokens": [
                        embed_tokens[0][cur_pos : cur_pos + (upper_bound - cur_pos) // 2]
                    ],
                }
            )
          
        cur_pos += overall_size

        if not adapt_seq_len:
           n_missing -= overall_size

        if n_missing == 0 or (adapt_seq_len and cur_pos == len(x)):
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert len(x_buffer) <= seq_len

            assert len(to_embed_buffer) == len(sizes)
            # we don't want to yield sequences with a mask filled with False
            if any(mask_buffer):
                return SequenceEmbedMaskAndSizes(
                    x=x_buffer,
                    y=y_buffer,
                    to_embed=to_embed_buffer,
                    mask=mask_buffer,
                    sizes=sizes,
                    data_type=data_type
                )

            if adapt_seq_len:
                break
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes
