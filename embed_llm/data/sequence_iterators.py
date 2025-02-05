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
    n_gap: int = 60,
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
    
    if max_embeds <= -1:
        nb_embed = abs(max_embeds)
    elif max_embeds > 1:
        nb_embed = np.random.randint(1, max_embeds + 1)
    elif max_embeds == 1:
        nb_embed = 1

    if len(x) - cur_pos >= (nb_embed + 1) * seq_len + n_gap:
        if nb_embed == 1:
            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(tokens[cur_pos : cur_pos + seq_len])],
                    "tokens": [tokens[cur_pos : cur_pos + seq_len]],
                }
            )
            end_embed = seq_len
        else:
            
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

        min_rand = min(max(1, int(start_point * end_embed)), end_embed + n_gap)
        start_lm = np.random.randint(min_rand, end_embed + n_gap)

        n_prefixes.append(end_embed - start_lm)
        x_buffer.extend(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        y_buffer.extend(y[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        sizes.append(size)
        cur_pos += start_lm + size

    # If not enought to put seqlen in both embedding and x, split the rest in two parts
    elif nb_embed == 1 and (len(x) - cur_pos) // 2 > max(n_gap,10) + 1:
        end_embed = (len(x) - cur_pos) // 2 - n_gap
        to_embed_buffer.append(
            {
                "text": [tokenizer.decode(tokens[cur_pos : cur_pos + end_embed])],
                "tokens": [tokens[cur_pos : cur_pos + end_embed]],
            }
        )
        min_rand = min(max(1, int(start_point * end_embed)), end_embed + n_gap)
        start_lm = np.random.randint(min_rand, end_embed + n_gap)

        n_prefixes.append(end_embed - start_lm)
        x_buffer.extend(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        y_buffer.extend(y[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        mask_buffer.extend(mask[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        size = len(x[start_lm + cur_pos : start_lm + cur_pos + seq_len])
        sizes.append(size)
        cur_pos += start_lm + size

    elif (
        nb_embed > 1
        and (len(x) - cur_pos)//2 > 20 + 1 # At least 20 tokens to generate
        and ((len(x) - cur_pos)//2) // nb_embed > 0
    ):
    
        new_embed = []
        n_embed_toks = 0

        for i in range(nb_embed):
            add_token = min(((len(x) - cur_pos)//2) // nb_embed, seq_len)
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

        min_rand = min(max(1, int(start_point * end_embed)), end_embed + 1)
        start_lm = np.random.randint(min_rand, end_embed + 1)

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
    cur_pos: int,
    n_missing: int,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
    is_eval: bool = False,
    max_embeds: int = 1,
    hybrid_training: bool = False,
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

    while cur_pos < len(x):
        size = len(x[cur_pos : cur_pos + n_missing])
        curr_mask = mask[cur_pos : cur_pos + n_missing]
        
        if not any(curr_mask):
            cur_pos += size
            # we have a sequence with a mask filled with False
            continue

        x_buffer.extend(x[cur_pos : cur_pos + n_missing])
        y_buffer.extend(y[cur_pos : cur_pos + n_missing])

        # If instruct data type do not split the passage into smaller embeddings
        if data_type == "reconstruction" and len(embed_tokens) == 1: 
            
            if max_embeds <= -1:
                nb_embed = abs(max_embeds)
            elif max_embeds > 1:
                nb_embed = np.random.randint(1, max_embeds + 1)
            elif max_embeds == 1:
                nb_embed = 1
                
            new_embed = []
            n_toks_per_embed = len(embed_tokens[0][cur_pos : cur_pos + n_missing]) // nb_embed
            
            for i in range(nb_embed):
                new_embed.append(
                    embed_tokens[0][cur_pos + i * n_toks_per_embed : cur_pos + (i + 1) * n_toks_per_embed]
                )

            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
           
        else:
            # If several passages loaded we use these passages directly
            # If we want to reconstruct from several chunks of embedded text, we need to be able to reconstruct the full passage
            assert adapt_seq_len
            if len(embed_tokens) < abs(max_embeds):
                # TODO SPLIT IF TOO LARGE
                new_embed_tokens =  [toks[:seq_len] for toks in embed_tokens]
                new_embed_text = [tokenizer.decode(toks[:seq_len]) for toks in embed_tokens]
            else:
                new_embed_tokens =  [toks[:seq_len] for toks in embed_tokens]
                new_embed_text = [tokenizer.decode(toks[:seq_len]) for toks in embed_tokens]
            
            to_embed_buffer.append({"text": new_embed_text, "tokens": new_embed_tokens})

        if is_eval and hybrid_training:
            curr_mask = [False] * (len(curr_mask)//10) + [True] * (len(curr_mask) - len(curr_mask)//10)
            
        mask_buffer.extend(curr_mask)
        
        if not adapt_seq_len:
            n_missing -= size

        sizes.append(size)

        cur_pos += size
        if n_missing == 0 or (adapt_seq_len and cur_pos == len(x)):
            
            try:
                assert len(mask_buffer) == len(x_buffer) == len(y_buffer), (
                    len(mask_buffer),
                    len(x_buffer),
                    len(y_buffer),)
                assert len(x_buffer) <= seq_len, f'Buffer to long {len(x_buffer)}'
            except AssertionError as e:
                print(e)
                return [], [], [], [], seq_len, []
            
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
                ), cur_pos

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
    cur_pos: int,
    seq_len: int,
    tokenizer: Tokenizer,
    adapt_seq_len: bool = False,
    data_type: str = "continuation",
    is_eval: bool = False,
    max_embeds: int = 1,
    hybrid_training: bool = False,
) -> SequenceEmbedMaskAndSizes:

    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    assert (
        len(embed_tokens) == 1
    ), "Continuation training only supports one passage per sample"

    while cur_pos < len(x):

        overall_size = len(x[cur_pos : cur_pos + n_missing])
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
                ), len(x) # ensures that it does not come back to this sample
            else:
                break

        upper_bound = min(cur_pos + n_missing, len(x))
        if not is_eval or not hybrid_training:
            x_buffer.extend(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            mask_buffer.extend(mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

            size = len(x[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])

            sizes.append(size)

        else:
            x_buffer.extend(x[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])
            y_buffer.extend(y[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])
            
            mask_buffer.extend([False]*len(mask[cur_pos + (upper_bound - cur_pos) // 3 : cur_pos + (upper_bound - cur_pos) // 2 ]) + 
                               mask[cur_pos + (upper_bound - cur_pos) // 2 : upper_bound])
            


            size = len(x[cur_pos + (upper_bound - cur_pos) // 3 : upper_bound])

            sizes.append(size)


        if len(embed_tokens) > 1:
            print("Continuation training only supports one passage per sample")
            
        if max_embeds <= -1:
            nb_embed = abs(max_embeds)
        elif max_embeds > 1:
            nb_embed = np.random.randint(1, max_embeds + 1)
        elif max_embeds == 1:
            nb_embed = 1
            
        new_embed = []
        n_toks_per_embed = len(embed_tokens[0][
                            cur_pos : cur_pos + (upper_bound - cur_pos) // 2
                        ]) // nb_embed
        
        for i in range(nb_embed):
            new_embed.append(
                embed_tokens[0][cur_pos + i * n_toks_per_embed : cur_pos + (i + 1) * n_toks_per_embed]
            )

        to_embed_buffer.append(
            {
                "text": [tokenizer.decode(toks) for toks in new_embed],
                "tokens": new_embed,
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
                ), cur_pos

            if adapt_seq_len:
                break
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes






def sequence_iterator_decompress_usage(
    x_buffer: list[int],
    y_buffer: list[int],
    to_embed_buffer: list[dict[str, str | int | list[int] | list[str]]],
    mask_buffer: Mask,
    cur_pos: int,
    n_missing: int,
    sizes: list[int],
    sample: TokenSample,
    seq_len: int,
    tokenizer: Tokenizer,
    max_embeds: int = 1,
    decompress_usage: str = "",
) -> SequenceEmbedMaskAndSizes:
    """
    Creates sequences of length `seq_len` from the dataset iterator by concatenating samples.
    """
    assert 0 <= len(x_buffer) < seq_len, len(x_buffer)
    assert n_missing == seq_len - len(
        x_buffer
    ), f"n_missing: {n_missing} | seq_len - len(x_buffer) {seq_len - len(x_buffer)}"
    
    tokens, mask = sample.tokens, sample.masks[1:]
    x, y = tokens[:-1], tokens[1:]
    embed_tokens = sample.passages.tokens
    data_type = sample.data_type

    while cur_pos < len(x):
        size = len(x[cur_pos : cur_pos + n_missing])
        # If instruct data type do not split the passage into smaller embeddings
        if data_type == "reconstruction" and len(embed_tokens) == 1: 
            
            if max_embeds <= -1:
                nb_embed = abs(max_embeds)
            elif max_embeds > 1:
                nb_embed = np.random.randint(1, max_embeds + 1)
            elif max_embeds == 1:
                nb_embed = 1
                
            new_embed = []
            n_toks_per_embed = len(embed_tokens[0][cur_pos : cur_pos + seq_len]) // nb_embed
            
            for i in range(nb_embed):
                new_embed.append(
                    embed_tokens[0][cur_pos + i * n_toks_per_embed : cur_pos + (i + 1) * n_toks_per_embed]
                )

            to_embed_buffer.append(
                {
                    "text": [tokenizer.decode(toks) for toks in new_embed],
                    "tokens": new_embed,
                }
            )
           
        else:
            raise ValueError("Decompress usage only supports one passage per sample")   

        if decompress_usage == 'middle_reconstruction':
            middle = len(x[cur_pos : cur_pos + seq_len]) // 2
            x_buffer.extend(x[cur_pos + middle : cur_pos + seq_len])
            y_buffer.extend(y[cur_pos + middle : cur_pos + seq_len])
            mask_buffer.extend(len(y[cur_pos + middle : cur_pos + seq_len]) * [True])
            x_size = len(x[cur_pos + middle : cur_pos + seq_len])
        elif decompress_usage == 'one_over_two_reconstruction':
            x_buffer.extend(x[cur_pos : cur_pos + seq_len:2])
            y_buffer.extend(y[cur_pos : cur_pos + seq_len:2])
            mask_buffer.extend(len(y[cur_pos : cur_pos + seq_len:2])*[True])
            x_size = len(x[cur_pos : cur_pos + seq_len:2])
        elif decompress_usage == 'reversed':
            x_buffer.extend(y[cur_pos + seq_len-1: cur_pos:-1])
            y_buffer.extend(x[cur_pos + seq_len-1: cur_pos:-1])
            mask_buffer.extend(len(y[cur_pos + seq_len-1: cur_pos:-1])*[True])
            x_size = len(y[cur_pos + seq_len-1: cur_pos:-1])
            if x_size == 0:
                return x_buffer, y_buffer, to_embed_buffer[:-1], mask_buffer, n_missing, sizes
        elif decompress_usage == 'from_prefix_reconstruct':
            start_prefix = np.random.randint(0,len(x[cur_pos : cur_pos + seq_len]) + 1 - 10)
            x_buffer.extend(x[cur_pos + start_prefix: cur_pos + seq_len])
            y_buffer.extend(y[cur_pos + start_prefix: cur_pos + seq_len])
            mask_buffer.extend(5*[False]+len(y[cur_pos + start_prefix + 5: cur_pos + seq_len])*[True]) # Give 5 prefix overlapping tokens to set from where to reconstruct
            x_size = len(x[cur_pos + start_prefix: cur_pos + seq_len])
        else:
            raise NotImplementedError(f"Decompress usage {decompress_usage} not supported")
            
        cur_pos += size
        sizes.append(x_size)

        n_missing -= x_size

        if n_missing <= 0:
            assert len(mask_buffer) == len(x_buffer) == len(y_buffer)
            assert all([size > 0 for size in sizes]), 'All sizes should be greater than 0'

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
                ), cur_pos
                
    return x_buffer, y_buffer, to_embed_buffer, mask_buffer, n_missing, sizes
