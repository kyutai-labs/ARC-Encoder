import glob
import os

def generate_haystacks( 
                tokenizer,
                needle: str,
                context_lengths: list[int],
                document_depth_percents: list[int]):
    
    haystacks_contexts = {}
    for context_length in context_lengths:
        haystacks_contexts[context_length] = {}
        for depth_percent in document_depth_percents:
            context = read_context_files(tokenizer = tokenizer, context_length = context_length)

            # Truncate the haystack dir essays to the context length you desire
            context = encode_and_trim(tokenizer, context_length, context)

            # Insert your random statement according to your depth percent
            context = insert_needle(tokenizer, context_length, needle, context, depth_percent)
            haystacks_contexts[context_length][depth_percent] = context
    return haystacks_contexts


def insert_needle(tokenizer, context_length, needle, context, depth_percent):
    tokens_needle = tokenizer.encode(needle, bos = False, eos = False)
    tokens_context = tokenizer.encode(context, bos = True, eos = False)

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = tokenizer.encode('.', bos = False, eos = False)
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context)
    return new_context


def read_context_files(tokenizer, context_length):
    context = ""
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

    while len(tokenizer.encode(context, bos = False, eos = False)) < context_length:
        for file in glob.glob(os.path.join(base_dir, "PaulGrahamEssays", "*.txt")):
            with open(file, 'r') as f:
                context += f.read()
    return context

def encode_and_trim(tokenizer,  context_length, context):
    tokens = tokenizer.encode(context, bos = True, eos = False)
    if len(tokens) > context_length:
        context = tokenizer.decode(tokens)
    return context
    
 