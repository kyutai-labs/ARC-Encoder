import logging
import math
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DOC_TEMPLATE = {
    "NQA": "Story:\n{text}",
    "Qspr": "Article:\n{text}",
    "GvRp": "Report:\n{text}",
    "QMSum": "Transcript:\n{text}",
}

DATA_TEMPLATE = {
    "NQA": "{instruction}Question:\n{question}\n\nAnswer:\n{answer}",
    "Qspr": "{instruction}Question:\n{question}\n\nAnswer:\n{answer}",
    "GvRp": "{instruction}Summary:\n{answer}",
    "QMSum": "{instruction}Query:\n{question}\n\nAnswer:\n{answer}",
}

TRUNCATE_SEPARATOR = {
    "NQA": "... [The rest of the story is omitted]\n\n",
    "Qspr": "... [The rest of the article is omitted]\n\n",
    "GvRp": "... [The rest of the report is omitted]\n\n",
    "QMSum": "... [The rest of the transcript is omitted]\n\n",
}

INSTRUCTION = {
    "NQA": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\n",
    "Qspr": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n\n',
    "GvRp": "Instruction: You are given a report by a government agency. Write a one-page summary of the report.\n",
    "QMSum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n",
}


def sample_demos(data, num):
    # sample the demos using a balanced sampling strategy from each class

    n_classes = len(data)
    n_sets = math.ceil(num / n_classes)
    new_data = [[] for _ in range(n_sets)]

    for classes in data:
        idx = random.sample(range(len(classes)), n_sets % len(classes))
        while len(idx) < n_sets:
            # if we want more sets than the number of demos in this class, we just sample with replacement
            idx += random.sample(range(len(classes)), len(classes))

        for i, id in enumerate(idx):
            new_data[i].append(classes[id])
    for d in new_data:
        random.shuffle(d)
    new_data = [item for sublist in new_data for item in sublist][:num]

    return new_data


def preprocess_demos(demos, balanced_sampling=False):
    # preprocess the demos by grouping them by answer if we are doing balanced sampling
    # else we put all demos into one group
    if not balanced_sampling:
        return [list(demos)]
    by_answer = defaultdict(list)
    for item in demos:
        by_answer[item["answer"]].append(item)
    return [v for k, v in by_answer.items()]


class TestItem:
    """
    Test item
    """

    def __init__(self, eval_model: str, test_item: dict[str], dataset: str):
        """
        This class represents a test item, which consists of a list of demos and a test item
        self.test_documents = ["Title + text of doc"], length = args.n_test_doc
        self.test_text = "the question"
        self.answer = "the answer"
        """

        # for the test item, we keep a list of documents
        self.test_documents = [test_item["passage"]]
        self.dataset_name = dataset
        self.question = test_item["question"] if "question" in test_item else None
        self.answer = test_item["answer"]
        test_item["answer"] = ""
        # instruction = test_item["instruction"]

        self.test_text = DATA_TEMPLATE[dataset].format(**test_item)
        test_item["answer"] = self.answer

        self.truncate_seperator = TRUNCATE_SEPARATOR[dataset]

        if "cepe" in eval_model:
            self.truncate_seperator = "\n\n"

    def format_documents(self, docs):
        # format the documents for the context input
        if isinstance(docs, list):
            return "\n\n".join(docs)
        else:
            return docs.strip()

    def format_decoder_inputs(
        self,
        test_docs,
        test_text,
        input_max_length,
        tokenizer,
        concat_N: int | None = None,
    ):
        text = "\n\n".join(test_docs) + ("\n\n" if len(test_docs) > 0 else "")
        text = DOC_TEMPLATE[self.dataset_name].format(text=text)
        query_start_index = len(text)
        text += test_text
        tokenized_text = tokenizer([text], return_tensors="pt")
        if tokenized_text.input_ids.shape[1] <= input_max_length:
            return tokenized_text.input_ids, tokenized_text.attention_mask, ""

        # # we need to truncate the text
        # logger.info(
        #     f"Prompt length exceeds max input length: {tokenized_text.input_ids.shape[1]} > {input_max_length}, truncating..."
        # )

        test_text = self.truncate_seperator + test_text
        tokenized_query = tokenizer(
            [test_text], return_tensors="pt", add_special_tokens=False
        )
        before_query = text[:query_start_index]
        tokenized_before_query = tokenizer(
            [before_query], return_tensors="pt", return_offsets_mapping=True
        )

        n_context_tokens = input_max_length - tokenized_query.input_ids.size(1)
        offset_mapping = tokenized_before_query.offset_mapping[0]
        max_tok = 2**18 if concat_N is None else concat_N

        input_ids = tokenized_before_query.input_ids[:, -n_context_tokens:]
        start_tok = max(
            -tokenized_before_query.input_ids.size(1), -n_context_tokens - max_tok + 1
        )
        overflown_text = text[
            offset_mapping[start_tok][0] : offset_mapping[-n_context_tokens][1]
        ]

        input_ids = torch.cat([input_ids, tokenized_query.input_ids], dim=1)

        return input_ids, torch.ones_like(input_ids), overflown_text

    def get_vanilla_inputs(
        self, generation_max_length: int, input_max_length: int, tokenizer
    ):
        # vanilla is just a simple concatenation of the prompt
        model_inputs = {}

        input_max_length = input_max_length - generation_max_length

        prefix_input_ids, prefix_attn_mask, overflown_text = self.format_decoder_inputs(
            self.test_documents,
            self.test_text,
            input_max_length,
            tokenizer,
            concat_N=None,
        )

        prefix_length = prefix_input_ids.size(1)
        model_inputs["prefix_length"] = prefix_length
        model_inputs["prefix_inputs"] = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attn_mask,
        }
        model_inputs["text"] = tokenizer.decode(
            prefix_input_ids[0], skip_special_tokens=True
        )

        return model_inputs

    def get_text(
        self,
        context_max_length: int,
        concat_N: int,
        llm_tokenizer,
        compressor_tokenizer,
    ):
        model_inputs = {}
        decoder_input = self.test_text
        passage_tokens = compressor_tokenizer.encode(
            self.test_documents, bos=False, eos=False
        )[:context_max_length]
        passage_tokens = [
            passage_tokens[ind : ind + len(passage_tokens) // concat_N]
            for ind in range(0, len(passage_tokens), len(passage_tokens) // concat_N)
        ]
        encoder_text_input = [
            DOC_TEMPLATE[self.dataset_name].format(
                text=compressor_tokenizer.decode(passage, skip_special_tokens=True)
            )
            for passage in passage_tokens
        ]
        encoder_inputs = [
            compressor_tokenizer.encode(text, bos=False, eos=False)
            for text in encoder_text_input
        ]
        model_inputs["embed_seqlens"] = [
            len(encoder_input) for encoder_input in encoder_inputs
        ]
        model_inputs["embeddings"] = torch.cat(
            [
                compressor_tokenizer.encode(text, bos=False, eos=False)
                for text in encoder_text_input
            ],
            dim=0,
        )
        model_inputs["prompt_tokens"] = llm_tokenizer.encode(
            decoder_input, add_special_tokens=False, bos=True, eos=False
        )
        model_inputs["insertion_lists"] = [1]  # Insert after BOS token
        return model_inputs

    def get_context_inputs(
        self,
        eval_model: str,
        input_max_length: int,
        generation_max_length: int,
        context_max_length: int,
        concat_N: int,
        tokenizer,
    ):
        """
        The context input can be the demos, the documents, or overflowing tokens from the encoder, or a combination of these
        """
        context_text = []
        decoder_docs = self.test_documents
        encoder_docs = []

        model_inputs = {}

        input_max_length = input_max_length - generation_max_length

        # step 2: get the decoder inputs (called prefix here)
        prefix_input_ids, prefix_attn_mask, overflown_text = self.format_decoder_inputs(
            decoder_docs, self.test_text, input_max_length, tokenizer, concat_N
        )
        prefix_length = prefix_input_ids.size(1)
        # 3. the test documents

        docs_in_encoder = [
            self.format_documents(encoder_docs[i]) for i in range(0, len(encoder_docs))
        ]

        context_text = [
            "\n\n".join(docs_in_encoder),
            overflown_text,
        ]
        context_text = [text for text in context_text if len(text) > 0]

        # get the encoder input ids (called context inputs here)
        if len(context_text) > 0:
            # print('Len context before', len(context_text))
            context_inputs = tokenizer(
                context_text,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                max_length=context_max_length,
                return_overflowing_tokens=True,
                truncation=True,
                add_special_tokens=False,
            )
            # print('Len context text', len(context_text[0]))
            # print('CONTEXT INPUTS', context_inputs.input_ids.shape)

            # unsqueeze bc we expect shape of bsz, n_context, seq_len
            encoder_input_ids = context_inputs.input_ids.unsqueeze(0)
            encoder_attention_mask = context_inputs.attention_mask.unsqueeze(0)

        else:
            # this is the rare case where we don't have any context
            # (e.g. zero-shot and the context can fit within the decoder)
            encoder_input_ids = None
            encoder_attention_mask = None

        # put everything into the model inputs
        model_inputs["prefix_length"] = prefix_length
        model_inputs["prefix_inputs"] = {
            "input_ids": prefix_input_ids,
            "attention_mask": prefix_attn_mask,
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return model_inputs


class TestItemDataset(Dataset):
    def __init__(
        self,
        eval_model,
        dataset,
        dataset_name,
        llm_tokenizer,
        compressor_tokenizer=None,
        generation_max_length: int | None = 1024,
        input_max_length: int | None = None,
        context_max_length: int | None = None,
        concat_N: int | None = None,
    ):
        self.eval_model = eval_model
        self.dataset_name = dataset_name
        self.test_data = dataset
        self.llm_tokenizer = llm_tokenizer
        self.compressor_tokenizer = compressor_tokenizer
        self.generation_max_length = generation_max_length
        self.context_max_length = context_max_length
        self.input_max_length = input_max_length
        self.concat_N = concat_N

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        test_item = TestItem(self.eval_model, self.test_data[idx], self.dataset_name)

        if "cepe" in self.eval_model:
            inputs = test_item.get_context_inputs(
                self.eval_model,
                self.input_max_length,
                self.generation_max_length,
                self.context_max_length,
                concat_N=self.concat_N,
                tokenizer=self.llm_tokenizer,
            )
        elif "baseline" in self.eval_model or "together" in self.eval_model:
            inputs = test_item.get_vanilla_inputs(
                generation_max_length=self.generation_max_length,
                input_max_length=self.input_max_length,
                tokenizer=self.llm_tokenizer,
            )
        elif "ours" in self.eval_model:
            inputs = test_item.get_text(
                self.context_max_length,
                self.concat_N,
                self.llm_tokenizer,
                self.compressor_tokenizer,
            )

        inputs["original_data"] = self.test_data[idx]
        inputs["test_item"] = test_item
        return inputs
