import argparse
import json
import random
import re  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from nltk.tokenize import sent_tokenize

import nltk  # type: ignore
from vllm import LLM, SamplingParams  # type: ignore
from embed_llm import DATA_PATH

nltk.download("punkt")

ATLAS_PATH = DATA_PATH + 'raw/Atlas_passages_validation.jsonl'
ARXIV_PATH = DATA_PATH + 'raw/arxiv.jsonl'
PG19_PATH = DATA_PATH + 'raw/pg19.jsonl'


PREFIX = {
    "arxiv": ["Article:\n{text}"],
    "atlas": ["Report:\n{text}", "Passage:\n{text}"],
    "books": ["Story:\n{text}", "Transcript:\n{text}"],
}

QA_instructs = {
    "arxiv": [
        "As a human instructor assessing students' comprehension of a scientific article, you craft a concise question that ideally requires a short phrase or sentence to answer. If the article lacks the necessary information, the answer should be 'unanswerable'. For yes/no questions, reply with 'yes', 'no', or 'unanswerable'. Then, supply the gold answer.",
        "You're a teacher aiming to evaluate how well students grasp a given scientific text. Create a question that can be answered briefly—preferably in one sentence or less. If the article doesn't contain the answer, use 'unanswerable'. For yes/no questions, respond with 'yes', 'no', or 'unanswerable'. Provide the gold answer afterward.",
        "As a human educator, your goal is to check student understanding of a scientific article. You should write a question that can be answered concisely. If the article doesn't provide the necessary details, mark the answer as 'unanswerable'. For binary questions, use 'yes', 'no', or 'unanswerable'. Then give the correct answer.",
        "Acting as a human teacher, you're evaluating students' knowledge of a scientific article. Create a question that is answerable in a single phrase or sentence. If there's not enough information in the article, mark it as 'unanswerable'. For yes/no questions, use 'yes', 'no', or 'unanswerable'. Then provide the gold answer.",
        "You are an educator reviewing a scientific article for comprehension. Formulate a concise question that a student could answer in a phrase or sentence. If the article lacks sufficient data, label the answer 'unanswerable'. For yes/no questions, reply with 'yes', 'no', or 'unanswerable'. Follow with the gold answer.",
        "Your role is that of a teacher checking how well students understood a scientific text. Write a brief question that can be answered in a short sentence. If the article does not support an answer, use 'unanswerable'. For yes/no types, answer with 'yes', 'no', or 'unanswerable'. Then give the correct answer.",
        "Imagine you’re a human instructor assessing students on a scientific article. Frame a question that should be answerable succinctly. If the content doesn’t allow for an answer, respond with 'unanswerable'. If it’s a yes/no question, limit the answer to 'yes', 'no', or 'unanswerable'. Then state the gold answer.",
        "As a teacher evaluating understanding of a scientific paper, design a short-answer question. It should be answerable in one phrase or sentence. If the article doesn’t provide the answer, return 'unanswerable'. For yes/no questions, use 'yes', 'no', or 'unanswerable'. Then, give the correct answer.",
        "You're a human teacher checking comprehension of a scientific article. Write a question that invites a brief answer. If the article lacks the relevant information, mark it 'unanswerable'. For yes/no questions, choose from 'yes', 'no', or 'unanswerable'. Then provide the gold standard answer.",
        "Take the role of a teacher testing students on a scientific article. Pose a concise question—ideally answerable in a phrase. If no answer can be found in the article, use 'unanswerable'. For yes/no questions, restrict your answer to 'yes', 'no', or 'unanswerable'. Then provide the gold answer.",
    ],
    "atlas": [
        "Given a passage from an encyclopedia, your task is to generate a question related to its topic that can be answered briefly—in a phrase or sentence. Then, provide the gold answer.",
        "You are provided with an encyclopedia excerpt. Create a question focused on the topic of the passage that can be answered concisely. Follow it with the correct (gold) answer.",
        "From the given encyclopedia passage, formulate a topic-related question that requires only a short phrase or sentence to answer. Then supply the gold answer.",
        "Using the provided encyclopedia text, construct a concise question that aligns with the passage's subject. Afterward, include the gold answer.",
        "You’re given a passage from an encyclopedia. Write a question on its topic that can be answered in a brief phrase or sentence, and then give the correct answer.",
    ],
    "books": [
        "You are given a story from a book. Your task is to create a question that can be answered in a short phrase or sentence. Then, provide the gold answer.",
        "From the provided book excerpt, formulate a question that can be answered concisely in a phrase or sentence. After that, supply the gold answer.",
        "Given a passage from a book, your task is to generate a question related to the story that can be answered briefly—in a phrase or sentence. Then, provide the gold answer.",
        "Using the provided book text, construct a concise question that can be answered in a short phrase or sentence. Afterward, include the gold answer.",
        "You are given a passage from a book. Write a question about the story that can be answered in a brief phrase or sentence, and then give the correct answer.",
    ],
}

SUM_INSTRUCTS = [
    "You are given a full book. Your task is to summarize the passage in a concise manner, ideally in a few sentences.",
    "You are given a story. Your task is to summarize the passage in a one-page document, focusing on the main points and key details.",
]


@dataclass
class Batch:
    q_passages: list[str]
    distractor_passages: list[str] = None
    full_passages: list[str] = None


def dataset_from_file(
    file_path,
    n_samples: int = 1,
):
    passages = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            data = json.loads(line)["text"]
            passages.append(data)
            if len(passages) == n_samples:
                yield passages
                passages = []


def dataloader_from_file(
    file_path,
    batch_size,
    n_samples: int = 1,
    n_q_passages: int = 1,
):
    dataset = dataset_from_file(file_path, n_samples=n_samples)

    batch_list = []
    while True:
        data = next(dataset)
        id_q_passages = random.sample(range(len(data)), k=n_q_passages)
        data_q_passages = [data[i] for i in id_q_passages]
        data_distractor_passages = [
            data[i] for i in range(len(data)) if i not in id_q_passages
        ]
        batch_list.append(
            Batch(
                q_passages=data_q_passages,
                distractor_passages=data_distractor_passages,
                full_passages=data,
            )
        )
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []


def synthesize_summ_data(
    model_folder_path: str,
    batch_size: int,
    data_path: str,
    output_path: str,
    download_freq: int,
    ds_name: str,
):
    dataset = (
        "atlas"
        if "atlas" in data_path.lower()
        else ("arxiv" if "arxiv" in data_path.lower() else "books")
    )

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file_path = output_path + ds_name + model_folder_path.split("/")[-1] + ".jsonl"

    llm = LLM(model=model_folder_path, dtype="bfloat16", max_model_len=16384)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        truncate_prompt_tokens=16384 - 1024,
    )
    dataloader = dataloader_from_file(
        data_path,
        batch_size,
        n_samples=10 if dataset == "atlas" else 1,
        n_q_passages=10 if dataset == "atlas" else 1,
    )

    instruct = SUM_INSTRUCTS

    output_buffer = []
    n_samples = 0
    step = 0
    while True:
        step += 1
        try:
            batch = next(dataloader, None)
        except StopIteration:
            batch = None
        if batch is None:
            print("No more batches to process.")
            break
        n_samples += len(batch)

        if dataset == "atlas":
            messages = [
                [
                    {
                        "role": "system",
                        "content": "Follow the instruction  without any additional comments! "
                        + random.choice(instruct),
                    },
                    {
                        "role": "user",
                        "content": random.choice(PREFIX[dataset]).format(
                            text="\n\n".join(sample.q_passages)
                        )[-42000:],
                    },
                ]
                for sample in batch
            ]
            try:
                outputs = llm.chat(messages, sampling_params)
            except ValueError as e:
                print("Skipping this batch", e)
                print('Len outputs', len(outputs))
                print('Batch', len(batch))
                continue
            for i, output in enumerate(outputs):
                if output.finished:
                    output_buffer.append(
                        {
                            "q_passages": batch[i].q_passages,
                            "full_output": output.outputs[0].text.strip(),
                        }
                    )

        else:
            messages = []
            n_submessages = []
            for sample in batch:
                # Split the text into sentences
                
                sentences = sample.q_passages[0].split("\n\n")
        
                if len(sentences) <= 10:
                    sentences = sent_tokenize(sample.q_passages[0])

                # Randomly select a subset of sentences to form the context
                for i in range(0, len(sentences), len(sentences) // 10):
                    context = " ".join(sentences[i : i + len(sentences) // 10]) 
                    messages.append(
                        [
                            {
                                "role": "system",
                                "content": "Follow the instruction  without any additional comments! "
                                + "You are given a story from a book. Your task is to summarize the story in a precise manner, ideally in several sentences but feel free to use as many as necessary. Do not exceed one page.",
                            },
                            {
                                "role": "user",
                                "content": random.choice(PREFIX[dataset]).format(
                                    text=context
                                )[-40000:],
                            },
                        ]
                    )
                n_submessages.append(len(range(0, len(sentences), len(sentences) // 10)))
                
            outputs = llm.chat(messages, sampling_params)
            for i, output in enumerate(outputs):
                if output.finished:
                    output_buffer.append(
                        {
                            "q_passages": messages[i][1]['content'],
                            "full_output": outputs[i].outputs[0].text.strip(),
                        }
                    )
            ind = 0
            new_messages = []
            for i, n_sub in enumerate(n_submessages):
                submessages = []
                for j in range(n_sub):
                    submessages.append(outputs[ind + j].outputs[0].text.strip())
                ind += n_sub
                new_messages.append(
                    [
                        {
                            "role": "system",
                            "content": "Follow the instruction  without any additional comments! "
                            + random.choice(instruct),
                        },
                        {
                            "role": "user",
                            "content": random.choice(PREFIX[dataset]).format(
                                text="\n".join(submessages)
                            )[-40000:],
                        },
                    ]
                )
            print('SUBMESSAGES: ', "\n".join(submessages))
            outputs = llm.chat(new_messages, sampling_params)

            for i, output in enumerate(outputs):
                if output.finished:
                    output_buffer.append(
                        {
                            "q_passages": batch[i].q_passages[0],
                            "full_output": outputs[i].outputs[0].text.strip(),
                        }
                    )

        if len(output_buffer) >= download_freq:
            print(
                "Current step:",
                step,
                "N SAMPLES:",
                n_samples,
                "Example:",
                output_buffer[-1],
            )
            with open(out_file_path, "a") as f:
                for item in output_buffer:
                    f.write(json.dumps(item) + "\n")
            output_buffer = []

        with open(out_file_path, "a") as f:
            for item in output_buffer:
                f.write(json.dumps(item) + "\n")


def synthesize_qa_data(
    model_folder_path: str,
    batch_size: int,
    data_path: str,
    output_path: str,
    download_freq: int,
    ds_name: str,
):
    dataset = (
        "atlas"
        if "atlas" in data_path.lower()
        else ("arxiv" if "arxiv" in data_path.lower() else "books")
    )

    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file_path = output_path + ds_name + model_folder_path.split("/")[-1] + ".jsonl"

    llm = LLM(model=model_folder_path, dtype="bfloat16", max_model_len=16384)
    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=128, truncate_prompt_tokens=16384 - 128
    )
    dataloader = dataloader_from_file(
        data_path,
        batch_size,
        n_samples=10 if dataset == "atlas" else 1,
        n_q_passages=1,
    )

    instruct = QA_instructs.get(dataset)

    output_buffer = []
    n_samples = 0
    step = 0
    while True:
        step += 1
        try:
            batch = next(dataloader, None)
        except StopIteration:
            batch = None
        if batch is None:
            print("No more batches to process.")
            break
        n_samples += len(batch)

        if dataset == "atlas":
            messages = [
                [
                    {
                        "role": "system",
                        "content": "Follow the instruction  without any additional comments! "
                        + random.choice(instruct),
                    },
                    {
                        "role": "user",
                        "content": random.choice(PREFIX[dataset]).format(
                            text="\n\n".join(sample.q_passages)
                        )[-40000:],
                    },
                ]
                for sample in batch
            ]

        else:
            messages = []
            for sample in batch:
                # Split the text into sentences
                sentences = sample.q_passages[0].split("\n\n")
                if len(sentences) < 10:
                    sentences = sent_tokenize(sample.q_passages[0])
                # Randomly select a subset of sentences to form the context
                if len(sentences) > 5:
                    end_index = random.randint(5, len(sentences))
                    context = " ".join(sentences[:end_index])
                else:
                    context = " ".join(sentences)

                messages.append(
                    [
                        {
                            "role": "system",
                            "content": "Follow the instruction  without any additional comments! "
                            + random.choice(instruct),
                        },
                        {
                            "role": "user",
                            "content": random.choice(PREFIX[dataset]).format(
                                text=context
                            )[-40000:],
                        },
                    ]
                )
        try:
            outputs = llm.chat(messages, sampling_params)
        except ValueError as e:
            print("Skipping this batch", e)
            print('Len outputs', len(outputs))
            print('Batch', len(batch))
            continue
        if len(outputs) != len(batch):
            print(
                "Warning: Number of outputs does not match number of samples in batch.",
                len(outputs),
                len(batch),
            )
            continue
        for i, output in enumerate(outputs):
            if output.finished:
                output_buffer.append(
                    {
                        "q_passages": batch[i].q_passages,
                        "full_passages": batch[i].full_passages,
                        "full_output": output.outputs[0].text.strip(),
                    }
                )
        if len(output_buffer) >= download_freq:
            print(
                "Current step:",
                step,
                "N SAMPLES:",
                n_samples,
                "Example:",
                output_buffer[-1],
            )
            with open(out_file_path, "a") as f:
                for item in output_buffer:
                    f.write(json.dumps(item) + "\n")
            output_buffer = []

    with open(out_file_path, "a") as f:
        for item in output_buffer:
            f.write(json.dumps(item) + "\n")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer",
        type=str,
        default="google/gemma-3-27b-it",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=ATLAS_PATH,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )

    parser.add_argument(
        "--download_freq",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--synthesize_summ",
        action="store_true",
        help="If set, will synthesize summarization data.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    dataset = (
        "atlas"
        if "atlas" in args.data_path.lower()
        else ("arxiv" if "arxiv" in args.data_path.lower() else "books")
    )
    if args.synthesize_summ:
        synthesize_summ_data(
            model_folder_path=args.transformer,
            batch_size=args.batch_size,
            data_path=args.data_path,
            output_path=args.output_path,
            download_freq=args.download_freq,
            ds_name="summ_" + dataset + "_" + args.transformer.split("/")[-1],
        )

    else:
        synthesize_qa_data(
            model_folder_path=args.transformer,
            batch_size=args.batch_size,
            data_path=args.data_path,
            output_path=args.output_path,
            download_freq=args.download_freq,
            ds_name="qa_" + dataset + "_" + args.transformer.split("/")[-1],
        )
