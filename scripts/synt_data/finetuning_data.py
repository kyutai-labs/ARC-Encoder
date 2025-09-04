import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams  # type: ignore
from transformers import AutoTokenizer
from dataclasses import dataclass
import numpy as np
import random
from embed_llm import DATA_PATH
import re  # noqa: F401


ATLAS_PATH = DATA_PATH + 'raw/Atlas_passages_validation.jsonl'


translate_prompts = [
    "Translate the previous document into {language}.",
    "Render the document into fluent {language} while preserving its meaning.",
    "Provide a {language} translation of the text above.",
    "As a translator, convert the document into {language} while maintaining its original meaning.",
    "Translate the document into {language} while ensuring clarity and accuracy.",
]

LANGUAGES = ["Spanish", "French", "German", "Danish"]
LOW_RESSOURCE_LANGUAGES = [
    "Hindi",
    "Russian",
    "Swahili",
    "Arabic",
    "Turkish",
    "Japanese",
    "Finnish",
    "Chinese (simplified)",
]



def passage_filter(passage: str, min_alpha_ratio: float = 0.75) -> bool:
    """
    Filter the passage to remove unwanted characters and format it.
    """
    if len(passage) < 10 or len(passage) > 16000:
        return False
    alpha_count = sum(c.isalpha() for c in passage)
    return alpha_count / len(passage) >= min_alpha_ratio


@dataclass
class Batch:
    passage: str
    instruction_prompts: str
    prompt_key: str


def dataset_from_file(
    file_path,
    n_passages: int = 1,
    precise_size: int = None,
    tokenizer: object = None,
):
    sample = []
    n_sample = random.randint(1, n_passages)
    while True:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if passage_filter(data["text"]):
                    if precise_size is not None and tokenizer is not None:
                        # Step 1: Split by double newlines
                        paragraphs = data["text"].strip().split("\n\n")

                        # Step 2: For each paragraph, split by single newlines
                        lines = [para.split("\n") for para in paragraphs]
                        lines = sum(lines, [])  # Flatten the list of lists
                        text = []
                        count = 0
                        for line in lines:
                            if count >= precise_size:
                                break
                            if len(line.strip()) > 0:
                                text.append(line.strip())
                                count += len(tokenizer.encode(line.strip()))
                        text = "\n".join(text)
                    else:
                        text = data["text"].strip()
                    sample.append(text)
                if len(sample) == n_sample:
                    yield ("\n".join(sample))[:15000]
                    n_sample = random.randint(1, n_passages)
                    sample = []


def dataloader_from_file(
    file_path,
    batch_size,
    n_passages: int = 1,
    precise_size: int = None,
    tokenizer: object = None,
):
    dataset = dataset_from_file(
        file_path,
        n_passages,
        precise_size=precise_size,
        tokenizer=tokenizer,
    )
    batch_list = []
    while True:
        language = random.choice(LOW_RESSOURCE_LANGUAGES)
        prompt = random.choice(translate_prompts).format(language=language)
        prompt_key = language
        passage = next(dataset)
        batch_list.append(
            Batch(
                instruction_prompts=prompt,
                passage=passage,
                prompt_key=str(prompt_key),
            )
        )
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []

def synthesize_data(
    temperature: float,
    top_p: float,
    model_folder_path: str,
    batch_size: int,
    data_path: str,
    output_path: str,
    max_steps: int,
    download_freq: int,
    ds_name: str,
    translate: bool = False,
    n_passages: int = 1,
    precise_size: int = None,
):
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file_path = output_path + ds_name + model_folder_path.split("/")[-1] + ".jsonl"

    llm = LLM(model=model_folder_path, dtype="bfloat16", max_model_len=16384)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1024,
    )
    dataloader = dataloader_from_file(
        data_path,
        batch_size,
        n_passages,
        precise_size,
        tokenizer=AutoTokenizer.from_pretrained(model_folder_path),
    )
    output_buffer = []
    n_samples = 0
    for step in range(max_steps):
        batch = next(dataloader)
        n_samples += len(batch)


        text_prompts = [
            "Answer to the instructions without any additional comments!\n\nDocument: "
            + b.passage
            + "\n\n"
            + b.instruction_prompts
            for b in batch
        ]

        outputs = llm.generate(text_prompts, sampling_params)
        for i, output in enumerate(outputs):
            if output.finished:
                question, answer = (
                    batch[i].instruction_prompts,
                    output.outputs[0].text.strip(),
                )
                output_buffer.append(
                    {
                        "question": question,
                        "passage": batch[i].passage.replace("\nDocument: ", "\n\n"),
                        "answer": answer,
                        "full_output": output.outputs[0].text.strip(),
                        "prompt_key": batch[i].prompt_key,
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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer",
        type=str,
        default="google/gemma-3-27b-it",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
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
        "--max_steps",
        type=int,
        default=2000000,
    )

    parser.add_argument(
        "--download_freq",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n_passages",
        default=1,
        type=int,
        help='Number of passages to stack and then translate'
    )

    parser.add_argument("--ds_name", type=str, default="synth_translation_")

    parser.add_argument(
        "--precise_size",
        type=int,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    synthesize_data(
        temperature=args.temperature,
        top_p=args.top_p,
        model_folder_path=args.transformer,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_path=args.output_path,
        max_steps=args.max_steps,
        download_freq=args.download_freq,
        ds_name=args.ds_name,
        n_passages=args.n_passages,
        precise_size=args.precise_size,
    )
