# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "argparse",
#     "dataclasses",
#     "pathlib",
#     "setuptools",
#     "vllm",
#     "numpy",
#     "transformers",
# ]
# ///
import argparse
import json
from pathlib import Path
from vllm import LLM, SamplingParams  # type: ignore
from dataclasses import dataclass
import re  # noqa: F401


PISCO_PATH = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/true_pisco/all_pisco_train.jsonl"


@dataclass
class Batch:
    passages: list[str]
    instruction_prompts: str
    question: str


def dataset_from_file(
    file_path,
):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i <= 452992:
                continue
            data = json.loads(line)
            yield data


def dataloader_from_file(
    file_path,
    batch_size,
):
    dataset = dataset_from_file(
        file_path,
    )

    batch_list = []
    while True:
        data = next(dataset)
        batch_list.append(
            Batch(
                instruction_prompts="You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.",
                passages=[passage.strip() for passage in data["passages"]],
                question=data["question"],
            )
        )
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []


def synthesize_data(
    model_folder_path: str,
    batch_size: int,
    data_path: str,
    output_path: str,
    download_freq: int,
    ds_name: str,
):
    if not Path(output_path).exists():
        Path(output_path).mkdir(parents=True, exist_ok=True)
    out_file_path = output_path + ds_name + model_folder_path.split("/")[-1] + ".jsonl"

    llm = LLM(model=model_folder_path, dtype="bfloat16", max_model_len=16384)
    sampling_params = SamplingParams(
        max_tokens=128,
    )
    dataloader = dataloader_from_file(
        data_path,
        batch_size,
    )

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

        messages = [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer questions as briefly as possible.",
                },
                {
                    "role": "user",
                    "content": "Background: "
                    + "\n".join(sample.passages)
                    + f"\nQuestion: {sample.question}",
                },
            ]
            for sample in batch
        ]

        outputs = llm.chat(messages, sampling_params)
        for i, output in enumerate(outputs):
            if output.finished:
                output_buffer.append(
                    {
                        "question": batch[i].question,
                        "passages": batch[i].passages,
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
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=PISCO_PATH,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/synthesized/",
    )

    parser.add_argument(
        "--download_freq",
        type=int,
        default=1,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    synthesize_data(
        model_folder_path=args.transformer,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_path=args.output_path,
        download_freq=args.download_freq,
        ds_name="pisco_ft_data_",
    )
