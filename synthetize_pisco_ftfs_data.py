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


PISCO_PATH = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/true_pisco/all_pisco_train_v2.jsonl"
QA_DATA_PATH = [
    "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/Reading_Comp/squad_v2_only_answered.jsonl",
    "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/Reading_Comp/hotpot_train_good_format.jsonl",
    "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/unfiltered_nocontext_triviaqa/trivia_qa_train.jsonl",
    "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/nq_open_data.jsonl",
]


@dataclass
class Batch:
    passages: list[str]
    dataset: str
    question: str
    label: list[str] 


def dataset_from_file(
    file_path,
):
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
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
        try:
            # Get the next item from the dataset
            data = next(dataset)
        except StopIteration:
            yield None
        if data.get("passages", None) is not None:
            passages = data["passages"]
        elif data.get("passage", None) is not None:
            passages = data["passage"]
        
        if isinstance(passages, str):
            passages = [passages]
        batch_list.append(
            Batch(
                dataset=data.get("dataset", file_path.split("/")[-1].split(".")[0]),
                passages=[passage.strip() for passage in passages],
                question=data["question"],
                label=data.get("label", [data["answer"]] if isinstance(data["answer"], str) else data["answer"]),
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
    dataloader_4_icl = dataloader_from_file(
        data_path,
        1,
    )
    dataloader_4_real = dataloader_from_file(
        data_path,
        batch_size,
    )

    icl_exs = {}

    for batch in dataloader_4_icl:
        if batch is None:
            print("No more batches to process for ICL examples.")
            break
        if batch[0].dataset not in icl_exs:
            icl_exs[batch[0].dataset] = []
            passage = "\n".join(batch[0].dataset)
            icl_exs[batch[0].dataset].append(
                f"Document: {passage}\nQuestion: {batch[0].question}\nAnswer: {batch[0].label[0]}"
            )
        elif batch[0].dataset in icl_exs and len(icl_exs[batch[0].dataset]) < 5:
            passage = "\n".join(batch[0].passages)
            icl_exs[batch[0].dataset].append(
                f"Document: {passage}\nQuestion: {batch[0].question}\nAnswer: {batch[0].label[0]}"
            )
        else:
            continue

    output_buffer = []
    n_samples = 0
    step = 0
    while True:
        step += 1
        try:
            batch = next(dataloader_4_real, None)
        except StopIteration:
            batch = None
        if batch is None:
            print("No more batches to process.")
            break
        n_samples += len(batch)

        prompts = [
            "\n\n".join(icl_exs[batch[0].dataset])
            + "\n\nDocument: "
            + "\n".join(sample.passages)
            + f"\nQuestion: {sample.question}\nAnswer:"
            for sample in batch
        ]

        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            if output.finished:
                output_buffer.append(
                    {
                        "question": batch[i].question,
                        "passages": batch[i].passages,
                        "full_output": output.outputs[0].text.strip(),
                        "gt_label": batch[i].label,
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
        default="mistralai/Mistral-7B-v0.3",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=PISCO_PATH,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/synthesized/silver_data/",
    )

    parser.add_argument(
        "--download_freq",
        type=int,
        default=1,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # synthesize_data(
    #     model_folder_path=args.transformer,
    #     batch_size=args.batch_size,
    #     data_path=args.data_path,
    #     output_path=args.output_path,
    #     download_freq=args.download_freq,
    #     ds_name="pisco_ftfs_data_",
    # )

    for qa_data_path in QA_DATA_PATH:
        synthesize_data(
            model_folder_path=args.transformer,
            batch_size=args.batch_size,
            data_path=qa_data_path,
            output_path=args.output_path,
            download_freq=args.download_freq,
            ds_name=qa_data_path.split("/")[-1].split(".")[0] + "_",
        )
