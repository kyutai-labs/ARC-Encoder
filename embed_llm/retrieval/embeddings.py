from gritlm import GritLM
from transformers import AutoTokenizer, AutoModel
from embed_llm.retrieval.nvembed.modeling_nvembed import custom_encode
import os
import torch.nn.functional as F
import torch
from tqdm.auto import tqdm
import json
import numpy as np
import pickle
import argparse


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_pretrained_embedder(model_name: str, device_map: str = "auto"):
    if model_name == "GritLM":
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", device_map="auto")
        return model

    elif model_name == "Contriever":
        model = AutoModel.from_pretrained("facebook/contriever", device_map="auto")
        return model
    elif model_name == "NVEmbed":
        model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2",
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype="bfloat16",
        )
        return model
    else:
        raise ValueError(f"Unknown model name {model_name}")


def encode_text(
    text: list[str] | str,
    model_name: str,
    model: GritLM | AutoModel,
    query_embedding: bool = True,
    tokenizer: AutoTokenizer | None = None,
    device: str = "cpu",
    no_pool: bool = False,
):
    if isinstance(text, str):
        text = [text]

    if model_name == "GritLM":
        with torch.no_grad():
            embedding = model.encode(text)
        if device == "cpu":
            return embedding.cpu().numpy()
        else:
            return embedding

    elif model_name == "Contriever":
        tokenizer = (
            AutoTokenizer.from_pretrained("facebook/contriever")
            if tokenizer is None
            else tokenizer
        )
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedding = model(**inputs)
        if device == "cpu":
            embedding = mean_pooling(embedding[0].cpu(), inputs["attention_mask"])
            return embedding.cpu().numpy()
        else:
            embedding = mean_pooling(embedding[0], inputs["attention_mask"])
            return embedding

    elif model_name == "NVEmbed":

        if query_embedding:
            task_name_to_instruct = {
                "example": "Given a question, retrieve passages that answer the question",
            }
            instruction = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
        else:
            instruction = ""

        with torch.no_grad():
            if no_pool:
                embedding, seqlens = custom_encode(
                    model, prompts=text, instruction=instruction, pool=False
                )
            else:
                # If needs a pooled embedding used the HF code (reduce possible mismatch between model and encode function)
                embedding = model.encode(
                    text, instruction=instruction, max_length=32768
                )

        if device == "cpu":
            return (
                (embedding.cpu().numpy(), seqlens)
                if no_pool
                else embedding.cpu().numpy()
            )
        else:
            return (embedding, seqlens) if no_pool else embedding
    else:
        raise ValueError(f"Unknown model name {model_name}")


def generate_embeddings(
    model_name: str,
    output_path: str,
    bs: int,
    dataset_path: str,
    n_gpu: int = 1,
    partition: int = 0,
    checkpoint: int = 1000,
):

    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    model = get_pretrained_embedder(model_name, device_map="cuda")
    os.makedirs(os.path.join(output_path, model_name), exist_ok=True)

    size_partition = len(dataset) // n_gpu
    used_texts = []

    if partition == n_gpu - 1:
        end_partition = len(dataset)
    else:
        end_partition = size_partition * (partition + 1)
    for i, row in tqdm(enumerate(dataset)):
        if i < size_partition * partition:
            continue
        elif i >= end_partition:
            break
        else:
            for passage in row["text"]:  # All passages must be useful
                # Truncate passages on the char level to 1024
                used_texts.append(passage[:1024])
            # All passages must be useful, atlas should already be preprocessed
            # if len(row["text"]) < 20:
            #     continue
            # Truncate passages on the char level to 2048
            # used_texts.append(row["text"][:2048].strip())
            used_texts.append({"id": row["id"], "text": row["text"].strip()})
    count = 0
    embeddings_array = []
    text_passages = []
    for ind, i in tqdm(enumerate(range(0, len(used_texts), bs))):
        passages = used_texts[i : i + bs]

        embeddings = encode_text(
            passages,
            model_name=model_name,
            model=model,
            query_embedding=False,
            device="cuda",
        )
        embeddings = (
            (
                F.normalize(embeddings, p=2, dim=1)
                if model_name == "NVEmbed"
                else embeddings
            )
            .detach()
            .cpu()
            .numpy()
        )

        text_passages.extend(passages)
        embeddings_array.append(embeddings)

        if ind % (checkpoint) == 0 and ind != 0:
            embeddings_array = np.concatenate(embeddings_array, axis=0)
            assert embeddings_array.shape[0] == len(text_passages)
            np.save(
                os.path.join(
                    output_path, model_name, f"{partition}_embeddings_{count}.npy"
                ),
                embeddings_array,
            )
            with open(
                os.path.join(
                    output_path, model_name, f"{partition}_embeddings_{count}.jsonl"
                ),
                "w",
            ) as f:
                for passage in text_passages:
                    json.dump({"text": passage}, f)
                    f.write("\n")

            embeddings_array = []
            text_passages = []
            count += 1

    embeddings_array = np.concatenate(embeddings_array, axis=0)
    assert embeddings_array.shape[0] == len(text_passages)
    np.save(
        os.path.join(output_path, model_name, f"{partition}_embeddings_{count}.npy"),
        embeddings_array,
    )

    with open(
        os.path.join(output_path, model_name, f"{partition}_embeddings_{count}.jsonl"),
        "w",
    ) as f:
        for passage in text_passages:
            json.dump({"text": passage}, f)
            f.write("\n")

    print("Saving embedding dataset with embeddings to", output_path)


def arg_parser():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "-outpath",
        "--save_output_path",
        type=str,
        default=None,
        help="Path to save the output",
    )
    parser.add_argument(
        "-data_path",
        "--data_name_to_load",
        type=str,
        default=None,
        help="Name of the dataset to load",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "-n_gpus",
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs on which the dataset is split",
    )
    parser.add_argument(
        "-partition",
        "--partition",
        type=int,
        default=0,
        help="Partition of the dataset to process",
    )
    return parser


if __name__ == "__main__":
    # Create index for different datasets
    parser = arg_parser()
    args = parser.parse_args()
    output_path = args.save_output_path
    data_path = args.data_name_to_load
    bs = args.batch_size
    output_path = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/atlas_passages_embeddings_2/"
    data_path = "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Atlas/enwiki-dec2021/text-list-100-sec.jsonl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_embeddings(
        "NVEmbed",
        output_path,
        bs,
        data_path,
        n_gpu=args.num_gpus,
        partition=args.partition,
    )
    print("Embeddings generated for NVEmbed")
