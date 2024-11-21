from gritlm import GritLM
from transformers import AutoTokenizer, AutoModel
import os
import torch.nn.functional as F
import torch
from tqdm.auto import tqdm
import json
import numpy as np


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_embedder(model_name: str, device_map: str = "auto"):
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
            device_map="auto",
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
):
    if isinstance(text, str):
        text = [text]

    if model_name == "GritLM":
        results = []
        with torch.no_grad():
            embedding = model.encode(text)
        if device == "cpu":
            results.append(embedding.cpu().numpy())
            return np.concatenate(results, axis=0)
        else:
            results.append(embedding)
            return torch.cat(results, dim=0)

    elif model_name == "Contriever":
        tokenizer = (
            AutoTokenizer.from_pretrained("facebook/contriever")
            if tokenizer is None
            else tokenizer
        )
        results = []
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedding = model(**inputs)
        if device == "cpu":
            embedding = mean_pooling(embedding[0].cpu(), inputs["attention_mask"])
            results.append(embedding.cpu().numpy())
            return np.concatenate(results, axis=0)
        else:
            embedding = mean_pooling(embedding[0], inputs["attention_mask"])
            results.append(embedding)
            return torch.cat(results, dim=0)

    elif model_name == "NVEmbed":

        if query_embedding:
            task_name_to_instruct = {
                "example": "Given a question, retrieve passages that answer the question",
            }
            instruction = "Instruct: " + task_name_to_instruct["example"] + "\nQuery: "
        else:
            instruction = ""

        results = []
        with torch.no_grad():
            embedding = model.encode(text, instruction=instruction)
        if device == "cpu":
            results.append(F.normalize(embedding, p=2, dim=1).cpu().numpy())
            return np.concatenate(results, axis=0)
        else:
            results.append(F.normalize(embedding, p=2, dim=1))
            return torch.cat(results, dim=0)

    else:
        raise ValueError(f"Unknown model name {model_name}")


# Maybe modify truncation
def generate_embeddings(
    model_name: str,
    output_path: str,
    bs: int,
    dataset,
    n_gpu: int = 1,
    partition: int = 0,
    checkpoint: int = 1000,
):
    model = get_embedder(model_name)
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
    count = 0
    embeddings_array = []
    text_passages = []
    for ind, i in tqdm(enumerate(range(0, len(used_texts), bs))):
        passages = used_texts[i : i + bs]
        embeddings = encode_text(passages, model_name=model_name, model=model)

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
