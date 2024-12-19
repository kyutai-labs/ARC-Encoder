# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import pickle
import glob
from pathlib import Path
import numpy as np
from embed_llm.retrieval.index import Indexer
from embed_llm.retrieval.embeddings import get_pretrained_embedder, encode_text
import time
import json
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    allpassages_ids = []

    for i, file_path in tqdm(
        enumerate(embedding_files), desc="Indexing data", total=len(embedding_files)
    ):
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            dic = pickle.load(fin)
            ids, embeddings = dic["ids"], dic["embeddings"]

        with open(file_path.replace("pkl", "jsonl"), "r") as text_file:
            passages = [json.loads(line)["text"] for line in text_file]

        allembeddings = (
            np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        )
        allpassages_ids.extend([{str(id): passage} for id, passage in zip(ids, passages)])
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(
            index, allembeddings, allids, indexing_batch_size
        )

    logger.info("Data indexing completed.")
    return allpassages_ids

def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def create_similar_passage_ds(
    path_passages: str,
    output_path: str,
    n_retrieved_doc: int,
    embed_dim: int = 4096,
    n_subquantizers: int = 8,
    n_bits: int = 8,
    indexing_batch_size: int = 1024,
    pathname_embeddings=r"^/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/KILT/NVEmbed/(\d{1,3})_embeddings_(\d{1,2})\.npy$",
    save_or_load_index: bool = True,
    model_name: str = "NVEmbed",
    split: str = "train"
):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_pretrained_embedder(model_name)
    index = Indexer(embed_dim, n_subquantizers, n_bits)
    # index all passages
    input_paths = glob.glob(pathname_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    (embeddings_dir / Path(split)).mkdir(parents=True, exist_ok=True)
    index_path = embeddings_dir / Path(split) / "index.faiss"

    if save_or_load_index and index_path.exists():
        index.deserialize_from(embeddings_dir / Path(split) )
    else:
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        allpassages_ids = index_encoded_data(index, input_paths, indexing_batch_size)
        with open(embeddings_dir / Path(split) / "allpassages.jsonl", "w") as fout:
            for passage_id in allpassages_ids:
                 json.dump(passage_id, fout) 
                 fout.write("\n")
        
                
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir / Path(split))

    passages = []

    total_passages = 0
    with open(path_passages, "r") as file:
        for line in file:
            total_passages += 1
    
    with open(path_passages, "r") as file:
        for i, line in tqdm(enumerate(file), desc="Retrieving similar passages", total=total_passages):
            data = json.loads(line)
            cur_pos = 0
            while cur_pos < len(data["text"]):
                if cur_pos + 512 < len(data["text"]):
                    passages.append(data["text"][cur_pos : cur_pos + 512])
                    cur_pos += 512
                else:
                    break

                if len(passages) > 128:
                    embeds = encode_text(
                        passages,
                        model_name=model_name,
                        model=model,
                        query_embedding=True,
                        device=model.device,
                        cross_att=False,
                    )
                    embeds = F.normalize(embeds, p=2, dim=1)

                    # get top k results
                    start_time_retrieval = time.time()
                    top_ids_and_scores = index.search_knn(embeds.cpu().numpy(), n_retrieved_doc)
                    logger.info(
                        f"Search time: {time.time()-start_time_retrieval:.1f} s."
                    )

                    paired_passages = []
                    for (doc_ids, _), passage in zip(top_ids_and_scores, passages):
                        paired_passage = []
                        doc_ids = [str(doc_id) for doc_id in doc_ids]
                        with open(embeddings_dir / Path(split) / "allpassages.jsonl", "r") as fin:
                            for line in fin:
                                dict_content = json.loads(line)
                                id = str(list(dict_content.keys())[0])
                                
                                if id in doc_ids:
                                    paired_passage.append(dict_content[id])
                                    doc_ids.remove(id)
                                    
                                if len(doc_ids) == 0:
                                    break

                        paired_passages.append(
                            {"passage": passage, "paired_passages": paired_passage}
                        )

                    with open(output_path, "a") as fout:
                        for entry in paired_passages:
                            json.dump(entry, fout)
                            fout.write("\n")

                    logger.info(f"Saved results to {output_path}")
                    passages = []


if __name__ == "__main__":
    path = "/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_train.jsonl"
    create_similar_passage_ds(
        path,
        output_path="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/KILT/similar_passages.jsonl",
        n_retrieved_doc=4,
        embed_dim=4096,
        n_subquantizers=8,
        n_bits=8,
        indexing_batch_size=9984,
        pathname_embeddings=r"/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/KILT/NVEmbed/*_embeddings_*.pkl",     
        save_or_load_index=True,
        model_name="NVEmbed",
        split = 'train'
    )
