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
from embed_llm.retrieval.embeddings import get_embedder, encode_text
import time
import json

logger = logging.getLogger(__name__)


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    allpassages = []
    for i, file_path in enumerate(embedding_files):
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)
            
        with open(file_path.replace('npy','jsonl'), "r") as text_file:
            passages = [json.loads(line)['text'] for line in text_file]
    

        allembeddings = (
            np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        )
        allpassages.extend([{'id':ids, 'text': passage}  for passage in passages])
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
    return allpassages
    



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
                            pathname_embeddings = r"^/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/KILT/NVEmbed/(\d{1,3})_embeddings_(\d{1,2})\.npy$",
                            save_or_load_index: bool = True,
                            model_name: str = "NVEmbed"):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    indexing_batch_size = 1024
  
    model = get_embedder(model_name)
    index = Indexer(embed_dim, n_subquantizers, n_bits)
    # index all passages
    input_paths = glob.glob(pathname_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / "index.faiss"

    if save_or_load_index and index_path.exists():
        index.deserialize_from(embeddings_dir)
    else:
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        allpassages = index_encoded_data(index, input_paths, indexing_batch_size)
        with open(embeddings_dir / "allpassages.jsonl", "w") as fout:
            for passage in allpassages:
                json.dump(passage, fout)
                fout.write('\n')
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir)

    passages = []
    with open(path_passages, "r") as file:
        for line in file:
            data = json.loads(line)
            cur_pos = 0
            while cur_pos < len(data['text']):
                if cur_pos + 512 < len(data['text']):
                    passages.append(data['text'][cur_pos:cur_pos+512])
                    cur_pos += 512
                else:
                    break
            
                if len(passages)>32:
                    embeds = encode_text(
                        passages,
                        model_name=model_name,
                        model=model,
                        query_embedding=True,
                        device=model.device,
                        cross_att=False,
                    )
                

                    # get top k results
                    start_time_retrieval = time.time()
                    top_ids_and_scores = index.search_knn(embeds, n_retrieved_doc)
                    logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

                    paired_passages = []
                    for (doc_ids, _), passage in zip(top_ids_and_scores, passages):
                        paired_passage = []
                        with open(embeddings_dir / "allpassages.jsonl", "r") as fin:
                            for line in fin:
                                if json.loads(line)['id'] in doc_ids:
                                    paired_passage.append(json.loads(line)['text'])
                                   
                        paired_passages.append({'passage': passage, 'paired_passages': paired_passage})
                    
                 


                    with open(output_path, "a") as fout:
                        for entry in paired_passages:
                            json.dump(entry, fout)
                            fout.write('\n')
                        
                    logger.info(f"Saved results to {output_path}")
                    passages = []

    

if __name__ == "__main__":
    pass