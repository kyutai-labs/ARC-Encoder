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
import index
import embeddings
import time
import json

logger = logging.getLogger(__name__)


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        logger.info(f'Loading file {file_path}')
        with open(file_path, 'rb') as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack(
            (allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(
                index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(
            index, allembeddings, allids, indexing_batch_size)

    logger.info('Data indexing completed.')


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids

# TODO


def load_passages(path):
    pass


# TODO
def add_passages(data, passages, top_passages_and_scores):
    pass
    # add passages to original data
    # merged_data = []
    # assert len(data) == len(top_passages_and_scores)
    # for i, d in enumerate(data):
    #     results_and_scores = top_passages_and_scores[i]
    #     docs = [passages[doc_id] for doc_id in results_and_scores[0]]
    #     scores = [str(score) for score in results_and_scores[1]]
    #     ctxs_num = len(docs)
    #     d['ctxs'] =[
    #             {
    #                 'id': results_and_scores[0][c],
    #                 'title': docs[c][1],
    #                 'text': docs[c][0],
    #                 'score': scores[c],
    #             } for c in range(ctxs_num)
    #         ]


if __name__ == '__main__':
    embed_dim = 768
    n_subquantizers = 8
    n_bits = 8
    pathname_embeddings = r'^(\d{1,3})_embeddings_(\d{1,2})\.npy$'
    save_or_load_index = False
    n_retrieved_doc = 10
    indexing_batch_size = 1024
    path_passages = 'data/kilt/kilt_knowledgesource.json'
    output_path = 'data/kilt/kilt_qa_retrieval_results.json'
    data = {}  # TODO see what type required
    embedding_bs = 32
    model_name = "NVEmbed"
    model = embeddings.get_embedder(model_name)

    index = index.Indexer(embed_dim, n_subquantizers, n_bits)

    # index all passages
    input_paths = glob.glob(pathname_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    index_path = embeddings_dir / 'index.faiss'

    if save_or_load_index and index_path.exists():
        index.deserialize_from(embeddings_dir)
    else:
        logger.info(f'Indexing passages from files {input_paths}')
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, indexing_batch_size)
        logger.info(f'Indexing time: {time.time()-start_time_indexing:.1f} s.')
        if save_or_load_index:
            index.serialize(embeddings_dir)

    # A modifier TODO
    questions = data['query']  # TODO
    questions_embedding = embeddings.encode_text(
        questions, model_name=model_name, model=model, bs=embedding_bs)

    # get top k results
    start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, n_retrieved_doc)
    logger.info(f'Search time: {time.time()-start_time_retrieval:.1f} s.')

    passages = load_passages(path_passages)
    # Reformat as we want TODO
    # TODO voir format data etc
    passages = {x[0]: (x[1], x[2]) for x in passages}

    add_passages(data, passages, top_ids_and_scores)

    # TODO see what we want to do with the result
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fout:
        json.dump(data, fout, indent=4)
    logger.info(f'Saved results to {output_path}')
