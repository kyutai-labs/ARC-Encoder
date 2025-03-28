# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# From https://github.com/facebookresearch/FiD/blob/main/passage_retrieval.py
import logging
import pickle
import faiss
import numpy as np
from pathlib import Path
import torch
from typing import Union

logger = logging.getLogger()


class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            self.index = faiss.IndexPQ(
                vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_db_id = np.empty((0), dtype=np.int64)

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype("float32")
        if not self.index.is_trained:
            logger.info(f"Training index on {embeddings.shape[0]} vectors")
            self.index.train(embeddings)
        self.index.add(embeddings)

        # logger.info(f"Total data indexed {len(self.index_id_to_db_id)}")

    def search_knn(
        self,
        query_vectors: Union[np.array, torch.Tensor],
        top_docs: int,
        index_batch_size=1024,
    ) -> list[tuple[list[object], list[float]]]:
        query_vectors = query_vectors.cpu().numpy().astype("float32")
        result = []
        nbatch = (len(query_vectors) - 1) // index_batch_size + 1
        for k in range(nbatch):
            start_idx = k * index_batch_size
            end_idx = min((k + 1) * index_batch_size, len(query_vectors))
            q = query_vectors[start_idx:end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [
                [str(self.index_id_to_db_id[i]) for i in query_top_idxs]
                for query_top_idxs in indexes
            ]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "index_meta.dpr"
        logger.info(f"Serializing index to {index_file}, meta data to {meta_file}")
        faiss.write_index(
            faiss.index_gpu_to_cpu(self.index), Path(index_file).as_posix()
        )
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = dir_path / "index.faiss"
        meta_file = dir_path / "index_meta.dpr"
        logger.info(f"Loading index from {index_file}, meta data from {meta_file}")

        self.index = faiss.read_index(Path(index_file).as_posix())

        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: list):
        new_ids = np.array(db_ids, dtype=np.int64)
        self.index_id_to_db_id = np.concatenate(
            (self.index_id_to_db_id, new_ids), axis=0
        )
