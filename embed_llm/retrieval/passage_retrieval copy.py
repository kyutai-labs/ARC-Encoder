# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import pickle
import torch
import glob
from pathlib import Path
import numpy as np
from embed_llm.retrieval.index import Indexer
from embed_llm.retrieval.embeddings import get_pretrained_embedder, encode_text
from embed_llm.monitoring.utils import set_logger
import time
import json
import torch.nn.functional as F
from tqdm import tqdm
import argparse

logger = logging.getLogger(__name__)


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])

    for file_path in tqdm(
        embedding_files, desc="Indexing data", total=len(embedding_files)
    ):
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            dico = pickle.load(fin)
        ids, embeddings = dico["ids"], dico["embeddings"]

        allids.extend(ids)
        allembeddings = (
            np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        )
        assert allembeddings.shape[0] == len(allids)

        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(
                index, allembeddings, allids, indexing_batch_size
            )

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(
            index, allembeddings, allids, indexing_batch_size
        )

    logger.info("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def retrieved_passage_4QA(
    path_QA: str,
    output_path: str,
    n_retrieved_doc: int,
    embed_dim: int = 4096,
    n_subquantizers: int = 8,
    n_bits: int = 8,
    indexing_batch_size: int = 1024,
    pathname_embeddings: str = "",
    save_or_load_index: bool = True,
    model_name: str = "NVEmbed",
    split: str = "train",
    batch_size: int = 16,
    passages_path: str = "",
):

    if isinstance(output_path, str):
        output_path = [output_path]

    for out_path in output_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    model = get_pretrained_embedder(model_name)
    index = Indexer(embed_dim, n_subquantizers, n_bits)
    # index all passages
    input_paths = glob.glob(pathname_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = Path(input_paths[0]).parent
    (embeddings_dir / Path(split)).mkdir(parents=True, exist_ok=True)
    index_path = embeddings_dir / Path(split) / "index.faiss"

    if save_or_load_index and index_path.exists():
        index.deserialize_from(embeddings_dir / Path(split))
    else:
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, indexing_batch_size)

        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if save_or_load_index:
            index.serialize(embeddings_dir / Path(split))

    if isinstance(path_QA, str):
        path_QA = [path_QA]

    assert len(path_QA) == len(output_path)

    logger.info(f"Loading passages from {passages_path}")
    with open(passages_path, "r") as fin:
        all_passages = {
            dic["id"]: dic["text"] for line in fin for dic in json.loads(line)
        }

    for qa_path, out_path in zip(path_QA, output_path):

        logger.info(f"Embedding questions from {qa_path}")
        # Embed questions
        total_QA = 0
        with open(qa_path, "r") as file:
            for line in file:
                total_QA += 1

        queries = []
        answers = []
        batch_query = []
        embeddings = []
        with open(qa_path, "r") as file:
            for i, line in tqdm(enumerate(file), desc="Embed queries", total=total_QA):
                data = json.loads(line)
                queries.append(data["question"])
                batch_query.append(
                    data["question"].split("\n\n")[0]
                )  # If multi_option question, only take the question and not the possible answers
                answers.append(data["answer"])

                if (i + 1) % batch_size == 0:

                    embeds = encode_text(
                        batch_query,
                        model_name=model_name,
                        model=model,
                        query_embedding=True,
                        device=model.device,
                        pool=False,
                    )
                    embeds = F.normalize(embeds, p=2, dim=1)
                    embeddings.append(embeds)
                    batch_query = []

            if len(batch_query) > 0:
                embeds = encode_text(
                    batch_query,
                    model_name=model_name,
                    model=model,
                    query_embedding=True,
                    device=model.device,
                    no_pool=False,
                )

                embeds = F.normalize(embeds, p=2, dim=1)
                embeddings.append(embeds)
                batch_query = []

        embeds = torch.cat(embeddings, dim=0)

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(
            query_vectors=embeds,
            top_docs=n_retrieved_doc,
            index_batch_size=indexing_batch_size,
        )
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        paired_passages = []
        for results_and_scores, query, answer in tqdm(
            zip(top_ids_and_scores, queries, answers),
            desc="Retrieving similar passages",
            total=total_QA,
        ):
            paired_passage = []

            for id in results_and_scores[0]:
                paired_passage.append(all_passages[id])

            paired_passages.append(
                {
                    "question": query,
                    "passages": paired_passage,
                    "answer": answer,
                    "scores": [str(score) for score in results_and_scores[1]],
                }
            )

        with open(output_path, "w") as fout:
            for entry in paired_passages:
                json.dump(entry, fout)
                fout.write("\n")

        logger.info(f"Saved results to {output_path}")


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
        default=64,
        help="Batch size for encoding queries",
    )

    parser.add_argument("--n_subquantizers", type=int, default=64)

    parser.add_argument("--n_bits", type=int, default=8)

    parser.add_argument("--idx_bs", type=int, default=100000)

    parser.add_argument("--n_retrieved_doc", type=int, default=5)

    return parser


if __name__ == "__main__":
    # Create index for different datasets
    parser = arg_parser()
    args = parser.parse_args()

    datapath = [
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_open_data/eval.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/triviaqa_data/test.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_data_old/test.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/commonsense_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/freebase_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/web_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/wiki_qa_good_answer.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/wiki_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/yahoo_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_open_data/train.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_data_old/train.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/dùatasets/Question_Answering/triviaqa_data/train.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/msmarco_qa.jsonl",
    ]

    output_path = [
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed_PQ/nq_open_data.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed_PQ/triviaqa_data.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed_PQ/nq_data_old.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/commonsense_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/freebase_qa.jsonl  ",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/web_qa.jsonl  ",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/wiki_qa_good_answer.jsonl  ",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/wiki_qa.jsonl  ",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/yahoo_qa.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/nq_open_data.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/nq_data_old.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/triviaqa_data.jsonl",
        "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed_PQ/msmarco_qa.jsonl ",
    ]

    set_logger(logging.INFO)
    output_path = (
        args.save_output_path if args.save_output_path is not None else output_path
    )
    datapath = (
        args.data_name_to_load if args.data_name_to_load is not None else datapath
    )

    retrieved_passage_4QA(
        path_QA=datapath,
        output_path=output_path,
        n_retrieved_doc=args.n_retrieved_doc,
        embed_dim=4096,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits,
        indexing_batch_size=args.idx_bs,  # Should use a large enough batch to train the IndexPQ
        pathname_embeddings=r"/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/atlas_passages_embeddings/NVEmbed/*_embeddings_*.pkl",
        save_or_load_index=True,
        model_name="NVEmbed",
        split="all_indexed_PQ",
        batch_size=args.batch_size,
    )
