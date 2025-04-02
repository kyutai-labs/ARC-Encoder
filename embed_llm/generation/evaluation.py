import os
import torch
import json
import numpy as np
import random
from tqdm import tqdm, trange
import argparse
import logging
import subprocess as sp
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from nltk.tokenize import sent_tokenize
from embed_llm.models.augmented_model import EmbedAugPipeline, load_pipeline
from embed_llm.models.utils import is_torchrun
from embed_llm.monitoring.utils import set_logger
from embed_llm.generation.utils import eval_logger_info, ensure_reproducibility
from embed_llm.generation.metrics import (
    word_overlap,
    get_bleu_score,
    get_meteor,
    get_em,
    get_f1_score,
    metric_max_over_ground_truths,
    get_approx_em,
    get_acc_factchecking,
    get_substring_match_score,
)

EVAL_DATA_PATH_COLBERT = {
    "NQ": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_ColBert/nq_open_hf.jsonl",  # nq_data.jsonl
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_ColBert/triviaqa_data.jsonl",
}
EVAL_DATA_PATH = {
    "NQ": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/nq_open_data.jsonl",  # nq_data.jsonl
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/triviaqa_data.jsonl",
    "FactKG": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/factkg_NVEmbed/factkg_test.jsonl",
    "HotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/Hotpot_qa_test.jsonl",
    "SQUAD": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/squad_test.jsonl",
}

METRIC_EVALUATION = {
    "NQ": get_em,
    "TRIVIAQA": get_em,
    "FactKG": get_acc_factchecking,
    "HotpotQA": get_em,
    "SQUAD": get_em,
}


logger = logging.getLogger(__name__)


# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


def create_prompt_prefix(
    queries: list[str],
    answers: list[str],
    docs: list[str] | None = None,
    max_examples: int | None = None,
    fact_checking: bool = False,
) -> str:
    max_examples = max_examples if max_examples is not None else len(queries)

    prompt = ""
    if not fact_checking:
        if docs is not None:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
                prompt += f"Document: {doc}\nQuestion: {query}\nAnswer: {answer}\n\n"
        else:
            for query, answer, _ in zip(queries, answers, range(max_examples)):
                prompt += f"Question: {query}\nAnswer: {answer}\n\n"
    else:
        if docs is not None:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
                prompt += f'Document: {doc}\nVerify the following claims with "True" or "False":\n{query}: {query}\nAnswer: {answer}\n\n'
        else:
            for query, answer, _ in zip(queries, answers, range(max_examples)):
                prompt += f'Verify the following claims with "True" or "False":\n{query}: {query}\nAnswer: {answer}\n\n'
    return prompt


def create_prompt(
    prefix: str, doc: str | list[str], query: str, wdoc: bool = True
) -> str:
    if isinstance(doc, list):
        doc = "\n".join(doc)
    if wdoc:
        return prefix + f"Document: {doc}\nQuestion: {query}\nAnswer:"
    else:
        return prefix + f"Question: {query}\nAnswer:"


def evaluate_QA(
    run_name: str,
    benchmarks: list[str],
    ckpt: int | None = None,
    max_seq_len: int = 256,
    temps: list[float] = [0, 0.5, 0.7, 1],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    icl_examples: int = 0,
    pipeline: EmbedAugPipeline | Transformer | None = None,
    w_embeds: bool = True,  # To test baseline LLM
    query_w_context: bool = True,
    icl_w_context: bool = True,
    mistral: bool = False,
    max_multi_passage: int = 1,
    kilt: bool = False,
    instruct_name: str = None,
    prompt_before_embed: bool = False,
    colbert: bool = False,
    split_to_multipassage: bool = False,
    icl_before_pref: bool = False,
    seed: float = 0.42,
    with_scores: float = 0.0,
    compress_rate: int | None = None,
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    # Loading model
    if not is_torchrun():
        device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
        device_count = torch.cuda.device_count()
        other_device = torch.device("cuda:1") if device_count > 1 else device
    else:
        device = "cuda"
        other_device = None

    pipeline, ckpt = load_pipeline(
        run_name=run_name,
        tmp_path=tmp_path,
        llm_path=llm_path,
        device=device,
        max_bs=max_bs,
        pipeline=pipeline,
        mistral=mistral,
        instruct_name=instruct_name,
        ckpt=ckpt,
        compress_rate=compress_rate,
    )

    if mistral:
        mistral_tokenizer = MistralTokenizer.from_file(
            "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/tokenizer.model.v3"
        ).instruct_tokenizer.tokenizer
        mistral_model = pipeline

    results = {benchmark: {} for benchmark in benchmarks}

    # Creating dataset
    metrics = {}

    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=len(benchmarks)
    ):
        if benchmark == "SQUAD" and max_multi_passage > 1:
            benchmarks.remove(benchmark)
            continue

        metrics[benchmark] = {}
        eval_data = (
            EVAL_DATA_PATH[benchmark]
            if not colbert
            else EVAL_DATA_PATH_COLBERT[benchmark]
        )
        context = []
        questions = []
        answers = []
        scores = []
        with open(eval_data, "r") as f:
            for line in f:
                data = json.loads(line)
                questions.append(data["question"].strip())

                if isinstance(data["answer"], str):
                    answers.append([data["answer"].strip()])

                elif isinstance(data["answer"], list):
                    answers.append(data["answer"])
                else:
                    raise ValueError("Invalid answer type")
                # Take the first ranked retrieved passage

                if max_multi_passage <= 1:
                    context.append(data["passages"][0].strip())

                else:
                    if split_to_multipassage:
                        l_sent = sent_tokenize(data["passages"][0])
                        if len(l_sent) < max_multi_passage:
                            l_sent = data["passages"][0].split(" ")
                        multi_passage = []
                        for i in range(max_multi_passage):
                            multi_passage.append(
                                " ".join(
                                    l_sent[
                                        i * len(l_sent) // max_multi_passage : (i + 1)
                                        * len(l_sent)
                                        // max_multi_passage
                                    ]
                                )
                            )
                        context.append(multi_passage)
                    else:
                        context.append(list(data["passages"][:max_multi_passage]))
                    if "scores" in data.keys() and with_scores > 0.0:
                        l_scores = [
                            float(score) for score in data["scores"][:max_multi_passage]
                        ]
                        if with_scores == 1.0:  # Marche pas
                            centered_scores = (
                                np.array(l_scores) - np.min(l_scores)
                            ) / (np.max(l_scores) - np.min(l_scores))
                        elif with_scores == 2.0:  # Seul qui fonctionne
                            centered_scores = (
                                np.array(l_scores) - np.min(l_scores) + 1e-2
                            ) / (np.max(l_scores) - np.min(l_scores) + 1e-2)
                        elif with_scores == 3.0:  # Bof
                            centered_scores = (
                                np.array(l_scores) - np.mean(l_scores)
                            ) / np.std(l_scores) + 1
                        elif with_scores == 4.0:  # Nul
                            centered_scores = l_scores
                        elif with_scores == 5.0:  # Non
                            centered_scores = (
                                np.array(l_scores) - np.mean(l_scores)
                            ) / np.std(l_scores)
                        scores.append(centered_scores.tolist())

        if with_scores > 0.0:
            print("Centered scores ex:", scores[0])
        if benchmark == "NQ" and kilt:
            test_pairs = []
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/KILT/pair_NVembed_test_NQ.json",
                "r",
            ) as f:
                for i, line in enumerate(f):
                    test_pairs.append(json.loads(line))

            questions = [pair["query"] for pair in test_pairs]
            answers = [pair["answer"] for pair in test_pairs]
            context = [pair["doc"] for pair in test_pairs]

        if len(scores) == 0:
            c = list(zip(questions, context, answers))

            # fixed_random = random.Random()
            # fixed_random.seed(42)
            # fixed_random.shuffle(c)
            random.shuffle(c, random=lambda: seed)
            questions, context, answers = zip(*c)
        else:
            c = list(zip(questions, context, answers, scores))
            random.shuffle(c, random=lambda: seed)
            questions, context, answers, scores = zip(*c)

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")

        if mistral and isinstance(context[0], list):
            context = ["\n".join(doc) for doc in context]

        if icl_w_context:
            prompt_prefix = create_prompt_prefix(
                queries=questions,
                answers=[answer[0] for answer in answers],
                docs=context,
                max_examples=icl_examples,
                fact_checking=(benchmark == "FactKG"),
            )
        else:
            prompt_prefix = create_prompt_prefix(
                queries=questions,
                answers=[answer[0] for answer in answers],
                docs=None,
                max_examples=icl_examples,
                fact_checking=(benchmark == "FactKG"),
            )

        new_context, new_questions, new_answers = (
            list(context[icl_examples:]),
            list(questions[icl_examples:]),
            list(answers[icl_examples:]),
        )

        if len(scores) > 0:
            new_scores = list(scores[icl_examples:])
            new_scores.reverse()

        new_context.reverse()
        new_questions.reverse()
        new_answers.reverse()

        for temp in temps:
            compress_ratio = 0
            generated_sequences = []
            n_samples = len(new_questions) if n_samples is None else n_samples
            for i in trange(0, n_samples, max_bs):
                bs = min(max_bs, n_samples - i)
                if w_embeds:
                    no_context_prompt = [
                        create_prompt(
                            prefix=prompt_prefix, doc="", query=query, wdoc=False
                        )
                        for query in new_questions[i : i + bs]
                    ]

                    context_prompt = [
                        create_prompt(
                            prefix=" answer the question following the examples:\n\n"
                            + prompt_prefix,
                            doc="",
                            query=query,
                            wdoc=False,
                        )
                        for query in new_questions[i : i + bs]
                    ]

                else:
                    if query_w_context:
                        no_context_prompt = [
                            create_prompt(
                                prefix=prompt_prefix,
                                doc=doc,
                                query=query,
                                wdoc=True,
                            )
                            for query, doc in zip(
                                new_questions[i : i + bs],
                                new_context[i : i + bs],
                            )
                        ]
                    else:
                        no_context_prompt = [
                            create_prompt(
                                prefix=prompt_prefix,
                                doc="",
                                query=query,
                                wdoc=False,
                            )
                            for query in new_questions[i : i + bs]
                        ]

                if not mistral:
                    if w_embeds:
                        text_conditioning = list(new_context[i : i + bs])
                    else:
                        text_conditioning = None

                    generated_sequence, embed_tokens, embeds = pipeline.generate(
                        prompt_pre_embed=(
                            [""] * bs
                            if not prompt_before_embed
                            else ["Based on the context "] * bs
                        )
                        if not icl_before_pref
                        else [prompt_prefix + "Document: "]
                        * bs,  # If model trained with task prefix before embedding
                        prompt_post_embed=(
                            context_prompt
                            if pipeline.pipeline_args.w_prefix_prompt
                            else no_context_prompt
                        )
                        if not icl_before_pref
                        else [
                            f"\nQuestion: {query}\nAnswer:"
                            for query in new_questions[i : i + bs]
                        ],
                        text_conditioning=text_conditioning,
                        temperature=temp,
                        max_tokens=max_seq_len,
                        truncate_line=True,
                        device=device,
                        device_generation=other_device,
                        give_n_tokens=True,
                        w_scores=None
                        if len(scores) == 0 or (with_scores == 0.0)
                        else list(new_scores[i : i + bs]),
                    )
                    if w_embeds:
                        compress_ratio += embeds / embed_tokens
                    else:
                        compress_ratio += 1
                    generated_sequences.extend(generated_sequence)
                else:
                    tokens = [
                        mistral_tokenizer.encode(prompt, bos=True, eos=False)
                        for prompt in no_context_prompt
                    ]

                    compress_ratio += sum([len(token) for token in tokens]) / sum(
                        [len(token) for token in tokens]
                    )
                    generated_sequence, logprobs = generate(
                        model=mistral_model,
                        encoded_prompts=tokens,
                        max_tokens=max_seq_len,
                        temperature=temp,
                        eos_id=mistral_tokenizer.eos_id,
                    )

                    generated_sequences.extend(
                        [
                            mistral_tokenizer.decode(gen).split("\n")[0]
                            for gen in generated_sequence
                        ]
                    )

            if METRIC_EVALUATION[benchmark] == get_em:
                value_em = sum(
                    [
                        metric_max_over_ground_truths(get_em, pred, gts)
                        for pred, gts in zip(generated_sequences, new_answers)
                    ]
                ) / len(generated_sequences)

                value_approx = sum(
                    [
                        metric_max_over_ground_truths(get_approx_em, pred, gts)
                        for pred, gts in zip(generated_sequences, new_answers)
                    ]
                ) / len(generated_sequences)

                if "EM" not in metrics[benchmark].keys():
                    metrics[benchmark]["EM"] = {}
                metrics[benchmark]["EM"][str(temp)] = {}

                if "F1" not in metrics[benchmark].keys():
                    metrics[benchmark]["F1"] = {}
                metrics[benchmark]["F1"][str(temp)] = {}

                n_answer_in_context = (
                    sum(
                        [
                            metric_max_over_ground_truths(get_approx_em, cont, gts)
                            for cont, gts in zip(
                                list(new_context), new_answers[:n_samples]
                            )
                        ]
                    )
                    / n_samples
                )

                value_xrag, _ = get_substring_match_score(
                    generated_sequences, new_answers[:n_samples]
                )

                metrics[benchmark]["EM"][str(temp)] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "w_context_in_examples": icl_w_context,
                    "w_context_w_query": query_w_context,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_passages": max_multi_passage,
                    "1 passage splitted ?": split_to_multipassage,
                    "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                    "w_scores": with_scores,
                }
                value_f1 = (
                    sum(
                        [
                            metric_max_over_ground_truths(get_f1_score, pred, gts)
                            for pred, gts in zip(generated_sequences, new_answers)
                        ]
                    )
                    / n_samples
                )

                metrics[benchmark]["F1"][str(temp)] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "w_context_in_examples": icl_w_context,
                    "w_context_w_query": query_w_context,
                    "Metric": value_f1,
                    "n_passages": max_multi_passage,
                    "1 passage splitted ?": split_to_multipassage,
                    "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                    "w_scores": with_scores,
                }
                eval_logger_info(logger, "Prompt prefix: " + prompt_prefix)
                eval_logger_info(
                    logger,
                    f"Context |  query | gen sequence | answer: {list(zip(new_context, new_questions, generated_sequences, new_answers))[-1]}",
                )

                eval_logger_info(
                    logger,
                    f"Temperature: {temp}, bench: {benchmark},  EM {value_em}, Approx EM {value_approx}, F1 {value_f1}",
                )
            else:
                value = (
                    sum(
                        [
                            metric_max_over_ground_truths(
                                METRIC_EVALUATION[benchmark], pred, gts
                            )
                            for pred, gts in zip(generated_sequences, new_answers)
                        ]
                    )
                    / n_samples
                )
                n_answer_in_context = (
                    sum(
                        [
                            metric_max_over_ground_truths(get_approx_em, cont, gts)
                            for cont, gts in zip(
                                list(new_context), new_answers[:n_samples]
                            )
                        ]
                    )
                    / n_samples
                )
                eval_logger_info(logger, f"Temperature: {temp} {benchmark}  {value}")

                metrics[benchmark][str(temp)] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value,
                    "w_context_in_examples": icl_w_context,
                    "n_passages": max_multi_passage,
                    "1 passage splitted ?": split_to_multipassage,
                    "prop context containing the answer": n_answer_in_context,
                    "w_scores": with_scores,
                }

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if run_name is not None:
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
                + run_name
                + "/results_generation.json",
                "a",
            ) as f:
                json.dump(results, f)
        elif instruct_name is not None:
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
                + instruct_name
                + "/results_generation.json",
                "a",
            ) as f:
                json.dump(results, f)

        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

        if mistral and query_w_context:
            run_name = "Mistral_RAG"
            ckpt = 0
        elif mistral and not query_w_context:
            run_name = "Mistral_no_RAG"
            ckpt = 0

        run_name = (
            instruct_name
            if instruct_name is not None and run_name is None
            else run_name
        )
        if run_name not in overall_results.keys():
            overall_results[run_name] = {}
        if str(ckpt) not in overall_results[run_name].keys():
            overall_results[run_name][str(ckpt)] = {}
        for benchmark in benchmarks:
            if benchmark not in overall_results[run_name][str(ckpt)].keys():
                overall_results[run_name][str(ckpt)][benchmark] = {}

        for benchmark in metrics.keys():
            for metric in metrics[benchmark].keys():
                if metric not in overall_results[run_name][str(ckpt)][benchmark].keys():
                    overall_results[run_name][str(ckpt)][benchmark][metric] = {}
                for temp in metrics[benchmark][metric].keys():
                    if (
                        temp
                        not in overall_results[run_name][str(ckpt)][benchmark][
                            metric
                        ].keys()
                    ):
                        overall_results[run_name][str(ckpt)][benchmark][metric][
                            temp
                        ] = []
                    overall_results[run_name][str(ckpt)][benchmark][metric][
                        temp
                    ].append(metrics[benchmark][metric][temp])

        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f)

    if mistral:
        return mistral_model

    return pipeline, ckpt


def evaluate_reconstruction_model(
    run_name: str,
    ckpt: int | None = None,
    pipeline: EmbedAugPipeline | None = None,
    output_file: str = None,
    temperatures: list[float] = [0, 0.5, 0.7, 1],
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    tmp_path: str = None,
    eval_data_type: str = "atlas",
    n_passages: int = 100,
    max_multi_passage: int = 1,
    instruct_name: str = None,
    prompt_before_embed: bool = False,
):
    reconstruct_benchmarks = [
        "Overlap",
        "Bleu",
        "Trunc Bleu",
        "AVG Bleu",
        "Meteor",
        "EM",
    ]

    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    if eval_data_type == "atlas":
        eval_data = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/wiki_passages_pretraining/valid_atlas_enwiki-dec2021_standard.jsonl"
    elif eval_data_type == "standard_dump":
        eval_data = "/lustre/scwpod02/client/kyutai-interns/datasets/modular_finetuning/enwiki-20220120_valid.jsonl"
    else:
        raise ValueError("Invalid eval_data_type")

    if max_multi_passage > 1:
        eval_data = "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/wiki_passages_pretraining/valid_atlas_enwiki-dec2021_50_30_20.jsonl"

    eval_logger_info(logger, "RUN NAME => " + run_name)

    # Loading model
    if not is_torchrun():
        device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"
        device_count = torch.cuda.device_count()
        other_device = torch.device("cuda:1") if device_count > 1 else device
    else:
        device = "cuda"
        other_device = None

    pipeline, ckpt = load_pipeline(
        run_name=run_name,
        tmp_path=tmp_path,
        llm_path=llm_path,
        device=device,
        max_bs=max_batch_size,
        pipeline=pipeline,
        mistral=False,
        instruct_name=instruct_name,
        ckpt=ckpt,
    )

    lim_toks = max_seq_len
    valid_passage = []

    if max_multi_passage == 1:
        with open(eval_data, "r") as f:
            for i, line in enumerate(f):
                if i == n_passages:
                    break

                if eval_data_type == "standard_dump":
                    valid_passage.append(
                        pipeline.tokenizer.decode(
                            pipeline.tokenizer.encode(
                                json.loads(line)["text"].split("\n\n")[1],
                                eos=True,
                                bos=True,
                            )[:lim_toks]
                        )
                    )
                elif eval_data_type == "atlas":
                    valid_passage.append(
                        pipeline.tokenizer.decode(
                            pipeline.tokenizer.encode(
                                json.loads(line)["text"], eos=True, bos=True
                            )[:lim_toks]
                        )
                    )
    else:
        with open(eval_data, "r") as f:
            for i, line in enumerate(f):
                if i == n_passages:
                    break
                valid_passage.append(json.loads(line)["passage"][:max_multi_passage])

    max_tokens = lim_toks

    results_generation = {}

    n_passages = len(valid_passage)
    assert n_passages == len(valid_passage)

    device_count = torch.cuda.device_count()

    if not is_torchrun():
        other_device = device if device_count <= 1 else torch.device("cuda:1")
    else:
        other_device = None

    for temp in temperatures:
        eval_logger_info(logger, f"Temperature: {temp}")
        generated_sequences = []
        for i in range(0, n_passages, max_batch_size):
            passage = list(valid_passage[i : i + max_batch_size])
            generated_sequence = pipeline.generate(
                prompt_pre_embed=(
                    [""] * len(passage)
                    if not prompt_before_embed
                    else ["In other words, background: "] * len(passage)
                ),
                prompt_post_embed=(
                    [""] * len(passage)
                    if not pipeline.pipeline_args.w_prefix_prompt
                    else [" is just another way of saying: "] * len(passage)
                ),
                text_conditioning=passage,
                temperature=temp,
                max_tokens=max_tokens,
                truncate_line=False,
                device=device if not is_torchrun() else "cuda",
                device_generation=other_device,
            )

            generated_sequences.extend(generated_sequence)
        results_generation[str(temp)] = generated_sequences

    metrics = {bench: {} for bench in reconstruct_benchmarks}

    for temp in results_generation.keys():
        # for split in results_generation[temp].keys():

        generated_sequences = results_generation[str(temp)]
        gt_passage = (
            valid_passage
            if isinstance(valid_passage[0], str)
            else [" ".join(p) for p in valid_passage]
        )

        overlap = word_overlap(gt_passage, generated_sequences)
        metrics["Overlap"][temp] = {
            "n_samples": n_passages,
            "Metric": overlap,
            "eval_data_type": eval_data_type,
        }
        bleu_score = get_bleu_score(gt_passage, generated_sequences)
        metrics["Bleu"][temp] = {
            "n_samples": n_passages,
            "Metric": bleu_score,
            "eval_data_type": eval_data_type,
        }
        trunc_bleu_score = get_bleu_score(gt_passage, generated_sequences, trunc=True)
        metrics["Trunc Bleu"][temp] = {
            "n_samples": n_passages,
            "Metric": trunc_bleu_score,
            "eval_data_type": eval_data_type,
        }
        bleu_score_avg = get_bleu_score(gt_passage, generated_sequences, avg=True)
        metrics["AVG Bleu"][temp] = {
            "n_samples": n_passages,
            "Metric": bleu_score_avg,
            "eval_data_type": eval_data_type,
        }
        meteor_score = get_meteor(gt_passage, generated_sequences)
        metrics["Meteor"][temp] = {
            "n_samples": n_passages,
            "Metric": meteor_score,
            "eval_data_type": eval_data_type,
        }
        em = np.mean(
            np.array(
                [get_em(gt, pred) for gt, pred in zip(gt_passage, generated_sequences)]
            )
        )
        metrics["EM"][str(temp)] = {
            "n_samples": n_passages,
            "Metric": em,
            "eval_data_type": eval_data_type,
        }

        eval_logger_info(
            logger,
            f"CKPT: {ckpt}, Temperature: {temp}, Overlap: {overlap}  \
            Bleu Score: {bleu_score} Truncated Bleu Score: {trunc_bleu_score} \
            EM: {em} Meteor: {meteor_score} Bleu Score Avg: {bleu_score_avg}",
        )

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        with open(
            "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
            + run_name
            + "/results_generation.json",
            "a",
        ) as f:
            json.dump(metrics, f)

        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

        if run_name not in overall_results.keys():
            overall_results[run_name] = {}
        if str(ckpt) not in list(overall_results[run_name].keys()):
            overall_results[run_name][str(ckpt)] = {}
        for benchmark in reconstruct_benchmarks:
            if benchmark not in list(overall_results[run_name][str(ckpt)].keys()):
                overall_results[run_name][str(ckpt)][benchmark] = {
                    str(temp): [] for temp in temperatures
                }

        for benchmark in metrics.keys():
            for temp in metrics[benchmark].keys():
                overall_results[run_name][str(ckpt)][benchmark][temp].append(
                    metrics[benchmark][temp]
                )

        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f)

    return pipeline, ckpt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--eval_reconstruction", action="store_true")
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--wo_embeds", action="store_false")
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--reconstruct_seq_len", type=int, default=256)
    parser.add_argument("--reconstruct_npassages", type=int, default=500)
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument("--colbert", action="store_true")
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--prompt_before_embed", action="store_true")
    parser.add_argument("--split_to_multipassage", action="store_true")
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--with_scores", type=float, default=0.0)
    parser.add_argument("--icl_exs", type=int, default=None)
    parser.add_argument("--llmemb_icl_w_context", action="store_true")
    parser.add_argument("--icl_before_pref", action="store_true")
    parser.add_argument("--compress_rate", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    temp_tests = [0]

    args = arg_parser()

    if args.benchmarks == "all":
        benchmarks = ["NQ", "TRIVIAQA", "HotpotQA", "SQUAD"]
    elif args.benchmarks == "two_main":
        benchmarks = ["NQ", "TRIVIAQA"]
    else:
        benchmarks = [args.benchmarks]
    icl_tests = [0, 2, 5] if args.icl_exs is None else [args.icl_exs]
    ensure_reproducibility(29)

    output_file = (
        "/home/hippolytepilchen/code/embed_llm/results/NVEmbed/mistral/eval_mistral_RAG_QA_final.json"
        if args.out_file is None
        else args.out_file
    )
    tmp_path = "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)

    max_seq_len = args.max_seq_len
    n_passages = args.n_passages

    if args.instruct_name is not None:
        print("Evaluating with instruction:", args.instruct_name)
    if args.run_name is not None:
        print("Evuating run:", args.run_name)
    # Evaluate Mistral using their code
    if args.mistral:
        assert not args.eval_reconstruction, (
            "Cannot evaluate reconstruction with Mistral"
        )
        if args.multi_passages == 1:
            print("EVALUATING WITHOUT CONTEXT")
            mistral_model = evaluate_QA(
                "",
                benchmarks,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                max_seq_len=max_seq_len,
                tmp_path=tmp_path,
                icl_examples=icl_tests[0],
                mistral=True,
                icl_w_context=False,
                query_w_context=False,
                w_embeds=False,
            )
            torch.cuda.empty_cache()
        eval_logger_info(logger, "EVALUATING WITH CONTEXT")
        mistral_model = evaluate_QA(
            "",
            benchmarks,
            temps=temp_tests,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_passages,
            max_seq_len=max_seq_len,
            tmp_path=tmp_path,
            icl_examples=icl_tests[0],
            mistral=True,
            icl_w_context=True,
            query_w_context=True,
            w_embeds=False,
            colbert=args.colbert,
            max_multi_passage=args.multi_passages,
            split_to_multipassage=args.split_to_multipassage,
            seed=args.seed,
        )
        torch.cuda.empty_cache()

        for icl_ex in icl_tests[1:]:
            if args.multi_passages == 1:
                print("EVALUATING WITHOUT CONTEXT")
                mistral_model = evaluate_QA(
                    "",
                    benchmarks,
                    temps=temp_tests,
                    max_bs=args.bs,
                    output_file=output_file,
                    n_samples=n_passages,
                    max_seq_len=max_seq_len,
                    tmp_path=tmp_path,
                    icl_examples=icl_ex,
                    mistral=True,
                    icl_w_context=False,
                    query_w_context=False,
                    w_embeds=False,
                    pipeline=mistral_model,
                )
                torch.cuda.empty_cache()
            eval_logger_info(logger, "EVALUATING WITH CONTEXT")
            mistral_model = evaluate_QA(
                "",
                benchmarks,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                max_seq_len=max_seq_len,
                tmp_path=tmp_path,
                icl_examples=icl_ex,
                mistral=True,
                icl_w_context=True,
                query_w_context=True,
                w_embeds=False,
                pipeline=mistral_model,
                colbert=args.colbert,
                max_multi_passage=args.multi_passages,
                split_to_multipassage=args.split_to_multipassage,
                seed=args.seed,
            )
            torch.cuda.empty_cache()

    else:
        if args.eval_reconstruction:
            if args.multi_passages > 1:
                pipeline, ckpt = evaluate_reconstruction_model(
                    args.run_name,
                    ckpt=args.ckpt,
                    output_file=output_file,
                    temperatures=temp_tests,
                    max_seq_len=args.reconstruct_seq_len,
                    tmp_path=tmp_path,
                    n_passages=args.reconstruct_npassages,
                    max_multi_passage=args.multi_passages,
                    instruct_name=args.instruct_name,
                    prompt_before_embed=args.prompt_before_embed,
                )
                torch.cuda.empty_cache()
            else:
                eval_logger_info(logger, "Standard Dump")
                pipeline, ckpt = evaluate_reconstruction_model(
                    args.run_name,
                    output_file=output_file,
                    ckpt=args.ckpt,
                    temperatures=temp_tests,
                    max_seq_len=args.reconstruct_seq_len,
                    tmp_path=tmp_path,
                    eval_data_type="standard_dump",
                    n_passages=args.reconstruct_npassages,
                    instruct_name=args.instruct_name,
                    prompt_before_embed=args.prompt_before_embed,
                )
                torch.cuda.empty_cache()
                eval_logger_info(logger, "Atlas")
                pipeline, ckpt = evaluate_reconstruction_model(
                    args.run_name,
                    output_file=output_file,
                    temperatures=temp_tests,
                    max_seq_len=args.reconstruct_seq_len,
                    tmp_path=tmp_path,
                    eval_data_type="atlas",
                    pipeline=pipeline,
                    ckpt=ckpt,
                    n_passages=args.reconstruct_npassages,
                    instruct_name=args.instruct_name,
                    prompt_before_embed=args.prompt_before_embed,
                )
                torch.cuda.empty_cache()

        pipeline, ckpt = evaluate_QA(
            args.run_name,
            benchmarks,
            ckpt=args.ckpt,
            temps=temp_tests,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_passages,
            max_seq_len=max_seq_len,
            tmp_path=tmp_path,
            icl_examples=icl_tests[0],
            w_embeds=args.wo_embeds,
            icl_w_context=args.llmemb_icl_w_context,
            max_multi_passage=args.multi_passages,
            instruct_name=args.instruct_name,
            colbert=args.colbert,
            prompt_before_embed=args.prompt_before_embed,
            split_to_multipassage=args.split_to_multipassage,
            seed=args.seed,
            with_scores=args.with_scores,
            icl_before_pref=args.icl_before_pref,
            compress_rate=args.compress_rate,
        )

        for icl_ex in icl_tests[1:]:
            pipeline, ckpt = evaluate_QA(
                args.run_name,
                benchmarks,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                max_seq_len=max_seq_len,
                tmp_path=tmp_path,
                icl_examples=icl_ex,
                w_embeds=args.wo_embeds,
                icl_w_context=args.llmemb_icl_w_context,
                pipeline=pipeline,
                ckpt=ckpt,
                max_multi_passage=args.multi_passages,
                instruct_name=args.instruct_name,
                colbert=args.colbert,
                prompt_before_embed=args.prompt_before_embed,
                split_to_multipassage=args.split_to_multipassage,
                seed=args.seed,
                with_scores=args.with_scores,
                icl_before_pref=args.icl_before_pref,
                compress_rate=args.compress_rate,
            )
