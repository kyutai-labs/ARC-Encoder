import argparse
import json
import logging
import os
import random

import torch
from tqdm import tqdm, trange


from embed_llm.generation.metrics import (
    get_approx_em,
    get_bleu_score,
    get_rouge_score,
    get_em,
    get_f1_score,
    get_substring_match_score,
    metric_max_over_ground_truths,
)
from embed_llm.generation.utils import (
    ensure_reproducibility,
    eval_logger_info,
    create_prompt_prefix,
    create_prompt,
    EVAL_DATA_PATH,
    METRIC_EVALUATION,
    TRAD_DATA_PATH,
)
from embed_llm.models.augmented_model import EmbedAugPipeline, load_pipeline
from embed_llm.models.utils.utils import is_torchrun
from embed_llm.monitoring.utils import set_logger
from embed_llm import TMP_PATH, MODEL_PATH


logger = logging.getLogger(__name__)


def evaluate_QA(
    run_name: str,
    benchmarks: list[str],
    llm_path: str | None = None,
    embed_path: str | None = None,
    ckpt: int | None = None,
    max_seq_len: int = 256,
    temps: list[float] = [0, 0.5, 0.7, 1],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    icl_examples: int = 0,
    pipeline: EmbedAugPipeline | None = None,
    max_multi_passage: int = 1,  # If >1, use multiple passages from retrieval
    seed: float = 0.42,
    comp_rate: int | None = None,
    llm_number: int = 0,
    cat_multi_passages: bool = False,  # If True, concatenate passages from retrieval
    max_doc_len: int | None = None,  # Maximum length of documents
    chunk_to: int
    | None = None,  # If not None, chunk documents to this size before embedding
):
    """Load the pipeline and evaluate it on the QA benchmarks"""
    if llm_path is not None:
        llm_name = llm_path.split("/")[-1]
    else:
        llm_name = None

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
        embedder_path=embed_path,
        device=device,
        max_bs=max_bs,
        pipeline=pipeline,
        ckpt=ckpt,
        comp_rate=comp_rate,
        llm_type="llama"
        if llm_path is not None and "llama" in llm_path.lower()
        else (
            "olmo" if llm_path is not None and "olmo" in llm_path.lower() else "mistral"
        ),
        embed_type="llama"
        if embed_path is not None and "llama" in embed_path.lower()
        else "mistral",
        llm_number=llm_number,
    )

    # Creating dataset
    metrics = {}

    total_benchmarks = len(benchmarks)
    max_sample = n_samples is None
    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=total_benchmarks
    ):
        if benchmark == "SQUAD" and max_multi_passage > 1:
            benchmarks.remove(benchmark)
            continue

        if benchmark == "CNN" and max_multi_passage > 1:
            benchmarks.remove(benchmark)
            continue

        metrics[benchmark] = {}
        eval_data = EVAL_DATA_PATH[benchmark]

        context = []
        questions = []
        answers = []
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
                    if max_doc_len is not None:
                        context.append(
                            [data["passages"][0].strip()[: max_doc_len * 3]]
                        )  # Approximate the number of tokens
                    else:
                        context.append([data["passages"][0].strip()])
                else:
                    context.append(list((data["passages"][:max_multi_passage])))
        c = list(zip(questions, context, answers))

        random.shuffle(c, random=lambda: seed)
        questions, context, answers = zip(*c)

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")

        prompt_str, to_embed_str = create_prompt_prefix(
            queries=questions,
            answers=[answer[0] for answer in answers],
            docs=context,
            max_examples=icl_examples,
            cat_multi_passages=cat_multi_passages,
        )

        new_context, new_questions, new_answers = (
            list(context[icl_examples:]),
            list(questions[icl_examples:]),
            list(answers[icl_examples:]),
        )

        new_context.reverse()
        new_questions.reverse()
        new_answers.reverse()

        for temp in temps:
            compress_ratio = 0
            generated_sequences = []
            n_samples = (
                len(new_questions)
                if (n_samples is None or max_sample)
                else min(len(new_questions), n_samples)
            )
            if benchmark == "CNN":
                n_samples = min(n_samples, 1000)
                max_seq_len = 256

            for i in trange(0, n_samples, max_bs):
                texts_to_embed = []
                batch_list_prompts = []
                bs = min(max_bs, n_samples - i)

                for query, doc in zip(
                    new_questions[i : i + bs], new_context[i : i + bs]
                ):
                    batch_list_prompt, text_to_embed = create_prompt(
                        prefix_prompt=prompt_str,
                        prefix_embed=to_embed_str,
                        doc=doc,
                        query=query,
                        wdoc=False,  # No text document, only compressed ones
                        cat_multi_passages=cat_multi_passages,
                    )

                    batch_list_prompts.append(batch_list_prompt)
                    texts_to_embed.append(text_to_embed)

                generated_sequence, sum_comp_ratio = pipeline.generate(
                    text_to_embed=texts_to_embed,
                    batch_list_prompts=batch_list_prompts,
                    temperature=temp,
                    max_tokens=max_seq_len,
                    truncate_line=True,
                    device=device,
                    device_generation=other_device,
                    give_n_tokens=True,
                    chunk_to=chunk_to,  # If not None, chunk the dataset to this size
                )
                compress_ratio += sum_comp_ratio  # N tokens to be compressed / final number of tokens after compression
                generated_sequences.extend(generated_sequence)

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

                metrics[benchmark]["EM"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio / n_samples,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "max_doc_len": max_doc_len,
                    "chunk_to": chunk_to if chunk_to is not None else 0,
                    "ckpt": ckpt if ckpt is not None else "None",
                    "temp": temp,
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

                metrics[benchmark]["F1"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_f1,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio / n_samples,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "max_doc_len": max_doc_len,
                    "chunk_to": chunk_to if chunk_to is not None else 0,
                    "ckpt": ckpt if ckpt is not None else "None",
                    "temp": temp,
                }

                eval_logger_info(
                    logger,
                    f"COMP RATE: {compress_ratio / n_samples} => Context |  query | gen sequence | answer: {list(zip(new_context, new_questions, generated_sequences, new_answers))[-1]}",
                )

                eval_logger_info(
                    logger,
                    f"Temperature: {temp}, bench: {benchmark},  EM {value_em}, Approx EM {value_approx}, F1 {value_f1}",
                )
            elif METRIC_EVALUATION[benchmark] == get_rouge_score:
                value_rouge = sum(
                    [
                        metric_max_over_ground_truths(get_rouge_score, pred, gts)
                        for pred, gts in zip(generated_sequences, new_answers)
                    ]
                ) / len(generated_sequences)

                if "ROUGE" not in metrics[benchmark].keys():
                    metrics[benchmark]["ROUGE"] = {}
                metrics[benchmark]["ROUGE"] = {}

                metrics[benchmark]["ROUGE"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_rouge,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "max_doc_len": max_doc_len,
                    "chunk_to": chunk_to if chunk_to is not None else 0,
                    "ckpt": ckpt if ckpt is not None else "None",
                    "temp": temp,
                }
                eval_logger_info(
                    logger,
                    f"COMP RATE: {compress_ratio / n_samples} => Context |  query | gen sequence | answer: {list(zip(new_context, new_questions, generated_sequences, new_answers))[-1]}",
                )

                eval_logger_info(
                    logger,
                    f"Temperature: {temp}, bench: {benchmark},  EM {value_rouge}",
                )
            else:
                raise NotImplementedError(
                    f"Metric {METRIC_EVALUATION[benchmark]} is not implemented for benchmark {benchmark}"
                )

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

        if run_name not in overall_results.keys():
            overall_results[run_name] = {}

        for benchmark in benchmarks:
            if benchmark not in overall_results[run_name].keys():
                overall_results[run_name][benchmark] = {}

        for benchmark in metrics.keys():
            for metric in metrics[benchmark].keys():
                if metric not in overall_results[run_name][benchmark].keys():
                    overall_results[run_name][benchmark][metric] = []

                overall_results[run_name][benchmark][metric].append(
                    metrics[benchmark][metric]
                )

        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f, indent=4)

    return pipeline, ckpt


def evaluate_trad(
    run_name: str,
    llm_path: str | None = None,
    embed_path: str | None = None,
    ckpt: int | None = None,
    max_seq_len: int = 128,
    temps: list[float] = [0, 0.5, 0.7, 1],
    benchmarks: list[str] = ["Danish", "French", "Spanish", "German"],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    pipeline: EmbedAugPipeline | None = None,
    seed: float = 0.42,
    comp_rate: int | None = None,
    llm_number: int = 0,
    chunk_to: int | None = None,
):
    # Loading model
    llm_name = llm_path.split("/")[-1]

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
        embedder_path=embed_path,
        llm_path=llm_path,
        device=device,
        max_bs=max_bs,
        pipeline=pipeline,
        ckpt=ckpt,
        comp_rate=comp_rate,
        llm_type="llama"
        if llm_path is not None and "llama" in llm_path.lower()
        else (
            "olmo" if llm_path is not None and "olmo" in llm_path.lower() else "mistral"
        ),
        embed_type="llama" if "llama" in embed_path.lower() else "mistral",
        llm_number=llm_number,
    )

    # Creating dataset
    metrics = {}
    max_samples = n_samples is None
    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=len(benchmarks)
    ):
        metrics[benchmark] = {}

        eval_data = TRAD_DATA_PATH[benchmark]

        text = []
        traduction = []

        with open(TRAD_DATA_PATH["English"], "r") as f:
            for line in f:
                data = json.loads(line)
                text.append(data["text"].strip())

        with open(eval_data, "r") as f:
            for line in f:
                data = json.loads(line)
                traduction.append(data["text"].strip())

        c = list(zip(text, traduction))

        random.shuffle(c, random=lambda: seed)
        text, traduction = zip(*c)
        embed_prompt = []

        text_prompt_prefix = ["Document: "]
        for doc, answ, _ in zip(text, traduction, range(4)):
            embed_prompt.append(doc.strip())
            text_prompt_prefix.append(
                f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer: {answ.strip()}\n\nDocument: "
            )
        embed_prompt.append(text[4].strip())
        text_prompt_prefix.append(
            f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer: {traduction[4].strip()}\n\nDocument: "
        )

        new_text, new_trad = (
            list(text[5:]),
            list(traduction[5:]),
        )

        text, traduction = new_text, new_trad

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")
        metrics[benchmark]["BLEU"] = {}
        for temp in temps:
            compress_ratio = 0
            generated_sequences = []
            n_samples = (
                len(text)
                if (n_samples is None or max_samples)
                else min(len(text), n_samples)
            )

            for i in trange(0, n_samples, max_bs):
                texts_to_embed = []
                batch_list_prompts = []
                bs = min(max_bs, n_samples - i)

                batch_list_prompts = [
                    text_prompt_prefix
                    + [
                        f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer:"
                    ]
                    for _ in range(bs)
                ]

                texts_to_embed = [embed_prompt + [seq] for seq in text[i : i + bs]]

                generated_sequence, sum_comp_ratio = pipeline.generate(
                    text_to_embed=texts_to_embed,
                    batch_list_prompts=batch_list_prompts,
                    temperature=temp,
                    max_tokens=max_seq_len,
                    truncate_line=True,
                    device=device,
                    device_generation=other_device,
                    give_n_tokens=True,
                    chunk_to=chunk_to,  # If not None, chunk the dataset to this size
                )

                compress_ratio += sum_comp_ratio  # N tokens to be compressed / final number of tokens after compression

                final_seq = []
                for seq in generated_sequence:
                    if len(seq.split("\n\n")[0]) > 1:
                        final_seq.append(seq.split("\n\n")[0].strip())
                    else:
                        final_seq.append(seq.strip())
                generated_sequences.extend(final_seq)

            bleu_score = get_bleu_score(
                traduction[: len(generated_sequences)], generated_sequences
            )
            print("BLEU score:", bleu_score)
            metrics[benchmark]["BLEU"] = {
                "n_samples": n_samples,
                "Metric": bleu_score,
                "compress_ratio": compress_ratio / n_samples,
                "language": benchmark,
                "new_template": True,
                "llm_name": llm_name,
                "chunk_to": chunk_to if chunk_to is not None else 0,
                "ckpt": ckpt if ckpt is not None else "None",
                "temp": temp,
            }

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

        if run_name not in overall_results.keys():
            overall_results[run_name] = {}

        for benchmark in metrics.keys():
            for metric in metrics[benchmark].keys():
                if metric not in overall_results[run_name].keys():
                    overall_results[run_name][metric] = []
                overall_results[run_name][metric].append(metrics[benchmark][metric])

        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f, indent=4)

    return pipeline, ckpt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--cat_multi_passages", action="store_true")
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--n_icl_exs", type=int, default=None)
    parser.add_argument(
        "--comp_rate", type=int, default=None
    )  # can enable to fix number of memory tokens if > 0
    parser.add_argument("--eval_trad", action="store_true")
    parser.add_argument("--llm_name", type=str, default="mistral_7B")
    parser.add_argument("--embed_name", type=str, default="mistral_7B")
    parser.add_argument("--max_doc_len", type=int, default=None)
    parser.add_argument(
        "--llm_number",
        type=int,
        default=0,
        help="Number of LLMs to use, starting from 0",
    )
    parser.add_argument(
        "--chunk_to",
        type=int,
        default=None,
        help="If not None, chunk each sample into samples of size chunk_to and compress them in parallel",
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    temp_tests = [0]

    args = arg_parser()

    if args.benchmarks == "all":
        benchmarks = ["NQ", "TRIVIAQA", "SQUAD"]
    else:
        benchmarks = [args.benchmarks]
    icl_tests = [0, 5] if args.n_icl_exs is None else [args.n_icl_exs]
    ensure_reproducibility(29)

    output_file = args.out_file

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)

    max_seq_len = args.max_seq_len
    n_samples = args.n_samples
    llm_path = MODEL_PATH + args.llm_name
    embed_path = MODEL_PATH + args.embed_name

    if args.run_name is not None:
        print("Evuating run:", args.run_name)

    if args.eval_trad:
        print("EVALUATING TRANSLATION")
        pipeline, ckpt = evaluate_trad(
            args.run_name,
            max_seq_len=max_seq_len,
            temps=temp_tests,
            llm_path=llm_path,
            embed_path=embed_path,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_samples,
            tmp_path=TMP_PATH,
            pipeline=None,
            seed=args.seed,
            comp_rate=args.comp_rate,
            benchmarks=benchmarks
            if args.benchmarks != "all"
            else ["Danish", "French", "Spanish", "German"],
            llm_number=args.llm_number,
            chunk_to=args.chunk_to,
        )
        torch.cuda.empty_cache()
    else:
        pipeline, ckpt = evaluate_QA(
            args.run_name,
            benchmarks,
            ckpt=args.ckpt,
            temps=temp_tests,
            llm_path=llm_path,
            embed_path=embed_path,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            tmp_path=TMP_PATH,
            icl_examples=icl_tests[0],
            max_multi_passage=args.multi_passages,
            seed=args.seed,
            comp_rate=args.comp_rate,
            cat_multi_passages=args.cat_multi_passages,
            max_doc_len=args.max_doc_len,
            llm_number=args.llm_number,
            chunk_to=args.chunk_to,
        )

        for icl_ex in icl_tests[1:]:
            pipeline, ckpt = evaluate_QA(
                args.run_name,
                benchmarks,
                temps=temp_tests,
                llm_path=llm_path,
                embed_path=embed_path,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                tmp_path=TMP_PATH,
                icl_examples=icl_ex,
                pipeline=pipeline,
                ckpt=ckpt,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
                comp_rate=args.comp_rate,
                cat_multi_passages=args.cat_multi_passages,
                max_doc_len=args.max_doc_len,
                llm_number=args.llm_number,
                chunk_to=args.chunk_to,
            )
