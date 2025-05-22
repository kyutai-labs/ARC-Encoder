import os
import torch
import json
import random
from tqdm import tqdm, trange
import argparse
import logging
import subprocess as sp
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from embed_llm.models.augmented_model import EmbedAugPipeline, load_pipeline
from embed_llm.models.utils import is_torchrun
from embed_llm.monitoring.utils import set_logger
from embed_llm.generation.utils import eval_logger_info, ensure_reproducibility
from embed_llm.generation.metrics import (
    get_em,
    get_f1_score,
    metric_max_over_ground_truths,
    get_approx_em,
    get_substring_match_score,
    get_bleu_score,
)


EVAL_DATA_PATH = {
    "NQ": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/nq_open_data.jsonl",  # nq_data.jsonl
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/triviaqa_data.jsonl",
    "HotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/Hotpot_qa_test.jsonl",
    "SQUAD": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/squad_test.jsonl",
}

METRIC_EVALUATION = {
    "NQ": get_em,
    "TRIVIAQA": get_em,
    "HotpotQA": get_em,
    "SQUAD": get_em,
}


TRAD_DATA_PATH = {
    "English": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/eng_Latn.jsonl",
    "Spanish": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/spa_Latn.jsonl",
    "French": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/fra_Latn.jsonl",
    "German": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/deu_Latn.jsonl",
    "Danish": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/dan_Latn.jsonl",
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
    compressed_doc_in_icl: bool = False,
    reversed_template: bool = False,
) -> tuple[list[str], list[str] | None]:
    max_examples = max_examples if max_examples is not None else len(queries)
    prompt_str = []
    to_embed_str = []

    prompt = ""

    if docs is not None:
        if compressed_doc_in_icl:
            if not reversed_template:
                for query, answer, doc, index in zip(
                    queries, answers, docs, range(max_examples)
                ):
                    if index == 0:
                        prompt_str.append("Document: ")
                        to_embed_str.append(doc.strip())
                    elif index == max_examples - 1:
                        prompt_str.append(f"\nQuestion: {query}\nAnswer: {answer}\n\n")
                    else:
                        prompt_str.append(
                            f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                        )
                        to_embed_str.append(doc.strip())
            else:
                prompt_str.append(f"\nQuestion: {query}\nDocument: ")
                to_embed_str.append(doc.strip())
                prompt_str.append(f"\nAnswer: {answer}\n\n")

            if max_examples == 0:
                prompt_str.append("")
        else:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
                if reversed_template:
                    prompt += (
                        f"Question: {query}\nDocument: {doc}\nAnswer: {answer}\n\n"
                    )
                else:
                    prompt += (
                        f"Document: {doc}\nQuestion: {query}\nAnswer: {answer}\n\n"
                    )

            to_embed_str = None
            prompt_str.append(prompt)

    else:
        for query, answer, _ in zip(queries, answers, range(max_examples)):
            prompt += f"Question: {query}\nAnswer: {answer}\n\n"

        prompt_str.append(prompt)
        to_embed_str = None

    return prompt_str, to_embed_str


def create_prompt(
    prefix_prompt: list[str],
    prefix_embed: list[str] | None,
    doc: str,
    query: str,
    wdoc: bool = True,
    w_embeds: bool = True,
    reversed_template: bool = False,
) -> tuple[list[str], list[str] | None]:
    list_prompt = prefix_prompt.copy()

    if prefix_embed is None and w_embeds:
        list_embed = []
    elif not w_embeds:
        list_embed = None
    else:
        list_embed = prefix_embed.copy()

    assert int(wdoc) * int(w_embeds) == 0, (
        "Cannot use both text context and embeddings as the document in the same time"
    )

    if wdoc:
        return [
            "".join(list_prompt.append(f"Document: {doc}\nQuestion: {query}\nAnswer:"))
        ], list_embed
    else:
        if w_embeds:
            if not reversed_template:
                last_prompt = list_prompt[-1]
                list_prompt[-1] = "".join([last_prompt, "Document: "])
                list_embed.append(doc.strip())
                list_prompt.append(f"\nQuestion: {query}\nAnswer:")
            else:
                last_prompt = list_prompt[-1]
                list_prompt[-1] = "".join(
                    [last_prompt, f"\nQuestion: {query}\nDocument: "]
                )
                list_embed.append(doc.strip())
                list_prompt.append("\nAnswer:")
        else:
            list_embed = None
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")

        return list_prompt, list_embed


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
    query_w_context: bool = False,
    icl_w_document: bool = True,
    mistral: bool = False,
    max_multi_passage: int = 1,
    seed: float = 0.42,
    compressed_doc_in_icl: bool = False,
    reversed_template: bool = False,
    comp_rate: int | None = None,
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
        ckpt=ckpt,
        comp_rate=comp_rate,
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

        if compressed_doc_in_icl and icl_examples == 0:
            continue

        if benchmark == "HotpotQA":
            max_multi_passage = 2

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
                    context.append(data["passages"][0].strip())

                else:
                    context.append(
                        "\n".join(list((data["passages"][:max_multi_passage])))
                    )

        c = list(zip(questions, context, answers))

        # fixed_random = random.Random()
        # fixed_random.seed(42)
        # fixed_random.shuffle(c)
        random.shuffle(c, random=lambda: seed)
        questions, context, answers = zip(*c)

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")

        if compressed_doc_in_icl:
            assert w_embeds, (
                "Compressed document in ICL is not compatible without embeddings"
            )
        if query_w_context:
            assert not w_embeds, "Query with context is not compatible with embeddings"

        prompt_str, to_embed_str = create_prompt_prefix(
            queries=questions,
            answers=[answer[0] for answer in answers],
            docs=None if not icl_w_document else context,
            max_examples=icl_examples,
            compressed_doc_in_icl=compressed_doc_in_icl,
            reversed_template=reversed_template,
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
            n_samples = len(new_questions) if n_samples is None else n_samples
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
                        w_embeds=w_embeds,
                        query=query,
                        wdoc=query_w_context,
                        reversed_template=reversed_template,
                    )
                    batch_list_prompts.append(batch_list_prompt)
                    texts_to_embed.append(text_to_embed)

                if not mistral:
                    generated_sequence, embed_tokens, embeds = pipeline.generate(
                        text_to_embed=texts_to_embed,
                        batch_list_prompts=batch_list_prompts,
                        temperature=temp,
                        max_tokens=max_seq_len,
                        truncate_line=True,
                        device=device,
                        device_generation=other_device,
                        give_n_tokens=True,
                    )
                    if w_embeds:
                        compress_ratio += (
                            embed_tokens / embeds
                        )  # N tokens to be compressed / final number of tokens after compression
                    else:
                        compress_ratio += 1
                    generated_sequences.extend(generated_sequence)
                else:
                    tokens = [
                        mistral_tokenizer.encode(prompt[0], bos=True, eos=False)
                        for prompt in texts_to_embed
                    ]

                    compress_ratio += 1
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
                    "w_context_in_examples": icl_w_document,
                    "w_context_w_query": query_w_context,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                    "compressed_icl": compressed_doc_in_icl,
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
                    "w_context_in_examples": icl_w_document,
                    "w_context_w_query": query_w_context,
                    "Metric": value_f1,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                    "compressed_icl": compressed_doc_in_icl,
                }

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
                    "w_context_in_examples": icl_w_document,
                    "n_passages": max_multi_passage,
                    "prop context containing the answer": n_answer_in_context,
                    "compressed_icl": compressed_doc_in_icl,
                }

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if run_name is not None:
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/hp_v2/"
                + run_name
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
            json.dump(overall_results, f, indent=4)

    if mistral:
        return mistral_model

    return pipeline, ckpt


def evaluate_trad(
    run_name: str,
    ckpt: int | None = None,
    max_seq_len: int = 512,
    temps: list[float] = [0, 0.5, 0.7, 1],
    benchmarks: list[str] = ["Danish", "French", "Spanish", "German"],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    pipeline: EmbedAugPipeline | Transformer | None = None,
    w_embeds: bool = True,  # To test baseline LLM
    mistral: bool = False,
    seed: float = 0.42,
    comp_rate: int | None = None,
    fine_tuned: bool = False,
):
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
        ckpt=ckpt,
        comp_rate=comp_rate,
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
        metrics[benchmark] = {}
        eval_data = TRAD_DATA_PATH[benchmark]

        text = []
        traduction = []

        with open(TRAD_DATA_PATH["English"], "r") as f:
            for line in f:
                data = json.loads(line)
                text.append(data["text"].strip())
                # Take the first ranked retrieved passage
        with open(eval_data, "r") as f:
            for line in f:
                data = json.loads(line)
                traduction.append(data["text"].strip())
                # Take the first ranked retrieved passage
        c = list(zip(text, traduction))

        # fixed_random = random.Random()
        # fixed_random.seed(42)
        # fixed_random.shuffle(c)
        random.shuffle(c, random=lambda: seed)
        text, traduction = zip(*c)

        prompt_prefix = "\n\n".join(
            [
                f"Document: {doc}\nTranslation: {answ}"
                for doc, answ, _ in zip(text, traduction, range(5))
            ]
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
            n_samples = len(text) if n_samples is None else n_samples
            for i in trange(0, n_samples, max_bs):
                texts_to_embed = []
                batch_list_prompts = []
                bs = min(max_bs, n_samples - i)
                if fine_tuned:
                    if benchmark == "Spanish":
                        batch_list_prompts = [
                            [
                                "Document: ",
                                "\nTranslate the previous document into Spanish.",
                            ]
                            for _ in range(bs)
                        ]
                    elif benchmark == "French":
                        batch_list_prompts = [
                            [
                                "Document: ",
                                "\nProvide a French translation of the text below.",
                            ]
                            for _ in range(bs)
                        ]
                    elif benchmark == "German":
                        batch_list_prompts = [
                            [
                                "Document: ",
                                "\nRender the document into fluent German while preserving its meaning.",
                            ]
                            for _ in range(bs)
                        ]
                    elif benchmark == "Danish":
                        batch_list_prompts = [
                            [
                                "Document: ",
                                "\nTranslate the previous document into Danish.",
                            ]
                            for _ in range(bs)
                        ]
                    else:
                        raise ValueError("Invalid benchmark")
                else:
                    batch_list_prompts = [
                        [
                            prompt_prefix + "\n\nDocument: ",
                            "\nTranslation:",
                        ]
                        for _ in range(bs)
                    ]

                texts_to_embed = [[seq] for seq in text[i : i + bs]]

                if not mistral:
                    generated_sequence, embed_tokens, embeds = pipeline.generate(
                        text_to_embed=texts_to_embed,
                        batch_list_prompts=batch_list_prompts,
                        temperature=temp,
                        max_tokens=max_seq_len,
                        truncate_line=False,
                        device=device,
                        device_generation=other_device,
                        give_n_tokens=True,
                    )
                    if w_embeds:
                        compress_ratio += (
                            embed_tokens / embeds
                        )  # N tokens to be compressed / final number of tokens after compression
                    else:
                        compress_ratio += 1

                    final_seq = []
                    for seq in generated_sequence:
                        if len(seq.split("\n\n")[0]) > 1:
                            final_seq.append(seq.split("\n\n")[0].strip())
                        elif "\nTranslation:" in seq.split("\n\n")[-1]:
                            final_seq.append(seq.split("\nTranslation:")[1].strip())
                        else:
                            final_seq.append(seq.strip())
                    generated_sequences.extend(final_seq)
                else:
                    if fine_tuned:
                        if benchmark == "Spanish":
                            prompts = [
                                "Document: "
                                + seq
                                + "\nTranslate the previous document into Spanish."
                                for seq in text[i : i + bs]
                            ]
                        elif benchmark == "French":
                            prompts = [
                                "Document: "
                                + seq
                                + "\nProvide a French translation of the text below."
                                for seq in text[i : i + bs]
                            ]

                        elif benchmark == "German":
                            prompts = [
                                "Document: "
                                + seq
                                + "\nRender the document into fluent German while preserving its meaning."
                                for seq in text[i : i + bs]
                            ]
                        else:
                            raise ValueError("Invalid benchmark")
                    else:
                        prompts = [
                            prompt_prefix + "\n\nDocument: " + seq + "\nTranslation:"
                            for seq in text[i : i + bs]
                        ]

                    prompt_tokens = [
                        mistral_tokenizer.encode(prompt, bos=True, eos=False)
                        for prompt in prompts
                    ]

                    compress_ratio += 1
                    generated_sequence, logprobs = generate(
                        model=mistral_model,
                        encoded_prompts=prompt_tokens,
                        max_tokens=max_seq_len,
                        temperature=temp,
                        eos_id=mistral_tokenizer.eos_id,
                    )

                    final_seq = []
                    for gen in generated_sequence:
                        seq = mistral_tokenizer.decode(gen).strip()
                        if len(seq.split("\n\n")[0]) > 1:
                            final_seq.append(seq.split("\n\n")[0].strip())
                        elif "\nTranslation:" in seq.split("\n\n")[-1]:
                            final_seq.append(seq.split("\nTranslation:")[1].strip())
                        else:
                            final_seq.append(seq.strip())
                    generated_sequences.extend(final_seq)

            bleu_score = get_bleu_score(
                traduction[: len(generated_sequences)], generated_sequences
            )
            print("BLEU score:", bleu_score)
            metrics[benchmark]["BLEU"][str(temp)] = {
                "n_samples": n_samples,
                "Metric": bleu_score,
                "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                "language": benchmark,
                "fine_tuned": fine_tuned,
            }

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if run_name is not None:
            with open(
                "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/hp_v2/"
                + run_name
                + "/results_generation.json",
                "a",
            ) as f:
                json.dump(results, f)

        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

        if mistral:
            run_name = "Mistral_RAG"
            ckpt = 0

        if run_name not in overall_results.keys():
            overall_results[run_name] = {}
        if str(ckpt) not in overall_results[run_name].keys():
            overall_results[run_name][str(ckpt)] = {}

        for benchmark in metrics.keys():
            for metric in metrics[benchmark].keys():
                if metric not in overall_results[run_name][str(ckpt)].keys():
                    overall_results[run_name][str(ckpt)][metric] = {}
                for temp in metrics[benchmark][metric].keys():
                    if temp not in overall_results[run_name][str(ckpt)][metric].keys():
                        overall_results[run_name][str(ckpt)][metric][temp] = []
                    overall_results[run_name][str(ckpt)][metric][temp].append(
                        metrics[benchmark][metric][temp]
                    )

        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f, indent=4)

    if mistral:
        return mistral_model

    return pipeline, ckpt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
    )
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mistral", action="store_true")
    parser.add_argument("--wo_embeds", action="store_false")
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--n_icl_exs", type=int, default=None)
    parser.add_argument("--icl_w_document", action="store_true")
    parser.add_argument("--compressed_doc_in_icl", action="store_true")
    parser.add_argument(
        "--comp_rate", type=int, default=None
    )  # can enable to fix number of memory tokens if > 0
    parser.add_argument("--eval_trad", action="store_true")
    parser.add_argument(
        "--tmp_path",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/tmp/hp_v2/",
    )
    parser.add_argument("--reversed_template", action="store_true")
    parser.add_argument("--fine_tuned", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    temp_tests = [0]

    args = arg_parser()

    if args.benchmarks == "all":
        benchmarks = ["NQ", "TRIVIAQA", "SQUAD", "HotpotQA"]
    else:
        benchmarks = [args.benchmarks]
    icl_tests = [0, 2, 5] if args.n_icl_exs is None else [args.n_icl_exs]
    ensure_reproducibility(29)

    output_file = (
        "/home/hippolytepilchen/code/hp_v2/results/NVEmbed/mistral/eval_mistral_translate.json"
        if args.out_file is None
        else args.out_file
    )
    tmp_path = args.tmp_path

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)

    max_seq_len = args.max_seq_len
    n_passages = args.n_passages

    if args.run_name is not None:
        print("Evuating run:", args.run_name)
    # Evaluate Mistral using their code
    if args.mistral:
        if args.eval_trad:
            print("EVALUATING TRANSLATION")
            mistral_model = evaluate_trad(
                args.run_name,
                max_seq_len=max_seq_len,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                tmp_path=tmp_path,
                pipeline=None,
                mistral=True,
                seed=args.seed,
                comp_rate=args.comp_rate,
                fine_tuned=args.fine_tuned,
                benchmarks=benchmarks
                if args.benchmarks != "all"
                else ["Danish", "French", "Spanish", "German"],
            )
            torch.cuda.empty_cache()

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
                icl_w_document=True,
                query_w_context=False,
                w_embeds=False,
                reversed_template=args.reversed_template,
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
            icl_w_document=True,
            query_w_context=True,
            w_embeds=False,
            max_multi_passage=args.multi_passages,
            seed=args.seed,
            reversed_template=args.reversed_template,
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
                    icl_w_document=False,
                    query_w_context=False,
                    w_embeds=False,
                    pipeline=mistral_model,
                    reversed_template=args.reversed_template,
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
                icl_w_document=True,
                query_w_context=True,
                w_embeds=False,
                pipeline=mistral_model,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
                reversed_template=args.reversed_template,
            )
            torch.cuda.empty_cache()

    else:
        if args.eval_trad:
            print("EVALUATING TRANSLATION")
            pipeline, ckpt = evaluate_trad(
                args.run_name,
                max_seq_len=max_seq_len,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                tmp_path=tmp_path,
                pipeline=None,
                mistral=False,
                seed=args.seed,
                comp_rate=args.comp_rate,
                fine_tuned=args.fine_tuned,
                benchmarks=benchmarks
                if args.benchmarks != "all"
                else ["Danish", "French", "Spanish", "German"],
            )
            torch.cuda.empty_cache()
        else:
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
                icl_w_document=args.icl_w_document,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
                compressed_doc_in_icl=args.compressed_doc_in_icl,
                reversed_template=args.reversed_template,
                comp_rate=args.comp_rate,
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
                    icl_w_document=args.icl_w_document,
                    pipeline=pipeline,
                    ckpt=ckpt,
                    max_multi_passage=args.multi_passages,
                    seed=args.seed,
                    compressed_doc_in_icl=args.compressed_doc_in_icl,
                    reversed_template=args.reversed_template,
                    comp_rate=args.comp_rate,
                )
