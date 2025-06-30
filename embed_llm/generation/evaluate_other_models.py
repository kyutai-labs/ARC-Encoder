import argparse
import json
import logging
import os
import random
import subprocess as sp
from llmlingua import PromptCompressor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm, trange
from embed_llm.models.utils.utils import is_torchrun  # noqa: E402
from embed_llm.generation.utils import ensure_reproducibility, eval_logger_info  # noqa: E402
from embed_llm.monitoring.utils import set_logger  # noqa: E402
from embed_llm.generation.metrics import (  # noqa: E402
    get_approx_em,
    get_bleu_score,
    get_em,
    get_f1_score,
    get_substring_match_score,
    metric_max_over_ground_truths,
)

logger = logging.getLogger(__name__)
EVAL_DATA_PATH = {
    "NQ": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/nq_open_data.jsonl",  # nq_data.jsonl
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/triviaqa_data.jsonl",  # unfiltered.nocontext /lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/unfiltered_nocontext_triviaqa/trivia_qa_valid.jsonl
    "HotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/Hotpot_qa_test.jsonl",
    "SQUAD": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/squad_test.jsonl",  # Dev set of the SQuAD v1 dataset
    "FullWikiHotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/hotpot_dev_fullwiki.jsonl",  # Dev set of the FullWiki HotpotQA dataset
    "NarrativeQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/narrativeqa_test.jsonl",
}

METRIC_EVALUATION = {
    "NQ": get_em,
    "TRIVIAQA": get_em,
    "HotpotQA": get_em,
    "SQUAD": get_em,
    "FullWikiHotpotQA": get_em,
    "NarrativeQA": get_em,
}


TRAD_DATA_PATH = {
    "English": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/eng_Latn.jsonl",
    "Spanish": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/spa_Latn.jsonl",
    "French": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/fra_Latn.jsonl",
    "German": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/deu_Latn.jsonl",
    "Danish": "/lustre/scwpod02/client/kyutai-interns/helium/eval/multilingual/flores/dan_Latn.jsonl",
}

EUROPARL_TRAD_DATA_PATH = {
    "Spanish": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/translation/Europarl_Corpus/europarl_en_es.jsonl",
    "French": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/translation/Europarl_Corpus/europarl_en_fr.jsonl",
    "German": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/translation/Europarl_Corpus/europarl_en_de.jsonl",
    "Danish": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/translation/Europarl_Corpus/europarl_en_da.jsonl",
}

logger = logging.getLogger(__name__)


# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


class Pipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def create_prompt_prefix(
    queries: list[str],
    answers: list[str],
    docs: list[list[str]] | None = None,
    max_examples: int | None = None,
    compressed_doc_in_icl: bool = False,
    together_multi_passages: bool = False,
) -> tuple[list[str], list[str] | None]:
    max_examples = max_examples if max_examples is not None else len(queries)
    prompt_str = []
    to_embed_str = []

    prompt = ""
    if docs is not None:
        if compressed_doc_in_icl:
            for query, answer, doc, index in zip(
                queries, answers, docs, range(max_examples)
            ):
                if len(doc) == 1:
                    doc = doc[0]
                elif len(doc) > 1 and not together_multi_passages:
                    doc = "\n".join(doc)

                if index == 0:
                    prompt_str.append("Document: ")

                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())

                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                    )
                elif index == max_examples - 1:
                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())
                    prompt_str.append(f"\nQuestion: {query}\nAnswer: {answer}\n\n")
                else:
                    if isinstance(doc, list):
                        for d in doc:
                            to_embed_str.append(d.strip())
                            prompt_str.append("")
                        prompt_str = prompt_str[:-1]  # Remove the last empty string
                    else:
                        to_embed_str.append(doc.strip())
                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                    )

            if max_examples == 0:
                prompt_str.append("")
        else:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
                doc = "\n".join(doc)
                prompt += f"Document: {doc}\nQuestion: {query}\nAnswer: {answer}\n\n"

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
    doc: list[str],
    query: str,
    wdoc: bool = True,
    w_embeds: bool = True,
    together_multi_passages: bool = False,
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
        doc = "\n".join(doc)
        list_prompt.append(f"Document: {doc.strip()}\nQuestion: {query}\nAnswer:")
        return list_prompt, list_embed
    else:
        if w_embeds:
            last_prompt = list_prompt[-1]
            list_prompt[-1] = "".join([last_prompt, "Document: "])

            if len(doc) == 1 or together_multi_passages:
                doc = "\n".join(doc)
                list_embed.append(doc.strip())
            else:
                for d in doc:
                    list_embed.append(d.strip())
                    list_prompt.append(
                        ""
                    )  # Add an empty string to separate passages so that there are embedded separately
                list_prompt = list_prompt[:-1]  # Remove the last empty string
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")
        else:
            list_embed = None
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")
        return list_prompt, list_embed


def evaluate_QA(
    benchmarks: list[str],
    prompt_compressor_name: str | None,
    llm_name: str,  # mistralai/Mistral-7B-v0.3 "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 256,
    pipeline: Pipeline | None = None,
    temps: list[float] = [0, 0.5, 0.7, 1],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    icl_examples: int = 0,
    w_embeds: bool = True,  # To test baseline LLM
    query_w_context: bool = False,
    icl_w_document: bool = True,
    max_multi_passage: int = 1,
    seed: float = 0.42,
    compressed_doc_in_icl: bool = False,
    comp_rate: int | None = None,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    together_multi_passages: bool = False,  # If True, use together multi-passage retrieval
    max_sample: bool = False,  # If True, use all the samples in the dataset
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

    results = {benchmark: {} for benchmark in benchmarks}
    if pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(llm_name)
        pipeline = Pipeline(
            model=model,
            tokenizer=tokenizer,
        )
    else:
        pipeline = pipeline

    # Creating dataset
    metrics = {}
    if prompt_compressor_name is None:
        llm_lingua = None
    else:
        llm_lingua = PromptCompressor(
            model_name=prompt_compressor_name, use_llmlingua2=use_llmlingua2
        )
    total_benchmarks = len(benchmarks)
    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=total_benchmarks
    ):
        if benchmark == "SQUAD" and max_multi_passage > 1:
            benchmarks.remove(benchmark)
            continue

        # if benchmark == "HotpotQA":
        #     max_multi_passage = 2

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
                    context.append([data["passages"][0].strip()])

                else:
                    context.append(list((data["passages"][:max_multi_passage])))
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
            together_multi_passages=together_multi_passages,
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
                len(new_questions) if n_samples is None or max_sample else n_samples
            )
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
                        together_multi_passages=together_multi_passages,
                    )

                    batch_list_prompts.append(batch_list_prompt)
                    texts_to_embed.append(text_to_embed)

                compressed_texts = []
                for to_embed_text in texts_to_embed:
                    compressed_text = []
                    for text in to_embed_text:
                        if not use_llmlingua2:
                            compressed_text.append(
                                llm_lingua.compress(
                                    text,
                                    instruction="",
                                    question="",
                                    target_token=len(text) // int(abs(comp_rate)),
                                )
                            )
                        else:
                            compressed_text.append(
                                llm_lingua.compress(
                                    text,
                                    instruction="",
                                    question="",
                                    rate=1 / abs(comp_rate),
                                )
                            )
                    compressed_texts.append(compressed_text)
                for batch_list_prompt, compressed_text in zip(
                    batch_list_prompts, compressed_texts
                ):
                    prompt = ""

                    for full_text, comp_text in zip(batch_list_prompt, compressed_text):
                        prompt += full_text + comp_text

                    inputs = pipeline.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    outputs = pipeline.model.generate(
                        **inputs,
                        max_new_tokens=max_seq_len,
                        temperature=temp,
                        do_sample=True,
                    )
                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    generated_sequences.append(generated_text.split("\n")[0])
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
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": together_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
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
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": together_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
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
                raise NotImplementedError(
                    f"Metric {METRIC_EVALUATION[benchmark]} is not implemented for benchmark {benchmark}"
                )

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if w_embeds:
            run_name = prompt_compressor_name.split("/")[-1] + llm_name.split("/")[-1]
        else:
            run_name = "baseline_" + llm_name.split("/")[-1]
        ckpt = 0
        try:
            if run_name is not None:
                with open(
                    tmp_path + run_name + "/results_generation.json",
                    "a",
                ) as f:
                    json.dump(results, f)
        except FileNotFoundError:
            print("No result generation file found, creating a new one.")
        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

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

    return pipeline


def evaluate_trad(
    prompt_compressor_name: str | None,
    llm_name: str,  # mistralai/Mistral-7B-v0.3 "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 512,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    temps: list[float] = [0, 0.5, 0.7, 1],
    benchmarks: list[str] = ["Danish", "French", "Spanish", "German"],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    tmp_path: str = None,
    pipeline: Pipeline | None = None,
    w_embeds: bool = True,  # To test baseline LLM
    seed: float = 0.42,
    comp_rate: int | None = None,
    compressed_doc_in_icl: bool = False,  # Not used for translation
    new_template: bool = True,  # If True, use the old template for translation (without "Document:" prefix)
    europarl: bool = False,  # If True, use Europarl dataset instead of Flores
    max_sample: bool = False,  # If True, use all the samples in the dataset
):
    results = {benchmark: {} for benchmark in benchmarks}
    if pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(llm_name)
        pipeline = Pipeline(
            model=model,
            tokenizer=tokenizer,
        )
    else:
        pipeline = pipeline

    # Creating dataset
    metrics = {}
    if prompt_compressor_name is None:
        llm_lingua = None
    else:
        llm_lingua = PromptCompressor(
            model_name=prompt_compressor_name, use_llmlingua2=use_llmlingua2
        )
    total_benchmarks = len(benchmarks)

    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=total_benchmarks
    ):
        metrics[benchmark] = {}
        if not europarl:
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

        else:
            eval_data = EUROPARL_TRAD_DATA_PATH[benchmark]

            text = []
            traduction = []

            with open(eval_data, "r") as f:
                for line in f:
                    data = json.loads(line)
                    traduction.append(data["answer"].strip())
                    text.append(data["passage"].strip())

        c = list(zip(text, traduction))

        # fixed_random = random.Random()
        # fixed_random.seed(42)
        # fixed_random.shuffle(c)
        random.shuffle(c, random=lambda: seed)
        text, traduction = zip(*c)
        embed_prompt = []
        if not new_template:
            if compressed_doc_in_icl:
                text_prompt_prefix = ["Document: "]
                for doc, answ, _ in zip(text, traduction, range(4)):
                    embed_prompt.append(doc.strip())
                    text_prompt_prefix.append(
                        f"\nTranslation: {answ.strip()}\n\nDocument: "
                    )
                embed_prompt.append(text[4].strip())
                text_prompt_prefix.append(
                    f"\nTranslation: {traduction[4].strip()}\n\nDocument: "
                )

            else:
                text_prompt_prefix = [
                    "\n\n".join(
                        [
                            f"Document: {doc}\nTranslation: {answ}"
                            for doc, answ, _ in zip(text, traduction, range(5))
                        ]
                    )
                ]
        else:
            if compressed_doc_in_icl:
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

            else:
                text_prompt_prefix = [
                    "\n\n".join(
                        [
                            f"Document: {doc}\nQuestion: Translate the previous document into {benchmark}.\nAnswer: {answ.strip()}"
                            for doc, answ, _ in zip(text, traduction, range(5))
                        ]
                    )
                ]

        new_text, new_trad = (
            list(text[5:]),
            list(traduction[5:]),
        )

        text, traduction = new_text, new_trad

        print(f"Evaluation dataset loaded for {benchmark}")
        metrics[benchmark]["BLEU"] = {}
        for temp in temps:
            compress_ratio = 0
            generated_sequences = []
            n_samples = len(text) if n_samples is None or max_sample else n_samples
            for i in trange(0, n_samples, max_bs):
                texts_to_embed = []
                batch_list_prompts = []
                bs = min(max_bs, n_samples - i)

                if not w_embeds:
                    texts_to_embed = [[""] for _ in range(bs)]
                    if not new_template:
                        batch_list_prompts = [
                            "".join(text_prompt_prefix) + seq + "\nTranslation:"
                            for seq in text[i : i + bs]
                        ]
                    else:
                        batch_list_prompts = [
                            "".join(text_prompt_prefix)
                            + seq
                            + f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer: "
                            for seq in text[i : i + bs]
                        ]
                else:
                    if not new_template:
                        batch_list_prompts = [
                            text_prompt_prefix + ["\nTranslation:"] for _ in range(bs)
                        ]
                    else:
                        batch_list_prompts = [
                            text_prompt_prefix
                            + [
                                f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer:"
                            ]
                            for _ in range(bs)
                        ]

                    texts_to_embed = [embed_prompt + [seq] for seq in text[i : i + bs]]

                compressed_texts = []
                for to_embed_text in texts_to_embed:
                    compressed_text = []
                    for text in to_embed_text:
                        if not use_llmlingua2:
                            compressed_text.append(
                                llm_lingua.compress(
                                    text,
                                    instruction="",
                                    question="",
                                    target_token=len(text) // int(abs(comp_rate)),
                                )
                            )
                        else:
                            compressed_text.append(
                                llm_lingua.compress(
                                    text,
                                    instruction="",
                                    question="",
                                    rate=1 / abs(comp_rate),
                                )
                            )
                    compressed_texts.append(compressed_text)
                for batch_list_prompt, compressed_text in zip(
                    batch_list_prompts, compressed_texts
                ):
                    prompt = ""

                    for full_text, comp_text in zip(batch_list_prompt, compressed_text):
                        prompt += full_text + comp_text

                    inputs = pipeline.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    outputs = pipeline.model.generate(
                        **inputs,
                        max_new_tokens=max_seq_len,
                        temperature=temp,
                        do_sample=True,
                    )
                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True
                    )
                    generated_sequences.append(generated_text.split("\n")[0])

            bleu_score = get_bleu_score(
                traduction[: len(generated_sequences)], generated_sequences
            )
            print("BLEU score:", bleu_score)
            metrics[benchmark]["BLEU"][str(temp)] = {
                "n_samples": n_samples,
                "Metric": bleu_score,
                "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
                "language": benchmark,
                "new_template": new_template,
                "compressed_icl": compressed_doc_in_icl,
                "llm_name": llm_name,
                "prompt_compressor_name": prompt_compressor_name,
                "llmlingua2": use_llmlingua2,
            }

    if not is_torchrun() or torch.distributed.get_rank() == 0:
        if w_embeds:
            run_name = prompt_compressor_name.split("/")[-1] + llm_name.split("/")[-1]
        else:
            run_name = "baseline_" + llm_name.split("/")[-1]

        ckpt = 0
        try:
            if run_name is not None:
                with open(
                    tmp_path + run_name + "/results_generation.json",
                    "a",
                ) as f:
                    json.dump(results, f)
        except FileNotFoundError:
            print("No result generation file found, creating a new one.")
        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)

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

    return pipeline


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_passages", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--wo_embeds", action="store_false")
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument(
        "--together_multi_passages",
        action="store_true",
    )
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
        "--llm_name",
        type=str,
        choices=["mistralai/Mistral-7B-v0.3", "meta-llama/Meta-Llama-3-8B"],
        default="mistralai/Mistral-7B-v0.3",
    )
    parser.add_argument("--query_w_context", action="store_true")
    parser.add_argument("--new_template", action="store_true")
    parser.add_argument("--europarl", action="store_true")
    parser.add_argument(
        "--max_samples",
        action="store_true",
        help="If True, use the maximum number of samples for each benchmark",
    )
    parser.add_argument(
        "--compressor_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_llmlingua2",
        action="store_true",
        help="If True, use llmlingua2 for prompt compression",
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)
    temp_tests = [0]

    args = arg_parser()

    if args.benchmarks == "all":
        benchmarks = ["NQ", "TRIVIAQA", "SQUAD", "HotpotQA"]
    else:
        benchmarks = [args.benchmarks]
    icl_tests = [0, 5] if args.n_icl_exs is None else [args.n_icl_exs]
    ensure_reproducibility(29)
    output_file = (
        "/home/hippolytepilchen/code/hp_v2/results/NVEmbed/mistral/eval_mistral_translate.json"
        if args.out_file is None
        else args.out_file
    )

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)
    max_seq_len = args.max_seq_len
    n_passages = args.n_passages

    if args.eval_trad:
        eval_logger_info(logger, "EVALUATING Translation")
        pipeline = evaluate_trad(
            max_seq_len=max_seq_len,
            temps=temp_tests,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_passages,
            w_embeds=args.wo_embeds,
            pipeline=None,
            seed=args.seed,
            comp_rate=args.comp_rate,
            benchmarks=benchmarks
            if args.benchmarks != "all"
            else ["Danish", "French", "Spanish", "German"],
            compressed_doc_in_icl=args.compressed_doc_in_icl,
            new_template=args.new_template,
            europarl=args.europarl,
            max_samples=args.max_samples,
            prompt_compressor_name=args.compressor_name,
            llm_name=args.llm_name,
            use_llmlingua2=args.use_llmlingua2,
        )
        torch.cuda.empty_cache()
    else:
        eval_logger_info(logger, "EVALUATING QA")
        pipeline = evaluate_QA(
            benchmarks,
            temps=temp_tests,
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_passages,
            max_seq_len=max_seq_len,
            icl_examples=icl_tests[0],
            pipeline=None,
            w_embeds=args.wo_embeds,
            icl_w_document=args.icl_w_document,
            max_multi_passage=args.multi_passages,
            seed=args.seed,
            compressed_doc_in_icl=args.compressed_doc_in_icl,
            comp_rate=args.comp_rate,
            query_w_context=args.query_w_context,
            together_multi_passages=args.together_multi_passages,
            max_samples=args.max_samples,
            prompt_compressor_name=args.compressor_name,
            llm_name=args.llm_name,
            use_llmlingua2=args.use_llmlingua2,
        )

        for icl_ex in icl_tests[1:]:
            pipeline = evaluate_QA(
                benchmarks,
                temps=temp_tests,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                max_seq_len=max_seq_len,
                icl_examples=icl_tests[0],
                pipeline=pipeline,
                w_embeds=args.wo_embeds,
                icl_w_document=args.icl_w_document,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
                compressed_doc_in_icl=args.compressed_doc_in_icl,
                comp_rate=args.comp_rate,
                query_w_context=args.query_w_context,
                together_multi_passages=args.together_multi_passages,
                max_samples=args.max_samples,
                prompt_compressor_name=args.compressor_name,
                llm_name=args.llm_name,
                use_llmlingua2=args.use_llmlingua2,
            )
