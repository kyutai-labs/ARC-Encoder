import argparse
import json
import logging
import os
import random
import subprocess as sp
from llmlingua import PromptCompressor
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm, trange
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
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/unfiltered_nocontext_triviaqa/trivia_qa_valid.jsonl",  # unfiltered.nocontext
    "HotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/Hotpot_qa_test.jsonl",
    "SQUAD": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/squad_test.jsonl",  # Dev set of the SQuAD v1 dataset
    "FullWikiHotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/hotpot_dev_fullwiki.jsonl",  # Dev set of the FullWiki HotpotQA dataset
    "NarrativeQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/narrativeqa_test.jsonl",
    "NarrativeQA_split": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/narrativeqa_test_split.jsonl",
    "DistractorHotpotQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_ReadComp/hotpot_dev_distractor_v1.jsonl",
}

METRIC_EVALUATION = {
    "NQ": get_em,
    "TRIVIAQA": get_em,
    "HotpotQA": get_em,
    "SQUAD": get_em,
    "FullWikiHotpotQA": get_em,
    "NarrativeQA": get_em,
    "NarrativeQA_split": get_em,
    "DistractorHotpotQA": get_em,  # Added for the Distractor HotpotQA dataset
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
        list_embed = []
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
            list_prompt.append(f"\nQuestion: {query}\nAnswer:")
        return list_prompt, list_embed


def create_QA_dataset(
    n_samples: int,
    new_questions: list[str],
    new_context: list[list[str]],
    new_answers: list[str],
    prompt_str: list[str],
    to_embed_str: list[str] | None = None,
    query_w_context: bool = False,
    w_embeds: bool = True,
    together_multi_passages: bool = False,
    llm_lingua: PromptCompressor | None = None,
    comp_rate: float | None = None,  # Compression rate for llmlingua2
    pipeline: Pipeline | None = None,  # Pipeline for the model
):
    overall_compressed_texts = []
    overall_list_prompts = []
    compress_ratio = 0
    for i in trange(0, n_samples):
        batch_list_prompt, text_to_embed = create_prompt(
            prefix_prompt=prompt_str,
            prefix_embed=to_embed_str,
            doc=new_context[i],
            w_embeds=w_embeds,
            query=new_questions[i],
            wdoc=query_w_context,
            together_multi_passages=together_multi_passages,
        )

        overall_list_prompts.append(batch_list_prompt)

        compressed_text = []
        for k, text in enumerate(text_to_embed):
            try:
                compressed_text.append(
                    llm_lingua.compress_prompt(
                        text,
                        instruction="",
                        question="",
                        rate=1 / comp_rate,
                    )["compressed_prompt"]
                )
            except AttributeError as e:
                logger.error(
                    f"Error compressing text: {text} with error: {e}. Using original text."
                )
                compressed_text.append(text)

            if k == len(text_to_embed) - 1:
                compress_ratio += len(
                    pipeline.tokenizer.encode(text, padding=False)
                ) / len(pipeline.tokenizer.encode(compressed_text[-1], padding=False))

        if len(compressed_text) == 0:
            compressed_text.append("")

        overall_compressed_texts.append(compressed_text)

        if not w_embeds:
            compress_ratio += 1

    dataset = []
    for list_prompt, compressed_text, new_answer in zip(
        overall_list_prompts, overall_compressed_texts, new_answers
    ):
        prompt = ""

        while len(compressed_text) < len(list_prompt):
            compressed_text.append("")

        for full_text, comp_text in zip(list_prompt, compressed_text):
            prompt += full_text + comp_text

        dataset.append({"prompt": prompt, "answers": new_answer})
    return dataset, compress_ratio / n_samples


def create_ts_dataset(
    n_samples: int,
    benchmark: str,
    eng_text: list[str],
    embed_prompt: list[str],
    traduction: list[str],
    text_prompt_prefix: list[str],
    w_embeds: bool = True,
    llm_lingua: PromptCompressor | None = None,
    comp_rate: float | None = None,  # Compression rate for llmlingua2
    new_template: bool = False,  # If True, use the new template for translation
    pipeline: Pipeline | None = None,  # Pipeline for the model
):
    overall_compressed_texts = []
    overall_list_prompts = []
    compress_ratio = 0
    for i in trange(0, n_samples):
        if not w_embeds:
            to_compress_text = []
            if not new_template:
                overall_list_prompts.append(
                    "".join(text_prompt_prefix) + eng_text[i] + "\nTranslation:"
                )
            else:
                overall_list_prompts.append(
                    "".join(text_prompt_prefix)
                    + eng_text[i]
                    + f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer: "
                )
        else:
            if not new_template:
                overall_list_prompts.append(text_prompt_prefix + ["\nTranslation:"])
            else:
                overall_list_prompts.append(
                    text_prompt_prefix
                    + [
                        f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer:"
                    ]
                )
            to_compress_text = embed_prompt + [eng_text[i]]
        compressed_text = []

        for k, text in enumerate(to_compress_text):
            try:
                compressed_text.append(
                    llm_lingua.compress_prompt(
                        text,
                        instruction="",
                        question="",
                        rate=1 / comp_rate,
                    )["compressed_prompt"]
                )
            except AttributeError as e:
                logger.error(
                    f"Error compressing text: {text} with error: {e}. Using original text."
                )
                compressed_text.append(text)

            if k == len(to_compress_text) - 1:
                compress_ratio += len(
                    pipeline.tokenizer.encode(text, padding=False)
                ) / len(pipeline.tokenizer.encode(compressed_text[-1], padding=False))
            if len(compressed_text) == 0:
                compressed_text.append("")
        overall_compressed_texts.append(compressed_text)

        if not w_embeds:
            compress_ratio += 1

    dataset = []
    for list_prompt, compressed_text, new_answer in zip(
        overall_list_prompts, overall_compressed_texts, traduction
    ):
        prompt = ""

        while len(compressed_text) < len(list_prompt):
            compressed_text.append("")

        for full_text, comp_text in zip(list_prompt, compressed_text):
            prompt += full_text + comp_text
        dataset.append({"prompt": prompt, "answers": new_answer})
    return dataset, compress_ratio / n_samples


def evaluate_QA(
    benchmarks: list[str],
    prompt_compressor_name: str | None,
    llm_name: str,  # mistralai/Mistral-7B-v0.3 "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 256,
    pipeline: Pipeline | None = None,
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    icl_examples: int = 0,
    w_embeds: bool = True,  # To test baseline LLM
    query_w_context: bool = False,
    icl_w_document: bool = True,
    max_multi_passage: int = 1,
    seed: float = 0.42,
    compressed_doc_in_icl: bool = False,
    comp_rate: float | None = None,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    together_multi_passages: bool = False,  # If True, use together multi-passage retrieval
    max_samples: bool = False,  # If True, use all the samples in the dataset
    max_doc_len: int | None = None,  # Maximum length of documents
    accelerator: Accelerator | None = None,  # For distributed training
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

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
                        # There is around 3 chars per tokens
                        context.append([data["passages"][0].strip()[: max_doc_len * 3]])
                    else:
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

        def collate_fn(batch):
            inputs = pipeline.tokenizer(
                [sample["prompt"] for sample in batch],
                return_tensors="pt",
                padding=True,
            )
            return {"inputs": inputs, "labels": [sample["answers"] for sample in batch]}

        n_samples = (
            len(new_questions) if n_samples is None or max_samples else n_samples
        )
        dataset, compress_ratio = create_QA_dataset(
            n_samples=n_samples,
            new_questions=new_questions,
            new_context=new_context,
            prompt_str=prompt_str,
            to_embed_str=to_embed_str,
            new_answers=new_answers,
            query_w_context=query_w_context,
            w_embeds=w_embeds,
            together_multi_passages=together_multi_passages,
            llm_lingua=llm_lingua,
            comp_rate=comp_rate,
            pipeline=pipeline,
        )
        eval_logger_info(logger, f"Compression ratio: {compress_ratio}")
        dataloader = DataLoader(dataset, batch_size=max_bs, collate_fn=collate_fn)
        dataloader = accelerator.prepare(dataloader)

        generated_sequences = []
        references = []
        for j, batch in enumerate(dataloader):
            with torch.no_grad():
                if batch["inputs"]["input_ids"].numel() > 32768 * len(
                    batch["inputs"]["input_ids"]
                ):  # Avoid OOM
                    batch["inputs"]["input_ids"] = batch["inputs"]["input_ids"][
                        :, :32768
                    ]
                outputs = pipeline.model.generate(
                    **batch["inputs"],
                    max_new_tokens=max_seq_len,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                )
            # Detach and move to CPU
            outputs = accelerator.gather(outputs)
            input_ids = accelerator.gather(batch["inputs"]["input_ids"])

            decoded = [
                pipeline.tokenizer.decode(
                    output[input_id.shape[-1] :],
                    skip_special_tokens=True,
                ).split("\n\n")[0]
                for output, input_id in zip(outputs, input_ids)
            ]
            generated_sequences.extend(decoded)
            references.extend(batch["labels"])

            if j % 50 == 0 or j == len(dataloader) - 1:
                logger.info(
                    f"Processed {(j + 1) * max_bs} samples out of {len(dataloader) * max_bs} for benchmark {benchmark}"
                )

        if accelerator.is_main_process:
            if METRIC_EVALUATION[benchmark] == get_em:
                value_em = sum(
                    [
                        metric_max_over_ground_truths(get_em, pred, gts)
                        for pred, gts in zip(generated_sequences, references)
                    ]
                ) / len(generated_sequences)

                value_approx = sum(
                    [
                        metric_max_over_ground_truths(get_approx_em, pred, gts)
                        for pred, gts in zip(generated_sequences, references)
                    ]
                ) / len(generated_sequences)

                if "EM" not in metrics[benchmark].keys():
                    metrics[benchmark]["EM"] = {}
                metrics[benchmark]["EM"]["0"] = {}

                if "F1" not in metrics[benchmark].keys():
                    metrics[benchmark]["F1"] = {}
                metrics[benchmark]["F1"]["0"] = {}

                n_answer_in_context = (
                    sum(
                        [
                            metric_max_over_ground_truths(get_approx_em, cont, gts)
                            for cont, gts in zip(
                                list(new_context), references[:n_samples]
                            )
                        ]
                    )
                    / n_samples
                )

                value_xrag, _ = get_substring_match_score(
                    generated_sequences, references
                )

                metrics[benchmark]["EM"]["0"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "w_context_in_examples": icl_w_document,
                    "w_context_w_query": query_w_context,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio,
                    "compressed_icl": compressed_doc_in_icl,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": together_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
                    "max_doc_len": max_doc_len,
                    "llmlingua2": use_llmlingua2,
                }
                value_f1 = (
                    sum(
                        [
                            metric_max_over_ground_truths(get_f1_score, pred, gts)
                            for pred, gts in zip(generated_sequences, references)
                        ]
                    )
                    / n_samples
                )

                metrics[benchmark]["F1"]["0"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "w_context_in_examples": icl_w_document,
                    "w_context_w_query": query_w_context,
                    "Metric": value_f1,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio,
                    "compressed_icl": compressed_doc_in_icl,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": together_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
                    "max_doc_len": max_doc_len,
                    "llmlingua2": use_llmlingua2,
                }

            else:
                raise NotImplementedError(
                    f"Metric {METRIC_EVALUATION[benchmark]} is not implemented for benchmark {benchmark}"
                )

            print(
                f"Context |  query | gen sequence | answer: {list(zip(new_context, new_questions, generated_sequences, new_answers))[-1]}",
            )

            print(
                f"Bench: {benchmark},  EM {value_em}, Approx EM {value_approx}, F1 {value_f1}",
            )
    if accelerator.is_main_process:
        if w_embeds:
            run_name = prompt_compressor_name.split("/")[-1] + llm_name.split("/")[-1]
        else:
            run_name = "baseline_" + llm_name.split("/")[-1]
        ckpt = 0

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
                if (
                    "0"
                    not in overall_results[run_name][str(ckpt)][benchmark][
                        metric
                    ].keys()
                ):
                    overall_results[run_name][str(ckpt)][benchmark][metric]["0"] = []
                overall_results[run_name][str(ckpt)][benchmark][metric]["0"].append(
                    metrics[benchmark][metric]["0"]
                )
        with open(
            output_file,
            "w",
        ) as f:
            json.dump(overall_results, f, indent=4)

    return pipeline


def evaluate_trad(
    prompt_compressor_name: str | None,
    llm_name: str,  # mistralai/Mistral-7B-v0.3 "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 2048,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    benchmarks: list[str] = ["Danish", "French", "Spanish", "German"],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    pipeline: Pipeline | None = None,
    w_embeds: bool = True,  # To test baseline LLM
    seed: float = 0.42,
    comp_rate: float | None = None,
    compressed_doc_in_icl: bool = False,  # Not used for translation
    new_template: bool = True,  # If True, use the old template for translation (without "Document:" prefix)
    europarl: bool = False,  # If True, use Europarl dataset instead of Flores
    max_samples: bool = False,  # If True, use all the samples in the dataset
    accelerator: Accelerator | None = None,  # For distributed training
):
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
            max_seq_len = 128

        else:
            eval_data = EUROPARL_TRAD_DATA_PATH[benchmark]

            text = []
            traduction = []

            with open(eval_data, "r") as f:
                for line in f:
                    data = json.loads(line)
                    traduction.append(data["answer"].strip().replace("\n", " "))
                    text.append(data["passage"].strip().replace("\n", " "))

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
                    + "\n\nDocument: "
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
                    + "\n\nDocument: "
                ]

        new_text, new_trad = (
            list(text[5:]),
            list(traduction[5:]),
        )

        text, traduction = new_text, new_trad

        def collate_fn(batch):
            inputs = pipeline.tokenizer(
                [sample["prompt"] for sample in batch],
                return_tensors="pt",
                padding=True,
            )
            return {"inputs": inputs, "labels": [sample["answers"] for sample in batch]}

        n_samples = len(text) if n_samples is None or max_samples else n_samples
        if europarl:
            n_samples = 1000 if n_samples is None or max_samples else n_samples
            max_seq_len = 2048
        dataset, compress_ratio = create_ts_dataset(
            n_samples=n_samples,
            w_embeds=w_embeds,
            llm_lingua=llm_lingua,
            comp_rate=comp_rate,
            pipeline=pipeline,
            benchmark=benchmark,
            eng_text=text,
            embed_prompt=embed_prompt,
            traduction=traduction,
            text_prompt_prefix=text_prompt_prefix,
            new_template=new_template,
        )
        eval_logger_info(logger, f"Compression ratio: {compress_ratio}")
        dataloader = DataLoader(dataset, batch_size=max_bs, collate_fn=collate_fn)
        dataloader = accelerator.prepare(dataloader)
        generated_sequences = []
        references = []
        metrics[benchmark]["BLEU"] = {}
        for j, batch in enumerate(dataloader):
            with torch.no_grad():
                outputs = pipeline.model.generate(
                    **batch["inputs"],
                    max_new_tokens=max_seq_len,
                    pad_token_id=pipeline.tokenizer.eos_token_id,
                )
            # Detach and move to CPU
            outputs = accelerator.gather(outputs)
            input_ids = accelerator.gather(batch["inputs"]["input_ids"])
            decoded = [
                pipeline.tokenizer.decode(
                    output[input_id.shape[-1] :],
                    skip_special_tokens=True,
                ).split("\n\n")[0]
                for output, input_id in zip(outputs, input_ids)
            ]
            generated_sequences.extend(decoded)
            references.extend(batch["labels"])

            if j % 100 == 0 or j == len(dataloader) - 1:
                logger.info(
                    f"Processed {(j + 1) * max_bs} samples out of {len(dataloader) * max_bs} for benchmark {benchmark}"
                )

        if accelerator.is_main_process:
            bleu_score = get_bleu_score(references, generated_sequences)
            print("BLEU score:", bleu_score)
            metrics[benchmark]["BLEU"]["0"] = {
                "n_samples": n_samples,
                "Metric": bleu_score,
                "compress_ratio": compress_ratio,
                "language": benchmark,
                "new_template": new_template,
                "compressed_icl": compressed_doc_in_icl,
                "llm_name": llm_name,
                "prompt_compressor_name": prompt_compressor_name,
                "llmlingua2": use_llmlingua2,
                "europarl": europarl,
            }

    if accelerator.is_main_process:
        if w_embeds:
            run_name = (
                prompt_compressor_name.split("/")[-1]
                + llm_name.split("/")[-1]
                + ("europarl" if europarl else "flores")
            )
        else:
            run_name = (
                "baseline_"
                + llm_name.split("/")[-1]
                + ("europarl" if europarl else "flores")
            )

        ckpt = 0

        with open(
            output_file,
            "r",
        ) as f:
            overall_results = json.load(f)
        if run_name not in overall_results.keys():
            overall_results[run_name] = {}
        if str(ckpt) not in overall_results[run_name].keys():
            overall_results[run_name][str(ckpt)] = {}

        for benchmark in metrics.keys():
            for metric in metrics[benchmark].keys():
                if metric not in overall_results[run_name][str(ckpt)].keys():
                    overall_results[run_name][str(ckpt)][metric] = {}
                if "0" not in overall_results[run_name][str(ckpt)][metric].keys():
                    overall_results[run_name][str(ckpt)][metric]["0"] = []
                overall_results[run_name][str(ckpt)][metric]["0"].append(
                    metrics[benchmark][metric]["0"]
                )
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
    parser.add_argument("--bs", type=int, default=64)
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
        "--comp_rate", type=float, default=None
    )  # can enable to fix number of memory tokens if > 0
    parser.add_argument("--eval_trad", action="store_true")

    parser.add_argument(
        "--llm_name",
        type=str,
        choices=["mistralai/Mistral-7B-v0.3", "meta-llama/Llama-3.1-8B"],
        default="mistralai/Mistral-7B-v0.3",
    )
    parser.add_argument("--query_w_context", action="store_true")
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
    parser.add_argument(
        "--max_doc_len",
        type=int,
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    args = arg_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.llm_name, device_map="auto")

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()
    pipeline = Pipeline(
        model=model,
        tokenizer=tokenizer,
    )

    if args.benchmarks == "all":
        benchmarks = ["NQ", "TRIVIAQA", "SQUAD"]
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
    eval_logger_info(
        logger, f"Evaluating with {n_passages} passages the model {args.llm_name}"
    )
    if args.eval_trad:
        eval_logger_info(
            logger,
            f"EVALUATING Translation with {'Flores' if not args.europarl else 'Europarl'} dataset",
        )
        pipeline = evaluate_trad(
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_passages,
            w_embeds=args.wo_embeds,
            pipeline=pipeline,
            seed=args.seed,
            comp_rate=args.comp_rate,
            benchmarks=benchmarks
            if args.benchmarks != "all"
            else ["Danish", "French", "Spanish", "German"],
            compressed_doc_in_icl=args.compressed_doc_in_icl,
            europarl=args.europarl,
            max_samples=args.max_samples,
            prompt_compressor_name=args.compressor_name,
            llm_name=args.llm_name,
            use_llmlingua2=args.use_llmlingua2,
            accelerator=accelerator,
            max_seq_len=max_seq_len,
        )
        torch.cuda.empty_cache()
    else:
        eval_logger_info(
            logger,
            f"EVALUATING QA with {args.multi_passages} passages and {icl_tests[0]} ICL examples",
        )
        pipeline = evaluate_QA(
            benchmarks,
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
            max_doc_len=args.max_doc_len,
            accelerator=accelerator,
        )

        for icl_ex in icl_tests[1:]:
            eval_logger_info(
                logger,
                f"EVALUATING QA with {args.multi_passages} passages and {icl_ex} ICL examples",
            )
            pipeline = evaluate_QA(
                benchmarks,
                max_bs=args.bs,
                output_file=output_file,
                n_samples=n_passages,
                max_seq_len=max_seq_len,
                icl_examples=icl_ex,
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
                max_doc_len=args.max_doc_len,
                accelerator=accelerator,
            )
