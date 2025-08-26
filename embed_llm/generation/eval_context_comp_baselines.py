import argparse
import json
import logging
import os
import random
from llmlingua import PromptCompressor
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm, trange
from embed_llm.generation.utils import (
    ensure_reproducibility, 
    eval_logger_info,
    create_prompt_prefix,
    create_prompt,
    EVAL_DATA_PATH,
    METRIC_EVALUATION,
    TRAD_DATA_PATH
)
from embed_llm.monitoring.utils import set_logger  # noqa: E402
from embed_llm.generation.metrics import (  # noqa: E402
    get_approx_em,
    get_bleu_score,
    get_em,
    get_f1_score,
    get_substring_match_score,
    get_rouge_score,  # noqa: E402
    metric_max_over_ground_truths,
)




logger = logging.getLogger(__name__)




class Pipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def create_QA_dataset(
    n_samples: int,
    new_questions: list[str],
    new_context: list[list[str]],
    new_answers: list[str],
    prompt_str: list[str],
    to_embed_str: list[str] | None = None,
    query_w_context: bool = False,
    w_embeds: bool = True,
    cat_multi_passages: bool = False,
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
            cat_multi_passages=cat_multi_passages,
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
    pipeline: Pipeline | None = None,  # Pipeline for the model
):
    overall_compressed_texts = []
    overall_list_prompts = []
    compress_ratio = 0
    for i in trange(0, n_samples):
        if not w_embeds:
            to_compress_text = []

            overall_list_prompts.append(
                "".join(text_prompt_prefix)
                + eng_text[i]
                + f"\nQuestion: Translate the previous document into {benchmark}.\nAnswer: "
            )
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
    max_multi_passage: int = 1,
    seed: float = 0.42,
    compressed_doc_in_icl: bool = False,
    comp_rate: float | None = None,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    cat_multi_passages: bool = False,  # If True, use together multi-passage retrieval
    max_doc_len: int | None = None,  # Maximum length of documents
    accelerator: Accelerator | None = None,  # For distributed training
    no_context: bool = False,  # If True, do not use context in examples and last query
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


        random.shuffle(c, random=lambda: seed)
        questions, context, answers = zip(*c)

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")

        prompt_str, to_embed_str = create_prompt_prefix(
            queries=questions,
            answers=[answer[0] for answer in answers],
            docs=None if no_context else context,
            max_examples=icl_examples,
            compressed_doc_in_icl=compressed_doc_in_icl,
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

        def collate_fn(batch):
            inputs = pipeline.tokenizer(
                [sample["prompt"] for sample in batch],
                return_tensors="pt",
                padding=True,
            )
            return {"inputs": inputs, "labels": [sample["answers"] for sample in batch]}

        n_samples = (
            len(new_questions) if n_samples is None  else min(n_samples, len(new_questions))
        )
        if benchmark == 'CNN':
            n_samples = min(n_samples, 1000)
            max_seq_len = 256
            
        dataset, compress_ratio = create_QA_dataset(
            n_samples=n_samples,
            new_questions=new_questions,
            new_context=new_context,
            prompt_str=prompt_str,
            to_embed_str=to_embed_str,
            new_answers=new_answers,
            query_w_context=True if llm_lingua is None  and not no_context else False, # If we want to test the decoder without context
            w_embeds= True if llm_lingua is not None else False,
            cat_multi_passages=cat_multi_passages,
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

                if "F1" not in metrics[benchmark].keys():
                    metrics[benchmark]["F1"] = {}

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

                metrics[benchmark]["EM"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_samples": n_samples,
                    "compress_ratio": compress_ratio,
                    "compressed_icl": compressed_doc_in_icl,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
                    "max_doc_len": max_doc_len,
                    "multi_passages": max_multi_passage,
                    "llmlingua2": use_llmlingua2,
                    "temp": 0.0,
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

                metrics[benchmark]["F1"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_f1,
                    "n_samples": n_samples,
                    "compress_ratio": compress_ratio,
                    "compressed_icl": compressed_doc_in_icl,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "prompt_compressor_name": prompt_compressor_name,
                    "multi_passages": max_multi_passage,
                    "max_doc_len": max_doc_len,
                    "llmlingua2": use_llmlingua2,
                    "temp": 0.0,
                }
                print(
                    f"Bench: {benchmark},  EM {value_em}, Approx EM {value_approx}, F1 {value_f1}",
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


                metrics[benchmark]["ROUGE"] = {
                    "n_samples": n_samples,
                    "icl_examples": icl_examples,
                    "Metric": value_rouge,
                    "n_samples": n_samples,
                    "compress_ratio": compress_ratio,
                    "compressed_icl": compressed_doc_in_icl,
                    "llm_name": "mistral" if llm_name is None else llm_name,
                    "together_mp": cat_multi_passages,
                    "multi_passages": max_multi_passage,
                    "max_doc_len": max_doc_len,
                    "temp": 0.0,
                }
                print(
                    f"Bench: {benchmark},  ROUGE {value_rouge}",
                )
            else:
                raise NotImplementedError(
                    f"Metric {METRIC_EVALUATION[benchmark]} is not implemented for benchmark {benchmark}"
                )

            print(
                f"Context |  query | gen sequence | answer: {list(zip(new_context, new_questions, generated_sequences, new_answers))[-1]}",
            )

    if accelerator.is_main_process:
        if llm_lingua is not None:
            run_name = prompt_compressor_name.split("/")[-1] + llm_name.split("/")[-1]
        else:
            run_name = "baseline_" + llm_name.split("/")[-1]
        
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

    return pipeline


def evaluate_trad(
    prompt_compressor_name: str | None,
    llm_name: str,  # mistralai/Mistral-7B-v0.3 "meta-llama/Meta-Llama-3-8B"
    max_seq_len: int = 128,
    use_llmlingua2: bool = False,  # If True, use llmlingua2 for prompt compression
    benchmarks: list[str] = ["Danish", "French", "Spanish", "German"],
    max_bs: int = 4,
    output_file: str = None,
    n_samples: int | None = 1000,
    pipeline: Pipeline | None = None,
    seed: float = 0.42,
    comp_rate: float | None = None,
    compressed_doc_in_icl: bool = False,  # Not used for translation
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

        # fixed_random = random.Random()
        # fixed_random.seed(42)
        # fixed_random.shuffle(c)
        random.shuffle(c, random=lambda: seed)
        text, traduction = zip(*c)
        embed_prompt = []

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

        n_samples = len(text) if n_samples is None  else min(n_samples, len(text))

        dataset, compress_ratio = create_ts_dataset(
            n_samples=n_samples,
            w_embeds=True if llm_lingua is not None else False,
            llm_lingua=llm_lingua,
            comp_rate=comp_rate,
            pipeline=pipeline,
            benchmark=benchmark,
            eng_text=text,
            embed_prompt=embed_prompt,
            traduction=traduction,
            text_prompt_prefix=text_prompt_prefix,
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
            metrics[benchmark]["BLEU"] = {
                "n_samples": n_samples,
                "Metric": bleu_score,
                "compress_ratio": compress_ratio,
                "language": benchmark,
                "compressed_icl": compressed_doc_in_icl,
                "llm_name": llm_name,
                "prompt_compressor_name": prompt_compressor_name,
                "llmlingua2": use_llmlingua2,
                "temp": 0.0,
            }

    if accelerator.is_main_process:
        if llm_lingua is not None:
            run_name = (
                prompt_compressor_name.split("/")[-1]
                + llm_name.split("/")[-1]
                + "flores")
            
        else:
            run_name = (
                "baseline_"
                + llm_name.split("/")[-1]
                +  "flores")
            


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

                overall_results[run_name][metric].append(
                    metrics[benchmark][metric]
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
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--multi_passages", type=int, default=1)
    parser.add_argument(
        "--cat_multi_passages",
        action="store_true",
    )
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--n_icl_exs", type=int, default=None)
    parser.add_argument("--no_context", action="store_true")
    parser.add_argument("--compressed_doc_in_icl", action="store_true")
    parser.add_argument(
        "--comp_rate", type=float, default=None
    )  # can enable to fix number of memory tokens if > 0
    parser.add_argument("--eval_trad", action="store_true")

    parser.add_argument(
        "--llm_name",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
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
    n_samples = args.n_samples
    eval_logger_info(
        logger, f"Evaluating with {n_samples} passages the model {args.llm_name}"
    )
    if args.eval_trad:
        eval_logger_info(
            logger,
            f"EVALUATING Translation with Flores dataset",
        )
        pipeline = evaluate_trad(
            max_bs=args.bs,
            output_file=output_file,
            n_samples=n_samples,
            pipeline=pipeline,
            seed=args.seed,
            comp_rate=args.comp_rate,
            benchmarks=benchmarks
            if args.benchmarks != "all"
            else ["Danish", "French", "Spanish", "German"],
            compressed_doc_in_icl=args.compressed_doc_in_icl,
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
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            icl_examples=icl_tests[0],
            pipeline=pipeline,
            max_multi_passage=args.multi_passages,
            seed=args.seed,
            compressed_doc_in_icl=args.compressed_doc_in_icl,
            comp_rate=args.comp_rate,
            cat_multi_passages=args.cat_multi_passages,
            prompt_compressor_name=args.compressor_name,
            llm_name=args.llm_name,
            use_llmlingua2=args.use_llmlingua2,
            max_doc_len=args.max_doc_len,
            accelerator=accelerator,
            no_context=args.no_context,
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
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                icl_examples=icl_ex,
                pipeline=pipeline,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
                compressed_doc_in_icl=args.compressed_doc_in_icl,
                comp_rate=args.comp_rate,
                cat_multi_passages=args.cat_multi_passages,
                prompt_compressor_name=args.compressor_name,
                llm_name=args.llm_name,
                use_llmlingua2=args.use_llmlingua2,
                max_doc_len=args.max_doc_len,
                accelerator=accelerator,
                no_context=args.no_context,
            )
