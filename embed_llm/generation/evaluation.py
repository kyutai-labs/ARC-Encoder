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
from nltk.tokenize import sent_tokenize
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
                if index == 0:
                    prompt_str.append(
                        "Document: "
                    )
                    to_embed_str.append(doc.strip())
                elif index == max_examples - 1:
                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\n"
                    )
                else:
                    prompt_str.append(
                        f"\nQuestion: {query}\nAnswer: {answer}\n\nDocument: "
                    )
                    to_embed_str.append(doc.strip())
                    
        else:
            for query, answer, doc, _ in zip(
                queries, answers, docs, range(max_examples)
            ):
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
    prefix_prompt: list[str], prefix_embed: list[str] | None, 
    doc: str, query: str, wdoc: bool = True, w_embeds: bool = True
) -> tuple[list[str], list[str] | None]:
            
    list_prompt = prefix_prompt
    
    if prefix_embed is None and w_embeds:
        prefix_embed = []
    else:
        prefix_embed = prefix_embed
        
    assert int(wdoc)*int(w_embeds)==0, (
        "Cannot use both text context and embeddings as the document in the same time"
    )
        
    if wdoc:
        return [''.join(list_prompt.append(f"Document: {doc}\nQuestion: {query}\nAnswer:"))], prefix_embed
    else:
        if w_embeds:
            list_prompt.append(
                "Document: "
            )
            prefix_embed.append(doc.strip())
            list_prompt.append(
                f"\nQuestion: {query}\nAnswer:"
            )
        else:
            prefix_embed = None
            list_prompt.append(
                f"\nQuestion: {query}\nAnswer:"
            )
        
        return list_prompt, prefix_embed


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
    instruct_name: str = None,
    seed: float = 0.42,
    compress_rate: int | None = None,
    compressed_doc_in_icl: bool = False,
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
                    context.append('\n'.join(list((data["passages"][:max_multi_passage]))))

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
            assert not w_embeds, (
                "Query with context is not compatible with embeddings"
            )
            
        prompt_str, to_embed_str = create_prompt_prefix(
            queries=questions,
            answers=[answer[0] for answer in answers],
            docs=None if not icl_w_document else context,
            max_examples=icl_examples,
            compressed_doc_in_icl=compressed_doc_in_icl,
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
                
                for query, doc in zip(new_questions[i : i + bs], new_context[i : i + bs]):
                    text_to_embed, batch_list_prompt = create_prompt(
                        prefix_prompt=prompt_str,
                        prefix_embed=to_embed_str,
                        doc=doc,
                        w_embeds=w_embeds,
                        query=query,
                        wdoc=query_w_context,
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
                        compress_ratio += embeds / embed_tokens
                    else:
                        compress_ratio += 1
                    generated_sequences.extend(generated_sequence)
                else:
                    tokens = [
                        mistral_tokenizer.encode(prompt[0], bos=True, eos=False)
                        for prompt in texts_to_embed
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
                    "w_context_in_examples": icl_w_document,
                    "w_context_w_query": query_w_context,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    "Prop context containing the answer": n_answer_in_context,
                    "xRAG metric": value_xrag,
                    "n_passages": max_multi_passage,
                    "compress_ratio": compress_ratio / len(range(0, n_samples, max_bs)),
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
    parser.add_argument("--instruct_name", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--n_icl_exs", type=int, default=None)
    parser.add_argument("--icl_w_document", action="store_true")
    parser.add_argument("--compressed_doc_in_icl", action="store_true")
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
    icl_tests = [0, 2, 5] if args.n_icl_exs is None else [args.n_icl_exs]
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
                icl_w_document=True,
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
            icl_w_document=True,
            query_w_context=True,
            w_embeds=False,
            max_multi_passage=args.multi_passages,
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
                    icl_w_document=False,
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
                icl_w_document=True,
                query_w_context=True,
                w_embeds=False,
                pipeline=mistral_model,
                max_multi_passage=args.multi_passages,
                seed=args.seed,
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
            instruct_name=args.instruct_name,
            seed=args.seed,
            compress_rate=args.compress_rate,
            compressed_doc_in_icl = args.compressed_doc_in_icl,
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
                instruct_name=args.instruct_name,
                seed=args.seed,
                compress_rate=args.compress_rate,
                compressed_doc_in_icl = args.compressed_doc_in_icl,
            )
