import os
import torch
import json
import numpy as np
import random
from tqdm import tqdm, trange
import argparse
import subprocess as sp
from pathlib import Path
from embed_llm.models.augmented_model import EmbedAugPipeline
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from embed_llm.generation.metrics import (
    word_overlap,
    get_bleu_score,
    get_meteor,
    get_em,
    get_f1_score,
    get_rougel_score,
    metric_max_over_ground_truths,
    get_approx_em,
)


EVAL_DATA_PATH = {
    "NQ": '/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA/nq_open_hf.jsonl', # nq_data.jsonl
    "TRIVIAQA": "/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA/triviaqa_data.jsonl",
}

METRIC_EVALUATION = {"NQ": get_em, "TRIVIAQA": get_em}

# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


def set_global_seed(seed=42):
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Additional PyTorch reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_reproducibility(seed=42):
    # Set seeds as before
    set_global_seed(seed)

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    w_embeds: bool = True, # To test baseline LLM
    doc_w_context: bool = True,
    mistral: bool = False
):
    llm_path = "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B"

    results = {benchmark: {} for benchmark in benchmarks}
    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"

    if not mistral:
        if pipeline is None:
            # Get last checkpoint
            last_ckpt = sorted(
                [
                    ckpt_name
                    for ckpt_name in os.listdir(tmp_path + run_name + "/checkpoints/")
                    if (
                        Path(tmp_path + run_name + "/checkpoints/") / ckpt_name / "params.json"
                    ).exists()
                ]
            )[-1]

            pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
                llm_path=llm_path,
                ckpt_path=tmp_path + run_name + "/checkpoints/" + last_ckpt,
                device=device,
                llm_name="Mistral7B",
                embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
                max_batch_size=max_bs,
            )
            ckpt = int(last_ckpt.split("_")[-1])
            print("Evaluating checkpoint", ckpt)
        else:
            pipeline: EmbedAugPipeline = pipeline

        if max_seq_len != pipeline.pipeline_args.max_seq_len:
            print(
                "Warning: max_seq_len during model training \
                ({}) is different from the one provided ({})".format(
                    pipeline.pipeline_args.max_seq_len, max_seq_len
                )
            )
    else:
        if pipeline is None:
            mistral_model = Transformer.from_folder('/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/', device = 'cuda:0', max_batch_size = max_bs, dtype = torch.float32)
        else:
            mistral_model = pipeline
        mistral_tokenizer = MistralTokenizer.from_file("/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/tokenizer.model.v3").instruct_tokenizer.tokenizer
            

    device_count = torch.cuda.device_count()
    other_device = torch.device("cuda:1") if device_count > 1 else device
    metrics = {}

    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=len(benchmarks)
    ):

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

                else:
                    answers.append(data["answer"])
                # Take the first ranked retrieved passage
                context.append(data["passages"][0].strip())

        c = list(zip(questions, context, answers))
        random.shuffle(c, random = lambda: 0.42)
        questions, context, answers = zip(*c)
        
        print("Evaluation dataset loaded for", benchmark)

        icl_ex = ""
        for doc, query, ans in zip(
            context[:icl_examples], questions[:icl_examples], answers[:icl_examples]
        ):
            if doc_w_context:
                icl_ex += f"Document: {doc}\nQuery: {query}\nAnswer: {ans}\n\n"
            else:
                icl_ex += f"Query: {query}\nAnswer: {ans}\n\n"
    
        context, questions, answers = list(context[icl_examples:]), list(questions[icl_examples:]), list(answers[icl_examples:])
        
        
        for temp in temps:
            generated_sequences = []
            n_samples = len(questions) if n_samples is None else n_samples
            for i in trange(0, n_samples, max_bs):
                
                if w_embeds:
                    no_context_prompt = [
                            icl_ex + f"Query: {query}\nAnswer: " for query in questions[i : i + max_bs]
                        ] 

                    context_prompt = [' answer the question following the examples:\n\n' + icl_ex + f"Query: {query}\nAnswer: "  for query in questions[i : i + max_bs]
                        ]
                else:
                    if doc_w_context:
                        no_context_prompt = [
                                icl_ex + f"Document: {cont}\nQuery: {query}\nAnswer: "  for query, cont in zip(questions[i : i + max_bs],context[i : i + max_bs])
                            ] 
                    else:
                        no_context_prompt = [
                                icl_ex + f"Query: {query}\nAnswer: "  for query, cont in zip(questions[i : i + max_bs],context[i : i + max_bs])
                            ]
   
                if not mistral:
                    generated_sequence = pipeline.generate(
                        prompt_pre_embed= (['']*len(questions[i : i + max_bs]) if not pipeline.pipeline_args.w_prefix_prompt 
                            else ['Based on the context ']*len(questions[i : i + max_bs])),
                        prompt_post_embed = context_prompt if pipeline.pipeline_args.w_prefix_prompt else no_context_prompt,
                        text_conditioning=list(context[i : i + max_bs]) if w_embeds else None,
                        temperature=temp,
                        max_tokens=max_seq_len,
                        truncate_double_space=True,
                        device=device,
                        device_generation=other_device,
                    )
                else:
               
                    
                    tokens = [mistral_tokenizer.encode(prompt, bos = True, eos = False) for prompt in no_context_prompt]
                    
                    generated_sequence = generate(
                        model = mistral_model,
                        encoded_prompts = tokens,
                        max_tokens = max_seq_len,
                        temperature = temp,
                        eos_id = mistral_tokenizer.eos_id)

                generated_sequences.extend(generated_sequence)
                
            
            if METRIC_EVALUATION[benchmark] == get_em:
                value_em = sum(
                    [
                        metric_max_over_ground_truths(get_em, pred, gts)
                        for pred, gts in zip(generated_sequences, answers)
                    ]
                ) / n_samples

                value_approx = sum(
                    [
                        metric_max_over_ground_truths(get_approx_em, pred, gts)
                        for pred, gts in zip(generated_sequences, answers)
                    ]
                ) / n_samples
       
                if 'EM' not in metrics[benchmark].keys():
                    metrics[benchmark]['EM'] = {}
                metrics[benchmark]['EM'][str(temp)] = {}
                
                if 'F1' not in metrics[benchmark].keys():
                    metrics[benchmark]['F1'] = {}
                metrics[benchmark]['F1'][str(temp)] = {}
                
                metrics[benchmark]['EM'][str(temp)] = {
                    "n_samples": n_samples,
                    'icl_examples': icl_examples,
                    "Metric": value_em,
                    "approx_Metric": value_approx,
                    'answer in context': sum(
                    [
                        metric_max_over_ground_truths(get_approx_em, context, gts)
                        for context, gts in zip(context[icl_examples: n_samples + icl_examples], answers)
                    ]
                ) / n_samples
                }
                value_f1 = sum(
                    [
                        metric_max_over_ground_truths(get_f1_score, pred, gts)
                        for pred, gts in zip(generated_sequences, answers)
                    ]
                ) / n_samples
                
                metrics[benchmark]['F1'][str(temp)] = {
                    "n_samples": n_samples,
                    'icl_examples': icl_examples,
                    "Metric": value_f1,
                }
                
                print(
                    "Temperature:",
                    temp,
                    benchmark + " EM: ",
                    value_em,
                    benchmark + " Approx EM: ",
                    value_approx,
                    benchmark + " F1: ",
                    value_f1,
                )
            else:
                value = sum(
                    [
                        metric_max_over_ground_truths(
                            METRIC_EVALUATION[benchmark], pred, gts
                        )
                        for pred, gts in zip(generated_sequences, answers)
                    ]
                ) / n_samples
                print("Temperature:", temp, benchmark + ": ", value)
                    
                metrics[benchmark][str(temp)] = {
                    "n_samples": n_samples,
                    'icl_examples': icl_examples,
                    "Metric": value,
                }

    if run_name != '':
        with open(
            "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
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

    if mistral and doc_w_context:
        run_name = 'Mistral_RAG'
    elif mistral and not doc_w_context:
        run_name = 'Mistral_no_RAG'

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
                if temp not in overall_results[run_name][str(ckpt)][benchmark][metric].keys():
                    overall_results[run_name][str(ckpt)][benchmark][metric][temp] = []
                overall_results[run_name][str(ckpt)][benchmark][metric][temp].append(
                    metrics[benchmark][metric][temp]
                )

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

    print("RUN NAME => ", run_name)

    device = torch.device("cuda", 0) if torch.cuda.is_available() else "cpu"

    if ckpt is None and pipeline is None:

        # Get last checkpoint
        last_ckpt = sorted(
            [
                ckpt_name
                for ckpt_name in os.listdir(tmp_path + run_name + "/checkpoints/")
                if (
                    Path(tmp_path + run_name + "/checkpoints/")
                    / ckpt_name
                    / "params.json"
                ).exists()
            ]
        )[-1]

        pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
            llm_path=llm_path,
            ckpt_path=tmp_path + run_name + "/checkpoints/" + last_ckpt,
            device=device,
            llm_name="Mistral7B",
            embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
            max_batch_size=max_batch_size,
        )
        ckpt = int(last_ckpt.split("_")[-1])
        print("Evaluating checkpoint", ckpt)

    elif pipeline is None:
        pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
            llm_path=llm_path,
            ckpt_path=tmp_path
            + run_name
            + "/checkpoints/checkpoint_"
            + str(ckpt).zfill(6),
            device=device,
            llm_name="Mistral7B",
            embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
            max_batch_size=max_batch_size,
        )
        print("Evaluating checkpoint", str(ckpt).zfill(6))

    else:
        assert ckpt is not None
        pipeline: EmbedAugPipeline = pipeline

    if max_seq_len != pipeline.pipeline_args.max_seq_len:
        print(
            "Warning: max_seq_len during model training \
            ({}) is different from the one provided ({})".format(
                pipeline.pipeline_args.max_seq_len, max_seq_len
            )
        )

    lim_toks = max_seq_len
    valid_passage = []

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

    max_tokens = lim_toks

    results_generation = {}

    n_passages = len(valid_passage)
    assert n_passages == len(valid_passage)

    device_count = torch.cuda.device_count()
    other_device = device if device_count <= 1 else torch.device("cuda:1")

    for temp in temperatures:
        print(f"Temperature: {temp}")
        generated_sequences = []
        for i in range(0, n_passages, max_batch_size):
            passage = list(valid_passage[i : i + max_batch_size])
            generated_sequence = pipeline.generate(
                prompt_pre_embed=(
                    [""] * len(passage)
                    if not pipeline.pipeline_args.w_prefix_prompt
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
                truncate_double_space=False,
                device=device,
                device_generation=other_device,
            )

            generated_sequences.extend(generated_sequence)
        results_generation[str(temp)] = generated_sequences

    metrics = {bench: {} for bench in reconstruct_benchmarks}
    for temp in results_generation.keys():
        # for split in results_generation[temp].keys():

        generated_sequences = results_generation[str(temp)]
        gt_passage = (
            valid_passage  # train_passage if split == 'train' else valid_passage
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

        print(
            f"CKPT: {ckpt}, Temperature: {temp}, Overlap: {overlap}",
            "Bleu Score:",
            bleu_score,
            "Truncated output Bleu score:",
            trunc_bleu_score,
            "EM:",
            em,
            "Meteor:",
            meteor_score,
            "Bleu Score Avg:",
            bleu_score_avg,
        )

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
    return parser.parse_args()

if __name__ == "__main__":

    args = arg_parser()
    
    ensure_reproducibility(29)
    
    
    output_file = "/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/eval_QA_mistral.json"
    tmp_path = "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/"
        
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)
        
    max_seq_len = 512
    n_passages = 500

    mistral_model = evaluate_QA(
            '',
            ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=0,
        mistral = True,
        doc_w_context=False,
        w_embeds = False,
        # pipeline=pipeline,
        # ckpt=ckpt,
    )
        
    torch.cuda.empty_cache()
    mistral_model = evaluate_QA(
        '',
        ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=2,
        pipeline=mistral_model,
        mistral = True,
        doc_w_context=False,
        w_embeds = False,
    )
    torch.cuda.empty_cache()
    mistral_model = evaluate_QA(
        '',
        ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=5,
        pipeline=mistral_model,
        mistral = True,
        doc_w_context=False,
        w_embeds = False,
    )
    torch.cuda.empty_cache()
    
    mistral_model = evaluate_QA(
            '',
            ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=0,
        mistral = True,
        doc_w_context=True,
        w_embeds = False,
        pipeline = mistral_model,
        # pipeline=pipeline,
        # ckpt=ckpt,
    )
        
    torch.cuda.empty_cache()
    mistral_model = evaluate_QA(
        '',
        ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=2,
        pipeline=mistral_model,
        mistral = True,
        doc_w_context=True,
        w_embeds = False,
    )
    torch.cuda.empty_cache()
    mistral_model = evaluate_QA(
        '',
        ["NQ","TRIVIAQA"],
        temps=[0, 0.5, 0.7],
        max_bs=4,
        output_file=output_file,
        n_samples=n_passages,
        max_seq_len=max_seq_len,
        tmp_path=tmp_path,
        icl_examples=5,
        pipeline=mistral_model,
        mistral = True,
        doc_w_context=True,
        w_embeds = False,
    )
    torch.cuda.empty_cache()



    # for run_name in ['nopref_pretrain_nollm_trained_cont_singpassage_5darr64']:
        
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #      ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=0,
        #     w_embeds=False,
        #     doc_w_context = False,
        # )
        
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #     ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=2,
        #     w_embeds = False,
        #     doc_w_context = False,
        # )
        
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #     ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=5,
        #     pipeline=pipeline,
        #     ckpt=ckpt,
        #     w_embeds = False,
        #     doc_w_context = False,
        # )
        
        # torch.cuda.empty_cache()
        # print("Finished run Mistral no RAG")
        
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #      ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=0,
        #     w_embeds=False,
        # )
        
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #     ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=2,
        #     w_embeds = False,
        # )
        
        # torch.cuda.empty_cache()
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #     ["NQ","TRIVIAQA"],
        #     temps=[0, 0.5],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=5,
        #     pipeline=pipeline,
        #     ckpt=ckpt,
        #     w_embeds = False,
        # )
        
        # torch.cuda.empty_cache()
        # print("Finished run Mistral RAG")




    # if args.run_name is not None:
    #     print('Memory:', get_gpu_memory())
    #     run_names = [args.run_name]
    # else:
    #     run_names = [
    #         # Done
    #         'nopref_pretrain_nollm_trained_rec_multipassage_5darr64',
    #         'nopref_pretrain_nollm_trained_cont_singpassage_5darr64',
    #         "pretrain_both_trained_rec_singpassage_0f6f2a1a",
    #         "pretrain_both_trained_cont_singpassage_17c38ada",
            
            # # "pretrain_llm_trained_rec_singpassage_054f63f8",
            # # "pretrain_both_trained_cont_singpassage_17c38ada",
            # # "pretrain_llm_trained_cont_singpassage_5daaa6bc",
            # # "pretrain_llm_trained_rec_multipassage_054f63f8",
            # # "pretrain_both_trained_02_singpassage_0f6f2a1a",
            # "pretrain_both_trained_rec_multipassage_0f6f2a1a",
            # # "pretrain_both_trained_1cont_0.2textcont_singpassage_17c38ada",
            # # "pretrain_both_trained_1cont_0.5textcont_singpassage_17c38ada",
            # # "pretrain_llm_trained_02_singpassage_054f63f8",
            # # "pretrain_llm_trained_05_singpassage_054f63f8",
            # "nopref_pretrain_llm_trained_cont_singpassage_5daaa6bc",
            # "nopref_pretrain_no_trained_cont_singpassage_5daaa6bc",
            # 'nopref_pretrain_no_trained_rec_singpassage_054f63f8',
            # 'nopref_pretrain_no_trained_rec_multipassage_054f63f8',
            # # 'pretrain_both_trained_05_singpassage_0f6f2a1a',
            # 'nopref_pretrain_both_trained_02_singpassage_0f6f2a1a',
            # 'nopref_pretrain_llm_trained_02_singpassage_054f63f8',
            # 'nopref_pretrain_llm_trained_rec_multipassage_054f63f8',
            # 'nopref_pretrain_pool_trained_cont_singpassage_5daaa6bc',

            # # 'nopref_pretrain_llm_trained_07_singpassage_054f63f8',
            # 'nopref_pretrain_pool_trained_rec_singpassage_054f63f8',
            # 'nopref_pretrain_both_trained_cont_singpassage_17c38ada',
            # 'nopref_pretrain_llm_trained_rec_singpassage_054f63f8',
            # # 'nopref_pretrain_both_trained_1cont_0.5textcont_singpassage_17c38ada',
            # 'nopref_pretrain_both_trained_1cont_0.2textcont_singpassage_17c38ada',
            # 'nopref_pretrain_no_trained_rec_singpassage_8gate_054f63f8',
            # 'nopref_pretrain_both_trained_rec_singpassage_0f6f2a1a',
            # # 'nopref_pretrain_both_trained_07_singpassage_0f6f2a1a',
            # 'nopref_pretrain_both_trained_rec_multipassage_0f6f2a1a',
            # 'nopref_pretrain_llm_trained_rec_singpassage_8gate_054f63f8',
            
            
            # Weird QA

        # still training
        # 'nopref_pretrain_both_trained_rec_singpassage_2gate_0f6f2a1a',
        # 'nopref_pretrain_both_trained_rec_singpassage_16gate_0f6f2a1a',
        # 'nopref_pretrain_both_trained_rec_singpassage_4gate_0f6f2a1a',
        # 'nopref_pretrain_both_trained_rec_singpassage_8gate_0f6f2a1a',
        # 'nopref_pretrain_nollm_trained_rec_singpassage_8gate_5darr64',
        # 'nopref_pretrain_both_trained_highuseless_hybrid_singpassage_0f6f2a1a',
        # 'nopref_pretrain_both_trained_lownoembed_hybrid_singpassage_0f6f2a1a',
        # 'nopref_pretrain_both_trained_std_hybrid_singpassage_0f6f2a1a',
        # 'nopref_pretrain_both_trained_std_hybrid_multipassage_0f6f2a1a',
        # 'nopref_pretrain_pool_trained_rec_singpassage_8gate_054f63f8',
        # 'nopref_pretrain_llm_trained_std_hybrid_multipassage_054f63f8',
        # 'nopref_pretrain_llm_trained_std_hybrid_singpassage_054f63f8',
        # 'nopref_pretrain_nollm_trained_std_hybrid_multipassage_5darr64',
        # 'nopref_pretrain_nollm_trained_std_hybrid_singpassage_5darr64',
        # 'nopref_pretrain_no_trained_std_hybrid_multipassage_054f63f8',
        # 'nopref_pretrain_no_trained_std_hybrid_singpassage_054f63f8',
        # 'nopref_pretrain_pool_trained_std_hybrid_singpassage_054f63f8',
    #     ]

    # for i, run_name in enumerate(run_names):

        # print("Standard Dump")
        # pipeline, ckpt = evaluate_reconstruction_model(
        #     run_name,
        #     output_file=output_file,
        #     temperatures=[0, 0.5, 0.7],
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     eval_data_type="standard_dump",
        #     n_passages=n_passages,
        # )  # 'atlas','standard_dump'
        
        # print("Atlas")
        # pipeline, ckpt = evaluate_reconstruction_model(
        #     run_name,
        #     output_file=output_file,
        #     temperatures=[0, 0.5, 0.7],
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     eval_data_type="atlas",
        #     pipeline=pipeline,
        #     ckpt=ckpt,
        #     n_passages=n_passages,
        # )



        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #         ["NQ","TRIVIAQA"],
        #     temps=[0, 0.7],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=0,
        #     # pipeline=pipeline,
        #     # ckpt=ckpt,
        # )
        
        # torch.cuda.empty_cache()
        # # pipeline, ckpt = evaluate_QA(
        # #     run_name,
        # #     ["NQ","TRIVIAQA"],
        # #     temps=[0, 0.5],
        # #     max_bs=4,
        # #     output_file=output_file,
        # #     n_samples=n_passages,
        # #     max_seq_len=max_seq_len,
        # #     tmp_path=tmp_path,
        # #     icl_examples=2,
        # #     pipeline=pipeline,
        # #     ckpt=ckpt,
        # # )
        # # torch.cuda.empty_cache()
        # pipeline, ckpt = evaluate_QA(
        #     run_name,
        #     ["NQ","TRIVIAQA"],
        #     temps=[0, 0.7],
        #     max_bs=4,
        #     output_file=output_file,
        #     n_samples=n_passages,
        #     max_seq_len=max_seq_len,
        #     tmp_path=tmp_path,
        #     icl_examples=5,
        #     pipeline=pipeline,
        #     ckpt=ckpt,
        # )
        # torch.cuda.empty_cache()
        # print("Finished run", run_name)

