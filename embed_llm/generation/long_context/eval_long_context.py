import argparse
import json
import logging
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast
import numpy as np
from embed_llm.generation.long_context.eval_downstream import (
    TestItemDataset,
)
from embed_llm.generation.long_context.modeling_llama_flash import (
    LlamaForCausalContextLM,
)
from embed_llm.models.augmented_model import load_pipeline

from embed_llm.models.generate import generate as transformer_generate
from embed_llm.generation.metrics import (  # noqa: E402
    get_em,
    get_f1_score,
    get_rouge_score,
    get_substring_match_score,
    metric_max_over_ground_truths,
)
from embed_llm.generation.utils import (
    eval_logger_info,
    get_max_memory,
    ensure_reproducibility,
)  # noqa: E402
from embed_llm.monitoring.utils import set_logger  # noqa: E402
from embed_llm.models.utils.utils import format_for_chat  # noqa: E402
from embed_llm import TMP_PATH, MODEL_PATH, DATA_PATH  # noqa: E402

logger = logging.getLogger(__name__)
EVAL_DATA_PATH = {
    "NQA": DATA_PATH + "long_context/narrativeqa_validation.jsonl",
    "Qspr": DATA_PATH + "long_context/qasper_validation.jsonl",
    "GvRp": DATA_PATH + "long_context/govreport_validation.jsonl",
    "QMSum": DATA_PATH + "long_context/qmsum_validation.jsonl",
}


METRIC_EVALUATION = {
    "NQA": get_f1_score,
    "Qspr": get_f1_score,
    "GvRp": get_rouge_score,
    "QMSum": get_rouge_score,
}

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.llm_tokenizer = tokenizer


def evaluate_long_context(
    benchmarks,
    output_file: str,
    n_samples: int,
    pipeline: Pipeline,
    max_seq_len: int = 64,
    seed: float = 0.42,
    comp_rate: float | None = None,
    max_samples: bool = False,
    llm_name: str = "meta-llama/Llama-3.1-8B",
    max_doc_len: int | None = None,
    model_name: str = "meta-llama/Llama-3.1-8B",
    chunk_to: int = 1,
    eval_model: str = "ours",
    input_max_len: int | None = None,
):
    """Load the pipeline and evaluate it on the QA benchmarks"""

    # Creating dataset
    metrics = {}
    total_benchmarks = len(benchmarks)
    for benchmark in tqdm(
        benchmarks, desc="Evaluating benchmarks", total=total_benchmarks
    ):
        metrics[benchmark] = {}
        data_path = EVAL_DATA_PATH[benchmark]

        dataset = []
        with open(data_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))

        if n_samples is not None and not max_samples:
            random.shuffle(dataset, random=lambda: seed)
            dataset = dataset[:n_samples]
        if max_samples:
            n_samples = len(dataset)

        stop = []
        stop = list(
            set(stop + ["\n", "Ċ", "ĊĊ", "<0x0A>"])
        )  # In Llama \n is <0x0A>; In OPT \n is Ċ
        stop_token_ids = list(
            set(
                [
                    pipeline.llm_tokenizer.convert_tokens_to_ids(stop_token)
                    for stop_token in stop
                ]
                + [pipeline.llm_tokenizer.eos_token_id]
            )
        )
        if "llama" in model_name.lower():
            if (
                pipeline.llm_tokenizer.unk_token_id is not None
                and pipeline.llm_tokenizer.unk_token_id in stop_token_ids
            ):
                stop_token_ids.remove(pipeline.llm_tokenizer.unk_token_id)

        eval_logger_info(logger, f"Evaluation dataset loaded for {benchmark}")
        torch_dataset = TestItemDataset(
            eval_model=eval_model,
            llm_tokenizer=pipeline.llm_tokenizer,
            dataset=dataset,
            dataset_name=benchmark,
            compressor_tokenizer=None
            if eval_model != "ours"
            else pipeline.embed_tokenizer,
            generation_max_length=max_seq_len,
            context_max_length=max_doc_len,
            input_max_length=input_max_len,
            concat_N=chunk_to if chunk_to > 1 else None,
        )

        dataloader = DataLoader(
            torch_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: x,
        )
        generated_sequences = []
        answers = []
        for idx, batch in enumerate(tqdm(dataloader)):
            inputs = batch[0]
            test_item = inputs.pop("test_item")

            if eval_model == "ours":
                # print('Embed seqlens before pooling',inputs["embed_seqlens"])
                embeddings, embed_seqlens = pipeline.model.embedder.forward_embedder(
                    input_ids=torch.tensor(inputs["embeddings"]).to('cuda'),
                    seqlens=sum(inputs["embed_seqlens"], []),
                )
                embed_seqlens = [embed_seqlens]
                embeddings = embeddings[: inputs["n_toks_left"]]
                new_embed_seqlens = []
                ind = 0
                for size in embed_seqlens[0]:
                    if ind + size > inputs["n_toks_left"]:
                        size = inputs["n_toks_left"] - ind
                        if size > 0:
                            new_embed_seqlens.append(size)
                        break

                    new_embed_seqlens.append(size)
                    ind += size
                embed_seqlens = [new_embed_seqlens]

                if pipeline.model.embedder.cont_tok is not None:
                    sp_cont_tok = pipeline.model.embedder.cont_tok(
                        torch.tensor([0]).to(embeddings.device)
                    )
                    new_embeddings = torch.zeros(
                        (
                            len(embed_seqlens[0]) + sum(embed_seqlens[0]),
                            embeddings.shape[1],
                        ),
                        device=embeddings.device,
                        dtype=embeddings.dtype,
                    )
                    ind = 0
                    ind_new = 0
                    for size in embed_seqlens[0]:
                        new_embeddings[ind_new : ind_new + size] = embeddings[
                            ind : ind + size
                        ]
                        ind_new += size
                        ind += size

                        new_embeddings[ind_new : ind_new + 1] = sp_cont_tok.clone()

                        ind_new += 1

                    embed_seqlens = [[size + 1 for size in embed_seqlens[0]]]
                    embeddings = new_embeddings.clone()
                if pipeline.model.bridge_module is not None:
                    embeddings = pipeline.model.bridge_module(embeddings)
                inputs["insertion_lists"] = [0] * len(
                    embed_seqlens[0]
                )  # Insert after BOS token
                prefix_prompt = (
                    inputs["instruct"].split("||")[0]
                    if inputs["instruct"] is not None
                    else ""
                )
                instr_prompt = (
                    inputs["instruct"].split("||")[1]
                    if inputs["instruct"] is not None
                    else ""
                )
                suffix_prompt = (
                    inputs["instruct"].split("||")[2]
                    if inputs["instruct"] is not None
                    else ""
                )

                new_toks, _, insert_list, _ = format_for_chat(
                    [],
                    inputs["insertion_lists"],
                    pipeline.llm_tokenizer.tokenizer,
                    system_message=None,
                    instruct_prompt=instr_prompt,
                    prefix_prompt=prefix_prompt,
                    suffix_prompt=suffix_prompt,
                    generation=True,
                )
                eos_id = pipeline.llm_tokenizer.tokenizer.eos_id

                generated_tokens = transformer_generate(
                    prompt_tokens=[new_toks],  # Since one batch
                    insertion_lists=[insert_list],
                    model=pipeline.model.llm,
                    max_tokens=max_seq_len,
                    temperature=0.7 if benchmark == "GvRp" else 0.0,
                    eos_id=eos_id,
                    embed_seqlens=embed_seqlens,
                    cat_embeddings=embeddings,
                )

                produced_text = [
                    pipeline.llm_tokenizer.tokenizer.decode(generated_tokens[i])
                    for i in range(len(generated_tokens))
                ]
                prediction = [text.strip() for text in produced_text][0]

            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # can't call .to() on the num_context for replug (a int)
                prefix_input = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs["prefix_inputs"].items()
                }
                # print('Prefix input:', len(prefix_input['encoder_input_ids']), len(prefix_input['encoder_input_ids'][0]), len(prefix_input['encoder_input_ids'][0][0]))
                with torch.no_grad():
                    outputs = pipeline.model.generate(
                        **prefix_input,
                        max_new_tokens=max_seq_len,
                        min_new_tokens=1,
                        do_sample=True if benchmark == "GvRp" else False,
                        temperature=1.0,
                        top_p=0.95,
                        eos_token_id=stop_token_ids,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                prediction = pipeline.llm_tokenizer.decode(
                    outputs.sequences[0][prefix_input["input_ids"].size(1) :],
                    skip_special_tokens=True,
                ).strip()

            generated_sequences.append(prediction)
            answers.append(test_item.answer)
        logger.info(f"Generated {len(generated_sequences)} sequences for {benchmark}")
        if "question" in dataset[0]:
            logger.info(
                f"Sequences: {generated_sequences[:5]}\nAnswers: {answers[:5]}\nQuestion: {[sample['question'] for sample in dataset[:5]]}"
            )
        else:
            logger.info(f"Sequences: {generated_sequences[:1]}\nAnswers: {answers[:1]}")

        value_em = np.mean(
            [
                metric_max_over_ground_truths(get_em, pred, gts)
                for pred, gts in zip(generated_sequences, answers)
            ]
        )

        value_acc, _ = get_substring_match_score(
            generated_sequences, answers[: len(generated_sequences)]
        )

        value_f1 = np.mean(
            [
                metric_max_over_ground_truths(get_f1_score, pred, gts)
                for pred, gts in zip(generated_sequences, answers)
            ]
        )

        value_rouge = np.mean(
            [
                metric_max_over_ground_truths(get_rouge_score, pred, gts)
                for pred, gts in zip(generated_sequences, answers)
            ]
        )

        metrics[benchmark] = {
            "llm_name": llm_name,
            "model_name": model_name,
            "n_samples": n_samples,
            "max_seq_len": max_seq_len,
            "eval_model": eval_model,
            "em": value_em,
            "acc": value_acc,
            "f1": value_f1,
            "rouge": value_rouge,
            "comp_rate": comp_rate if comp_rate is not None else 1.0,
            "max_doc_len": max_doc_len,
            "chunk_to": chunk_to,
            "seed": seed,
        }

    with open(
        output_file,
        "r",
    ) as f:
        overall_results = json.load(f)

    for benchmark, metric in metrics.items():
        if benchmark not in overall_results:
            overall_results[benchmark] = []
        overall_results[benchmark].append(metric)
    with open(
        output_file,
        "w",
    ) as f:
        json.dump(overall_results, f, indent=4)


# Define a custom argument type for a list of integers
def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_model",
        type=str,
        choices=["ceped", "together_instruct", "baseline", "ours"],
        default="ours",
    )
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--chunk_to", type=int, default=1)
    parser.add_argument("--benchmarks", type=str, default="all")
    parser.add_argument("--seed", type=list_of_floats, default=[0.42])

    parser.add_argument(
        "--comp_rate", type=float, default=None
    )  # can enable to fix number of memory tokens if > 0

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
    )
    parser.add_argument(
        "--max_samples",
        action="store_true",
        help="If True, use the maximum number of samples for each benchmark",
    )
    parser.add_argument(
        "--max_doc_len",  # context max length
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input_max_len",
        type=int,
        default=None,
    )

    # CEPED "hyen/CEPED-LLaMA-2-Chat-7B"
    # Baseline "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-3.1-8B"
    # Together Instruct "togethercomputer/Llama-2-7B-32K-Instruct"

    return parser.parse_args()


if __name__ == "__main__":
    set_logger(logging.INFO)

    args = arg_parser()

    if args.eval_model == "ours":
        llm_path = os.path.join(MODEL_PATH, "Llama2-7B-Chat")
        embed_path = os.path.join(MODEL_PATH, "Llama3.2-3B")
        pipeline, ckpt = load_pipeline(
            run_name=args.model_name,
            tmp_path=TMP_PATH,
            llm_path=llm_path,
            embedder_path=embed_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_bs=1,
            comp_rate=args.comp_rate,
            bridge_ckpt=None,
            llm_type="llama_2",
            embed_type="llama",
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = LlamaConfig.from_pretrained(args.model_name)
        tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name)

        if args.eval_model == "ceped":
            model_cls = LlamaForCausalContextLM
        elif args.eval_model == "together_instruct" or args.eval_model == "baseline":
            model_cls = LlamaForCausalLM
        else:
            raise ValueError(f"Unknown eval_model: {args.eval_model}")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = config.max_position_embeddings
        config._flash_attn_2_enabled = True
        model = model_cls.from_pretrained(
            args.model_name,
            config=config,
            max_memory=get_max_memory(),
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
        model.eval()
        pipeline = Pipeline(model=model, tokenizer=tokenizer)
    if args.benchmarks == "all":
        benchmarks = EVAL_DATA_PATH.keys()
    else:
        benchmarks = [args.benchmarks]

    output_file = args.out_file
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            json.dump({}, f)

    max_seq_len = args.max_seq_len
    n_samples = args.n_samples
    eval_logger_info(
        logger, f"Evaluating with {n_samples} passages the model {args.llm_name}"
    )

    for seed in args.seed:
        ensure_reproducibility(int(seed * 100))
        evaluate_long_context(
            benchmarks,
            output_file=output_file,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            pipeline=pipeline,
            seed=seed,
            comp_rate=args.comp_rate,
            max_samples=args.max_samples,
            llm_name=args.llm_name,
            max_doc_len=args.max_doc_len,
            model_name=args.model_name,
            chunk_to=args.chunk_to,
            eval_model=args.eval_model,
            input_max_len=args.input_max_len,
        )
