" This file create a config dictionary to launch training with the specified hyperparameters. "
import argparse
from hashlib import sha1
import yaml


def main(args):

    if args.llm_name == "Mistral7B":
        with open(
            "/home/hippolytepilchen/code/embed_llm/config/default/default_mistral.yaml"
        ) as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError(f"{args.llm_name} not supported yet !")

    config["continuation"] = args.continuation
    config["pipeline"]["prefix_prompt"] = args.prefix_prompt
    config["llm_name"] = args.llm_name

    config["pipeline"]["trainable_llm"] = not args.not_train_llm
    config["pipeline"]["train_only_pooling"] = args.train_only_pooling
    config["pipeline"]["w_embeds"] = not args.wo_embeds
    config["pipeline"]["mlp_project"]["n_layers"] = args.proj_n_layers
    config["pipeline"]["n_truncated_layers"] = args.n_truncated_layers
    config["pipeline"]["gate_bottleneck"] = args.gate_bottleneck
    assert args.embedder_name == "NVEmbed"
    config["pipeline"]["do_pool"] = args.not_pool
    config["pipeline"]["trainable_llm"] = not args.not_train_llm

    if args.no_data:
        if "data" in config.keys():
            del config["data"]

    if args.train_embedder:
        config["pipeline"]["embedder_name"] = args.llm_name
        config["pipeline"]["trainable_embedder"] = True
        config["pipeline"]["pooling_module"]["type"] = args.pooling

    else:
        del config["pipeline"]["pooling_module"]

    if not args.not_do_hybrid_task:
        config["hybrid_task"] = {}
        config["hybrid_task"]["do"] = not args.not_do_hybrid_task
        config["hybrid_task"][
            "prop_noembed_continuation"
        ] = args.prop_noembed_continuation
        config["hybrid_task"]["max_embeds"] = args.max_embeds
        config["hybrid_task"]["start_point"] = args.start_point

    if not args.not_cross_att:
        config["pipeline"]["do_both"] = not args.not_do_both
        config["pipeline"]["shared_kv"] = args.shared_kv
        config["pipeline"]["cross_att"] = not args.not_cross_att
        config["pipeline"]["cross_att_layers"] = args.cross_att_layers
        config["pipeline"]["every_cross_att"] = args.every_cross_att
        config["pipeline"]["pooled_cross_att"] = (not args.not_pooled_cross_att) and (
            args.max_embeds == 1
        )

        if args.mlm:
            config["pipeline"]["mlm"] = args.mlm

    if args.instruct_tune:
        config["instruct_tuning"] = {}
        config["instruct_tuning"]["do"] = args.instruct_tune
        config["instruct_tuning"]["cross_entropy"] = args.cross_entropy
        config["instruct_tuning"]["kl"] = args.kl
        config["instruct_tuning"]["alpha"] = args.alpha
        config["instruct_tuning"]["temp"] = args.temp

    config["batch_size"] = args.batch_size
    config["max_steps"] = args.max_steps
    config["seq_len"] = args.seq_len

    config["optim"]["max_lr"] = args.max_lr
    config["optim"]["warm_up_steps"] = args.warm_up_steps
    config["optim"]["initial_lr"] = args.initial_lr
    config["optim"]["final_lr"] = args.final_lr

    config["log_freq"] = args.log_freq
    config["eval_freq"] = args.eval_freq
    config["ckpt_freq"] = args.ckpt_freq

    # To perform gradient accumulation
    config["num_microbatches"] = args.grad_acum_steps

    name = (
        str(args.instruct_tune)
        + str(args.cross_entropy)
        + str(args.kl)
        + str(args.alpha)
        + str(args.temp)
        + str(args.no_data)
        + str(args.not_do_hybrid_task)
        + str(args.not_train_llm)
        + str(args.train_embedder)
        + str(args.max_embeds)
        + str(args.not_pooled_cross_att)
        + str(args.prop_noembed_continuation)
        + str(args.start_point)
        + str(args.batch_size)
    )

    if args.prefix:
        config["exp_name"] = args.prefix + sha1(name.encode("utf8")).hexdigest()[:8]
        config["wandb"]["run_name"] = (
            args.prefix + sha1(name.encode("utf8")).hexdigest()[:8]
        )
    else:

        name = (
            "Hybrid_LLM_"
            + str(not args.not_train_llm)
            + "_Emb_"
            + str(args.train_embedder)
            + "_MaxEmb_"
            + str(args.max_embeds)
            + "_PNoEmbed_"
            + str(args.prop_noembed_continuation)
            + "_StartPoint_"
            + str(args.start_point)
            + "_"
            + str(args.batch_size)
            + "BS"
        )

        config["exp_name"] = name
        config["wandb"]["run_name"] = name

    if args.llm_name == "Mistral7B":
        with open(
            f'/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/{config["exp_name"]}.yaml',
            "w",
        ) as file:
            yaml.dump(config, file, sort_keys=False)
    else:
        raise ValueError(f"{args.llm_name} not supported yet !")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_name",
        type=str,
        default="Mistral7B",
        choices=["Gemma7B", "Llama3.2-3B", "Mistral7B"],
    )
    parser.add_argument(
        "--wo_embeds",
        action="store_true",
        help="Whether to use word embeddings as preconditioning",
    )

    parser.add_argument(
        "--proj_n_layers",
        type=int,
        default=1,
        help="Number of layers of the projection MLP",
    )

    parser.add_argument(
        "--max_lr", type=float, default=5e-5, help="Maximum learning rate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Maximum number of steps"
    )
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=500,
        help="Percentage of steps used for the warm-up",
    )
    parser.add_argument(
        "--initial_lr", type=float, default=1e-20, help="Initial learning rate"
    )
    parser.add_argument(
        "--final_lr", type=float, default=1e-10, help="Final learning rate"
    )
    parser.add_argument("--log_freq", type=int, default=10, help="Logging frequency")
    parser.add_argument(
        "--eval_freq", type=int, default=100, help="Evaluation frequency"
    )
    parser.add_argument(
        "--ckpt_freq", type=int, default=5000, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="Prefix for the experiment"
    )
    parser.add_argument(
        "--grad_acum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--embedder_name", type=str, default="NVEmbed", help="Embedder name"
    )

    parser.add_argument(
        "--pooling",
        type=str,
        default="latent_attention",
        help="Pooling method",
        choices=["mean", "eos", "latent_attention", "reversed_latent_attention"],
    )
    parser.add_argument(
        "-n_trunc",
        "--n_truncated_layers",
        type=int,
        default=8,
        help="Number of truncated layers to extract embedding",
    )

    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Whether to train on continuation task rather than next token prediction for reconstruction",
    )
    parser.add_argument(
        "--not_cross_att",
        action="store_true",
        help="Whether to use cross-attention",
    )
    parser.add_argument(
        "--cross_att_layers",
        type=int,
        default=16,
        help="Number of layers to apply cross-attention",
    )

    parser.add_argument(
        "--not_pool",
        action="store_false",
        help="Whether to use pooling module",
    )

    parser.add_argument(
        "--shared_kv",
        action="store_true",
        help="Whether to share keys and values in cross-attention",
    )

    parser.add_argument(
        "--not_do_both",
        action="store_true",
        help="Whether to both cross-attended and concatenated embeddings",
    )

    parser.add_argument(
        "--every_cross_att",
        type=int,
        default=None,
        help="Every n layers to apply cross-attention",
    )

    parser.add_argument("--mlm", action="store_true", help="Whether to use MLM loss")

    parser.add_argument(
        "--prefix_prompt",
        action="store_true",
        help="Whether to use a prefix prompt",
    )
    parser.add_argument(
        "--train_only_pooling",
        action="store_true",
        help="Whether to use a LLM embedder but with trainable pooling",
    )

    parser.add_argument(
        "--max_n_prefixes",
        type=int,
        default=1,
        help="Maximum number of prefixes",
    )

    parser.add_argument(
        "--min_n_prefixes",
        type=int,
        default=0,
        help="Minimum number of prefixes",
    )

    parser.add_argument(
        "--prop_continuation",
        type=float,
        default=0.0,
        help="Proportion of continuation",
    )

    parser.add_argument(
        "--gate_bottleneck",
        type=int,
        default=8,
        help="Gate bottleneck",
    )

    parser.add_argument(
        "--instruct_tune",
        action="store_true",
        help="Whether to perform instruction tuning",
    )

    parser.add_argument(
        "--cross_entropy",
        action="store_true",
        help="Whether to use cross entropy loss",
    )

    parser.add_argument(
        "--kl",
        action="store_true",
        help="Whether to use KL loss",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Alpha parameter for KL loss",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature parameter for KL loss",
    )

    parser.add_argument(
        "--no_data",
        action="store_true",
        help="Whether to not use data params inside config file",
    )

    parser.add_argument(
        "--not_do_hybrid_task",
        action="store_true",
        help="Whether to use a hybrid task",
    )

    parser.add_argument(
        "--not_train_llm",
        action="store_true",
        help="Whether to train the llm",
    )

    parser.add_argument(
        "--train_embedder",
        action="store_true",
        help="Whether to train the embedder, if True embedder_name = llm_name",
    )

    parser.add_argument(
        "--max_embeds",
        type=int,
        default=1,
        help="Maximum number of embeddings",
    )

    parser.add_argument(
        "--not_pooled_cross_att",
        action="store_true",
        help="Whether to use pooled cross-attention",
    )

    parser.add_argument(
        "--prop_noembed_continuation",
        type=float,
        default=0.0,
        help="Proportion of noembed continuation",
    )

    parser.add_argument(
        "--start_point", type=float, default=0.0, help="Start gen point for hybrid task"
    )

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

    # import os
    # import yaml
    # path_config = '/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/'
    # filenames =  [file for file in os.listdir(path_config)]
    # for filename in filenames:
    #     if filename.endswith(".yaml"):
    #         with open(path_config+filename,'r') as file:
    #             config = yaml.safe_load(file)
    #         config['seq_len'] = 256
    #         with open(path_config+filename.replace('pretrain_','nopref_pretrain_'), 'w') as file:
    #             yaml.dump(config, file)
