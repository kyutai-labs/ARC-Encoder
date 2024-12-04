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
    # elif args.llm_name == "Gemma7B":
    #     with open(
    #         "/home/hippolytepilchen/code/embed_llm/config/default/default_gemma.yaml"
    #     ) as file:
    #         config = yaml.safe_load(file)
    # elif args.llm_name == "Llama3.2-3B":
    #     with open(
    #         "/home/hippolytepilchen/code/embed_llm/config/default/default_llama.yaml"
    #     ) as file:
    #         config = yaml.safe_load(file)
    else:
        raise ValueError(f"{args.llm_name} not supported yet !")

    config["llm_name"] = args.llm_name
    config["pipeline"]["w_embeds"] = not args.wo_embeds
    config["pipeline"]["norm_wo_embeds"] = args.norm_wo_embeds
    config["pipeline"]["continuation"] = args.continuation
    config["pipeline"]["mlp_project"]["hidden_dim"] = args.proj_hidden_dim
    config["pipeline"]["mlp_project"]["n_layers"] = args.proj_n_layers
    config["pipeline"]["mlp_project"]["act"] = args.proj_act
    config["pipeline"]["n_truncated_layers"] = args.n_truncated_layers
    assert args.embedder_name == "NVEmbed"
    config["pipeline"]["do_pool"] = args.not_pool
    config["pipeline"]["normalize_embeddings"] = args.no_norm_embeds
    if args.train_embedder:
        config["pipeline"]["embedder_name"] = args.llm_name
        config["pipeline"]["trainable_embedder"] = True
        config["pipeline"]["causal"] = args.causal
        config["pipeline"]["pooling_module"]["type"] = args.pooling
        config["pipeline"]["pooling_module"]["r"] = args.latent_dim
        config["pipeline"]["pooling_module"]["n_heads"] = args.n_heads
    else:
        del config["pipeline"]["causal"]
        del config["pipeline"]["pooling_module"]

    if args.cross_att:
        config["pipeline"]["do_both"] = args.do_both
        config["pipeline"]["shared_kv"] = args.shared_kv
        config["pipeline"]["cross_att"] = args.cross_att
        config["pipeline"]["cross_att_layers"] = (
            None if args.cross_att_layers is None else args.cross_att_layers
        )

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
        args.llm_name
        + str(args.wo_embeds)
        + str(args.norm_wo_embeds)
        + str(args.proj_hidden_dim)
        + str(args.proj_n_layers)
        + args.proj_act
        + str(args.batch_size)
        + str(args.max_steps)
        + str(args.seq_len)
        + str(args.max_lr)
        + str(args.warm_up_steps)
        + str(args.initial_lr)
        + str(args.final_lr)
        + str(args.log_freq)
        + str(args.eval_freq)
        + str(args.ckpt_freq)
        + str(args.train_embedder)
        + str(args.pooling)
        + str(args.n_truncated_layers)
        + str(args.causal)
        + str(args.continuation)
        + str(args.cross_att)
        + str(args.cross_att_layers)
        + str(args.not_pool)
    )

    config["exp_name"] = (
        args.prefix + args.llm_name + sha1(name.encode("utf8")).hexdigest()[:20]
    )
    config["wandb"]["run_name"] = (
        args.prefix + args.llm_name + sha1(name.encode("utf8")).hexdigest()[:20]
    )

    if args.llm_name == "Mistral7B":
        with open(
            f'/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/{config["exp_name"]}.yaml',
            "w",
        ) as file:
            yaml.dump(config, file, sort_keys=False)
    # elif args.llm_name == "Gemma7B":
    #     with open(
    #         f'/home/hippolytepilchen/code/embed_llm/config/experiments/gemma/{config["exp_name"]}.yaml',
    #         "w",
    #     ) as file:
    #         yaml.dump(config, file, sort_keys=False)
    # elif args.llm_name == "Llama3.2-3B":
    #     with open(
    #         f'/home/hippolytepilchen/code/embed_llm/config/experiments/llama/{config["exp_name"]}.yaml',
    #         "w",
    #     ) as file:
    #         yaml.dump(config, file, sort_keys=False)
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
        "--norm_wo_embeds",
        action="store_true",
        help="Whether to normalize without word embeddings if using w_embeds",
    )
    parser.add_argument(
        "--proj_hidden_dim",
        type=int,
        default=4096,
        help="Hidden dimension of the projection MLP",
    )
    parser.add_argument(
        "--proj_n_layers",
        type=int,
        default=0,
        help="Number of layers of the projection MLP",
    )
    parser.add_argument(
        "--proj_act",
        type=str,
        default="gelu",
        help="Activation function of the projection MLP",
        choices=["id", "gelu", "relu"],
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
        "--ckpt_freq", type=int, default=500, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--prefix", type=str, default="default", help="Prefix for the experiment"
    )
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
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
        "--train_embedder",
        action="store_true",
        help="Whether to train the embedder, if True embedder_name = llm_name",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        help="Pooling method",
        choices=["mean", "eos", "latent_attention"],
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=512,
        help="Latent dimension for latent attention pooling",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of heads for latent attention pooling",
    )
    parser.add_argument(
        "-n_trunc",
        "--n_truncated_layers",
        type=int,
        default=4,
        help="Number of truncated layers to extract embedding",
    )

    parser.add_argument(
        "--causal",
        action="store_true",
        help="Whether to use a causal embedder",
    )

    parser.add_argument(
        "--continuation",
        action="store_true",
        help="Whether to train on continuation task rather than next token prediction for reconstruction",
    )
    parser.add_argument(
        "--cross_att",
        action="store_true",
        help="Whether to use cross-attention",
    )
    parser.add_argument(
        "--cross_att_layers",
        type=int,
        default=None,
        help="Number of layers to apply cross-attention",
    )

    parser.add_argument(
        "--not_pool",
        action="store_false",
        help="Whether to use pooling module",
    )

    parser.add_argument(
        "--no_norm_embeds",
        action="store_false",
        help="Whether to normalize embeddings",
    )
    
    parser.add_argument(
        "--shared_kv",
        action="store_true",
        help="Whether to share keys and values in cross-attention",
    )
    
    parser.add_argument(
        "--do_both",
        action="store_true",
        help="Whether to both cross-attended and concatenated embeddings")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

    # import os
    # import yaml
    # filenames =  ['mistral/'+file for file in os.listdir("config/experiments/mistral")]
    # for filename in filenames:
    #     if filename.endswith(".yaml"):
    #         with open("config/experiments/"+filename,'r') as file:
    #             config = yaml.safe_load(file)
    #         config['batch_size'] = 32
    #         with open("config/experiments/"+filename, 'w') as file:
    #             yaml.dump(config, file)
