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
    elif args.llm_name == "Gemma7B":
        with open(
            "/home/hippolytepilchen/code/embed_llm/config/default/default_gemma.yaml"
        ) as file:
            config = yaml.safe_load(file)
    elif args.llm_name == "Llama3.2-3B":
        with open(
            "/home/hippolytepilchen/code/embed_llm/config/default/default_llama.yaml"
        ) as file:
            config = yaml.safe_load(file)
    else:
        raise ValueError(f"{args.llm_name} not supported yet !")

    config["llm_name"] = args.llm_name
    config["w_embeds"] = args.w_embeds
    config["norm_wo_embeds"] = args.norm_wo_embeds
    config["projector"]["hidden_dim"] = args.proj_hidden_dim
    config["projector"]["n_layers"] = args.proj_n_layers
    config["projector"]["act"] = args.proj_act

    config["batch_size"] = args.batch_size
    config["max_steps"] = args.max_steps
    config["seq_len"] = args.seq_len

    config["optim"]["max_lr"] = args.max_lr
    config["optim"]["pct_start"] = args.pct_start
    config["optim"]["initial_lr"] = args.initial_lr
    config["optim"]["final_lr"] = args.final_lr

    config["log_freq"] = args.log_freq
    config["eval_freq"] = args.eval_freq
    config["ckpt_freq"] = args.ckpt_freq

    # To perform gradient accumulation
    config["num_microbatches"] = args.grad_acum_steps

    name = (
        args.llm_name
        + str(args.w_embeds)
        + str(args.norm_wo_embeds)
        + str(args.proj_hidden_dim)
        + str(args.proj_n_layers)
        + args.proj_act
        + str(args.batch_size)
        + str(args.max_steps)
        + str(args.seq_len)
        + str(args.max_lr)
        + str(args.pct_start)
        + str(args.initial_lr)
        + str(args.final_lr)
        + str(args.log_freq)
        + str(args.eval_freq)
        + str(args.ckpt_freq)
    )

    if args.prefix == "lr":
        config["exp_name"] = args.prefix + str(args.max_lr) + args.llm_name
        config["wandb"]["run_name"] = args.prefix + str(args.max_lr) + args.llm_name

    elif args.prefix == "bs":
        config["exp_name"] = args.prefix + str(args.batch_size) + args.llm_name
        config["wandb"]["run_name"] = args.prefix + str(args.batch_size) + args.llm_name

    elif args.prefix == "n_layers":
        config["exp_name"] = args.prefix + str(args.proj_n_layers) + args.llm_name
        config["wandb"]["run_name"] = (
            args.prefix + str(args.proj_n_layers) + args.llm_name
        )

    elif args.prefix == "hidden_dim":
        config["exp_name"] = args.prefix + str(args.proj_hidden_dim) + args.llm_name
        config["wandb"]["run_name"] = (
            args.prefix + str(args.proj_hidden_dim) + args.llm_name
        )

    elif args.prefix == "act":
        config["exp_name"] = args.prefix + args.proj_act + args.llm_name
        config["wandb"]["run_name"] = args.prefix + args.proj_act + args.llm_name

    elif args.prefix == "norm_wo_embeds":
        config["exp_name"] = args.prefix + str(args.norm_wo_embeds) + args.llm_name
        config["wandb"]["run_name"] = (
            args.prefix + str(args.norm_wo_embeds) + args.llm_name
        )

    else:
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
            yaml.dump(config, file)
    elif args.llm_name == "Gemma7B":
        with open(
            f'/home/hippolytepilchen/code/embed_llm/config/experiments/gemma/{config["exp_name"]}.yaml',
            "w",
        ) as file:
            yaml.dump(config, file)
    elif args.llm_name == "Llama3.2-3B":
        with open(
            f'/home/hippolytepilchen/code/embed_llm/config/experiments/llama/{config["exp_name"]}.yaml',
            "w",
        ) as file:
            yaml.dump(config, file)
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
        "--w_embeds",
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
        default=3,
        help="Number of layers of the projection MLP",
    )
    parser.add_argument(
        "--proj_act",
        type=str,
        default="id",
        help="Activation function of the projection MLP",
        choices=["id", "gelu", "relu"],
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max_lr", type=float, default=5e-5, help="Maximum learning rate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum number of steps"
    )
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.2,
        help="Percentage of steps used for the warm-up",
    )
    parser.add_argument(
        "--initial_lr", type=float, default=1e-20, help="Initial learning rate"
    )
    parser.add_argument(
        "--final_lr", type=float, default=1e-10, help="Final learning rate"
    )
    parser.add_argument("--log_freq", type=int, default=5, help="Logging frequency")
    parser.add_argument(
        "--eval_freq", type=int, default=50, help="Evaluation frequency"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)

    # import os
    # import yaml
    # filenames =  ['mistral/'+file for file in os.listdir("config/experiments/mistral")] + ['gemma/'+file for file in os.listdir("config/experiments/gemma")] + ['llama/'+file for file in os.listdir("config/experiments/llama")]
    # for filename in filenames:
    #     if filename.endswith(".yaml"):
    #         with open("config/experiments/"+filename,'r') as file:
    #             config = yaml.safe_load(file)

            # if config['llm_name'] == 'Mistral7B':
            #     if config['batch_size'] > 16:
            #         config['batch_size'] = 16
            #     with open("config/experiments/"+filename, 'w') as file:
            #         yaml.dump(config, file)

            # if config['llm_name'] == 'Llama3.2-3B':
            #     config['w_embeds'] = True
            #     with open("config/experiments/"+filename, 'w') as file:
            #         yaml.dump(config, file)

            # elif config['llm_name'] == 'Gemma7B':
            #     config['w_embeds'] = True
            #     with open("config/experiments/"+filename, 'w') as file:
            #         yaml.dump(config, file)
