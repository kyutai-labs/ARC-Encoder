" This file create a config dictionary to launch training with the specified hyperparameters. "
import argparse
import json


def main(args):
    header_params = {
        "adapt_seq_len": args.adapt_seq_len,
        "data_types": args.data_type,
        "eval_data_common_path_prefix": args.eval_data_common_path_prefix,
        "train_data_common_path_prefix": args.train_data_common_path_prefix,
        "shuffle": args.shuffle,
    }

    if args.overwrite:
        with open(args.folder_path + args.output_file, "w") as f:
            f.write(json.dumps(header_params) + "\n")

    if args.train_files is not None:
        assert args.train_weights is None or  len(args.train_files.split(",")) == len(args.train_weights.split(","))
        if args.train_weights is None:
            args.train_weights = "1," * len(args.train_files.split(","))
        with open(args.folder_path + args.output_file, "a") as f:
            for file, weight in zip(
                args.train_files.split(","), args.train_weights.split(",")
            ):
                f.write(
                    json.dumps({"train_data": {"path": file, "weight": weight}}) + "\n"
                )

    if args.eval_files is not None:
        assert args.eval_weights is None or len(args.eval_files.split(",")) == len(args.eval_weights.split(","))
        if args.eval_weights is None:
            args.eval_weights = "1," * len(args.eval_files.split(","))
        with open(args.folder_path + args.output_file, "a") as f:
            for file, weight in zip(
                args.eval_files.split(","), args.eval_weights.split(",")
            ):
                f.write(
                    json.dumps({"eval_data": {"path": file, "weight": weight}}) + "\n"
                )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        default="instruct",
        help="Type of data used",
    )

    parser.add_argument(
        "--adapt_seq_len",
        action="store_true",
        help="Whether to adapt sequence length to the one of the passage (until max_seq_len)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to not use data params inside config file",
    )

    parser.add_argument(
        "--folder_path",
        type=str,
        default="/home/hippolytepilchen/code/embed_llm/config/experiments/data_configs/",
        help="Path to data params",
    )

    parser.add_argument(
        "-eval_dcpp",
        "--eval_data_common_path_prefix",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA/",
    )

    parser.add_argument(
        "-train_dcpp",
        "--train_data_common_path_prefix",
        type=str,
        default="/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle data",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save data params",
    )

    parser.add_argument("--train_files", type=str, default=None)

    parser.add_argument("--eval_files", type=str, default=None)

    parser.add_argument("--train_weights", type=str, default=None)

    parser.add_argument("--eval_weights", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
