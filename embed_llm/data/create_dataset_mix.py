from embed_llm.data.utils import templates_for_qa
import json
import argparse
import random
import os



def main(args):
    mix_dataset = []
    
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as f:
            f.write("")
    
    with open(args.output_file, "r") as f:
        for line in f:
            mix_dataset.append(json.loads(line))
            
            
    
    to_add = []
    with open(args.to_add_file, "r") as f:
        for i, line in enumerate(f):
            
            if i < args.start_at_n:
                continue
            
            sample = json.loads(line)
            answer = sample['answer']
            if isinstance(sample['answer'], list):
                answer = '\n'.join(sample['answer'])
            if 'No Answer Present'.lower() in answer.lower() and args.remove_no_answer :
                continue
        
            if 'I don\'t know.'.lower() in answer.lower() and args.remove_no_answer:
                continue
            
            if  len(answer) == 0 and args.remove_no_answer:
                continue
            
            if args.no_answer_only and not ('I don\'t know.'.lower() in answer.lower() or 'No Answer Present'.lower() in answer.lower()):
                continue
             
            if args.add_query_template:
                sample["question"] = random.choice(templates_for_qa).format(question=sample['question'])
            
            to_add.append(sample)
            
            if len(to_add) == args.n_sample_to_add:
                break
            
    print('Adding', len(to_add), 'samples')
    for sample in to_add:
        mix_dataset.append(sample)
        
    if args.shuffle:
        random.shuffle(mix_dataset)
    
    with open(args.output_file, "w") as f:
        for sample in mix_dataset:
            f.write(json.dumps(sample) + "\n")


def arg_parser():
    parser = argparse.ArgumentParser()
   
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

    parser.add_argument(
        "--remove_no_answer",
        action="store_true",
        help="Whether to remove samples with no answer",
    )
    parser.add_argument("--no_answer_only", action="store_true")
    parser.add_argument("--to_add_file", type=str, default=None)

    parser.add_argument("--n_sample_to_add", type=int, default=None)

    parser.add_argument("--add_query_template", action="store_true")
    parser.add_argument("--start_at_n", default = 0, type = int)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
