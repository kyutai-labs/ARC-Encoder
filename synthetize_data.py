import random
import argparse
import json
import time
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from dataclasses import dataclass
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch.distributed import get_rank, get_world_size, init_process_group, destroy_process_group
from pathlib import Path
import torch
import os
import torch.distributed as dist

instruction_prompts = [
    "\nCreate a summary of the above context:\n",
    "\nAsk questions concerning the preceding passage and provide the answers.\n",
    "\nExtract key takeaways and list them as bullet points.\n",
    "\nRewrite the passage in simpler terms for a younger audience.\n",
    "\nSummarize the passage:\n",
    "\nIdentify and explain any complex or technical terms used.\n",
    "\nProvide a counterargument or critique of the passage.\n",
    "\nRewrite the background in a more formal/professional tone.\n",
    "\nConvert the information into a persuasive argument.\n",
    "\nGenerate a list of keywords that summarize the main topics.\n",
    "\nIdentify any logical fallacies or biases in the passage.\n",
    "\nRewrite the passage in the style of a famous author or historical figure.\n",
    "\nProvide a real-world example or analogy to illustrate the main idea.\n",
    "\nSuggest a list of follow-up questions for further discussion.\n",
    "\nGenerate a tweet-length summary of the passage (under 280 characters).\n",
    "\nIdentify any missing information or unanswered questions in the passage.\n",
    "\nConvert the passage into a short dialogue between two characters.\n",
    "\nTurn the passage into a poem or a short piece of creative writing.\n",
    "\nCompare and contrast the main ideas with another topic or perspective.\n"
]
@dataclass()
class BatchSample:
    passages: list[str]
    question: str
    tokens: list[int]

    
def dataset_from_file(file_path):
    while True:
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                if not idx % get_world_size() == get_rank():

                    data = json.loads(line)
                    if "rand" in data.keys() and float(data["rand"]) >= 0.8:
                        continue
                    yield data['text'].strip()
                
def dataloader_from_file(file_path, batch_size, tokenizer, seq_sizes):
    dataset = dataset_from_file(file_path)
    batch_list = []
    for sample in dataset:

        seq = random.choice(seq_sizes)
        
        if len(sample) < seq*2:
            continue
        
        splitted_sample = sample.split("\n")
        
        passage = []
        for i, s in enumerate(splitted_sample):
  
            if i == 0:
                tok_seq = tokenizer.encode(s, bos = True, eos = False)
            else:
                tok_seq = tokenizer.encode(s, bos = False, eos = False)
            
            passage += tok_seq
            if len(passage) >= seq:
                break
    

        passages = [tokenizer.decode(passage)]

        instruction = random.choice(instruction_prompts)
        
        batch_list.append(BatchSample(passages = passages, question = instruction, tokens =passage + tokenizer.encode(instruction, bos = False, eos = False)))
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []
    if len(batch_list) > 0:
        yield batch_list
        
def main(args):
    
    # Initialize default process group
    init_process_group(backend="nccl")
    # Get local rank for this process
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    if get_rank() == 0:
        if Path(args.output_path).exists():
            print("Output repo already exists")
        else:
            # Create the output directory
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
            
    if get_rank() == 0:
        for rank in range(1, get_world_size()):
            (Path(args.output_path) / args.output_file.replace('.jsonl',str(rank) + '.jsonl')).touch()
            

    if args.n_samples is not None:
        args.max_steps = args.n_samples // args.batch_size
        
    mistral_tokenizer = MistralTokenizer.from_file(args.tokenizer_path)
    model = Transformer.from_folder(args.model_folder_path, max_batch_size=args.batch_size)
    data_loader = dataloader_from_file(args.data_path, args.batch_size, mistral_tokenizer.instruct_tokenizer.tokenizer, args.seq_sizes)
    data_buffer = []
    n_data = torch.tensor([0], device="cuda")
    n_toks = torch.tensor([0], device="cuda")
    start_time = time.time()
    for step in range(args.max_steps):

        batch = next(data_loader)
        tokens = [sample.tokens for sample in batch]
        out_tokens, _ = generate(tokens, model, max_tokens=args.max_gen_toks, temperature=args.temp, eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)        

        truncated_list = []
        for j in range(args.batch_size):
            truncated_list.append([])
            for tok in out_tokens[j]:
                if tok == mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id:
                    break
                truncated_list[j].append(tok)

        for j in range(args.batch_size):
            result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(truncated_list[j])
            data_buffer.append({"passages": batch[j].passages, "question": batch[j].question, "answer": result})
            
        n_data += torch.tensor(args.batch_size).item()
        n_toks += torch.tensor(sum([len(l) for l in truncated_list])).item()
        

        if step % 100 == 0:
            buffer_data = torch.tensor([n_data.item()], dtype=torch.int32, device=f"cuda:{local_rank}")
            buffer_toks = torch.tensor([n_toks.item()], dtype=torch.int32, device=f"cuda:{local_rank}")
            dist.all_reduce(buffer_data, op=dist.ReduceOp.SUM)
            dist.all_reduce(buffer_toks, op=dist.ReduceOp.SUM)

            if get_rank() == 0:
                print(f"Step {step} took {time.time() - start_time} seconds", "Data processed:", buffer_data[0].item(), "Tokens generated:", buffer_toks[0].item())
            
        start_time = time.time()
        if step%args.freq_load_data == 0:
            with open(Path(args.output_path) / args.output_file.replace('.jsonl',str(get_rank()) + '.jsonl'), "a") as f:
                for sample in data_buffer:
                    json.dump(sample, f)
                    f.write("\n")
            data_buffer = []
    destroy_process_group()

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/tokenizer.model.v3',
    )
    parser.add_argument(
        "--model_folder_path",
        type=str,
        default='/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/',
    )
    parser.add_argument(
        "-bs","--batch_size",
        type=int,
        default=32,
    )
    
    parser.add_argument(
        '--freq_load_data',
        type=int,
        default=50,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default='/lustre/scwpod02/client/kyutai-interns/datasets/crawl_2/train_en_00_of_18.jsonl',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/crawl/synth_data/',
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='synth_data.jsonl',
    )
    
    parser.add_argument('--seq_sizes', type=int, nargs='+', 
                        help='A list of integers (space separated)', default=[128, 256, 512, 1024, 2048])
    
    
    parser.add_argument(
        "--max_gen_toks",
        type=int,
        default=256,
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=40000,
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000000,
    )
    
    parser.add_argument(
        "--temp",
        type=float,
        default=.8,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)