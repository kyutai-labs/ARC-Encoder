import random
import argparse
import json
import time
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from dataclasses import dataclass
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from torch.distributed import get_rank, get_world_size, init_process_group, destroy_process_group
from embed_llm.training.distributed import is_torchrun
from pathlib import Path
import torch
import os
import subprocess as sp
import torch.distributed as dist

# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


instruction_prompts = [
    "\nCreate a summary of the above context:\n",
    "\nAsk questions concerning the preceding passage and provide the answers.",
    "\nExtract key takeaways and list them as bullet points.",
    "\nRewrite the passage in simpler terms for a younger audience.",
    "\nSummarize the passage:\n",
    "\nIdentify and explain any complex or technical terms used.",
    "\nProvide a counterargument or critique of the passage.",
    "\nRewrite the background in a more formal/professional tone.",
    "\nConvert the information into a persuasive argument.",
    "\nGenerate a list of keywords that summarize the main topics.",
    "\nIdentify any logical fallacies or biases in the passage.",
    "\nRewrite the passage in the style of a famous author or historical figure!",
    "\nProvide a real-world example or analogy to illustrate the main idea.",
    "\nSuggest a list of follow-up questions for further discussion:\n",
    "\nGenerate a tweet-length summary of the passage (under 280 characters).\n",
    "\nIdentify any missing information or unanswered questions in the passage.\n",
    "\nConvert the passage into a short dialogue between two characters.\n",
    "\nTurn the passage into a poem or a short piece of creative writing.",
    "\nCompare and contrast the main ideas with another topic or perspective."
]
@dataclass()
class BatchSample:
    passages: list[str]
    question: str
    tokens: list[int]

    
# def filter_samples(data_path: str):
#     with open(data_path, "r") as f:
#         for idx, line in enumerate(f):
#             data = json.loads(line)
#             if "rand" in data.keys() and float(data["rand"]) >= 0.8:
#                 continue
#             yield data['text'].strip()


def dataset_from_file(file_path, num_gen: int = None, overall_gen: int = 8):
    while True:
        with open(file_path, "r") as f:
            for idx, line in enumerate(f):
                if num_gen is None:
                    if not idx % get_world_size() == get_rank():
                        continue

                    data = json.loads(line)
                    if "rand" in data.keys() and float(data["rand"]) >= 0.8:
                        continue
                    yield data['text'].strip()
                else:
                    if not idx % overall_gen == num_gen:
                        continue
                    data = json.loads(line)
                    if "rand" in data.keys() and float(data["rand"]) >= 0.8:
                        continue
                    yield data['text'].strip()
                
def dataloader_from_file(file_path, batch_size, tokenizer, seq_sizes, num_gen: int = None, overall_gen: int = 8, adapt_seq_len: bool = False, instruct_model: bool = False):
    dataset = dataset_from_file(file_path, num_gen, overall_gen)
    batch_list = []
    for sample in dataset:

        if not adapt_seq_len:
            seq = random.choice(seq_sizes)
            
            if len(sample) < seq*2:
                continue
            
            splitted_sample = sample.split("\n")
            
            passage = []
            for i, s in enumerate(splitted_sample):
    
                if i == 0:
                    tok_seq = tokenizer.instruct_tokenizer.tokenizer.encode(s, bos = True, eos = False)
                else:
                    tok_seq = tokenizer.instruct_tokenizer.tokenizer.encode(s, bos = False, eos = False)
                
                passage += tok_seq
                if len(passage) >= seq or len(passage) >= 8192:
                    passage = passage[:8192]
                    break
        else:
            if len(sample) < 50:
                continue
            tok_seq = tokenizer.instruct_tokenizer.tokenizer.encode(sample, bos = True, eos = False)
            passage = tok_seq[:8192]
    

        passages = [tokenizer.decode(passage)]

        instruction = random.choice(instruction_prompts)
        
        if args.instruct_model:
            # chat completion request
            completion_request = ChatCompletionRequest(messages=[UserMessage(content=passages[0] + instruction)])
            # encode message
            tokens = tokenizer.encode_chat_completion(completion_request).tokens
            batch_list.append(BatchSample(passages = passages, question = instruction, tokens = tokens))
        else:
            batch_list.append(BatchSample(passages = passages, question = instruction, tokens =passage + tokenizer.instruct_tokenizer.tokenizer.encode(instruction, bos = False, eos = False)))
            
        if len(batch_list) == batch_size:
            yield batch_list
            batch_list = []
    if len(batch_list) > 0:
        yield batch_list
        
def check_tensor(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"{name} contains NaNs or Infs")
    else:
        print(f"{name} is clean")
        
def main(args):
    
    if is_torchrun():
        # Initialize default process group
        init_process_group(backend="nccl")
        
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    if not is_torchrun() or get_rank() == 0:
        if Path(args.output_path).exists():
            print("Output repo already exists")
        else:
            # Create the output directory
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
            
    if not is_torchrun() or get_rank() == 0:
        if args.num_gen is None:
            for rank in range(1, get_world_size()):
                (Path(args.output_path) / args.output_file.replace('.jsonl',str(rank) + '.jsonl')).touch()
        else:
            (Path(args.output_path) / args.output_file.replace('.jsonl',str(args.num_gen) + '.jsonl')).touch()
            

    if args.n_samples is not None:
        args.max_steps = args.n_samples // args.batch_size
        

    mistral_tokenizer = MistralTokenizer.from_file(args.tokenizer_path)


        
    model = Transformer.from_folder(args.model_folder_path, max_batch_size=args.batch_size, device = f"cuda:{local_rank}")
    data_loader = dataloader_from_file(args.data_path, args.batch_size, mistral_tokenizer, 
                                       args.seq_sizes, num_gen = args.num_gen, overall_gen = args.overall_gen, adapt_seq_len = args.adapt_seq_len, instruct_model = args.instruct_model)
    data_buffer = []
    n_data = torch.tensor([0], device=f"cuda:{local_rank}")
    n_toks = torch.tensor([0], device=f"cuda:{local_rank}")
    max_token_size = 0
    start_time = time.time()
    for step in range(args.max_steps):
 
        batch = next(data_loader)
        tokens = [sample.tokens for sample in batch]

        

        try:
            out_tokens, _ = generate(tokens, model, max_tokens=args.max_gen_toks, temperature=args.temp, eos_id=None)  
        except Exception as e:
            print("Error during generation", step)
            print('Memory:', get_gpu_memory())
            print('Max memory:', torch.cuda.max_memory_allocated())
            print('Exception:', e)
            continue
        
        torch.cuda.empty_cache() 

        max_token_size = max(max_token_size, max([len(l) for l in tokens]))
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
            if is_torchrun():
                buffer_data = torch.tensor([n_data.item()], dtype=torch.int32, device=f"cuda:{local_rank}")
                buffer_toks = torch.tensor([n_toks.item()], dtype=torch.int32, device=f"cuda:{local_rank}")
                dist.all_reduce(buffer_data, op=dist.ReduceOp.SUM)
                dist.all_reduce(buffer_toks, op=dist.ReduceOp.SUM)
                if get_rank() == 0:
                    print(f"Step {step} took {time.time() - start_time} seconds", "Data processed:", buffer_data[0].item(), "Tokens generated:", buffer_toks[0].item(), "Max token size:", max_token_size)
            else:
                print(f"Step {step} took {time.time() - start_time} seconds", "Data processed:", n_data.item(), "Tokens generated:", n_toks.item(), "Max token size:", max_token_size)
        start_time = time.time()
        if step%args.freq_load_data == 0:
            print('Data buffer size:', data_buffer[-1])
            if args.num_gen is None:
                with open(Path(args.output_path) / args.output_file.replace('.jsonl',str(get_rank()) + '.jsonl'), "a") as f:
                    for sample in data_buffer:
                        json.dump(sample, f)
                        f.write("\n")
                data_buffer = []
            else:
                with open(Path(args.output_path) / args.output_file.replace('.jsonl',str(args.num_gen) + '.jsonl'), "a") as f:
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
    # /lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B_instruct/tokenizer.model.v3
    parser.add_argument(
        "--model_folder_path",
        type=str,
        default='/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/',
    )
    # /lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B_instruct
    parser.add_argument(
        "-bs","--batch_size",
        type=int,
        default=32,
    )
    
    parser.add_argument(
        '--freq_load_data',
        default=5,
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
        default=0.8,
    )
    
    parser.add_argument(
        "--num_gen",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overall_gen",
        type=int,
        default=8,
    )
    
    parser.add_argument(
        "--adapt_seq_len",
        action='store_true',
    )
    
    parser.add_argument(
        "--instruct_model",
        action='store_true',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)