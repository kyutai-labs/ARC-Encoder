#!/bin/bash
# SBATCH options
#SBATCH --partition=
#SBATCH --array=
#SBATCH --nodes=1        # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir= # TO SET
#SBATCH --job-name=eval_models
#SBATCH --output=out_dire/training/test_%A_%a.out

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used



CONFIG_FILES=(
Path to your config files here
)


# Modify QA path in train data


# Get the specific config file for this array task
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

# Get number of GPUs allocated to this task, -z checks if CUDA_VISIBLE_DEVICES is empty
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If CUDA_VISIBLE_DEVICES is not set, try to get from SLURM
    # Replace , with newline and count lines
    N_GPUS=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
else
    # Count GPUs from CUDA_VISIBLE_DEVICES
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Starting job array ${SLURM_ARRAY_TASK_ID} with config ${CONFIG}"
echo "Using $N_GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"

# Run the actual job, allocate with srun to refresh the context
srun --gpus=$N_GPU \
    uv run torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train $CONFIG

RUN_NAME=$(basename "$CONFIG" .yaml)

echo "Starting evaluation of run $RUN_NAME"


# Set a rule depending on the name see eval.sh
case $RUN_NAME in

# Default case if multi-decoder ARC-Encoder
*) 
    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp   --out_file out_dire/eval/eval_results.json \
      --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B   Llama3.2-3B  --llm_number 0  --n_icl_exs 5 

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp --out_file out_dire/eval/eval_results.json  \
        --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B    Llama3.2-3B   --llm_number 0  

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file out_dire/eval/eval_results.json \
       --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME   Llama3.2-3B  --llm_number 1  --n_icl_exs 5

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp   --out_file out_dire/eval/eval_results.json  \
       --eval_trad --run_name $RUN_NAME  Llama3.2-3B  --llm_number 1  

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp   --out_file out_dire/eval/eval_results.json \
     --max_seq_len 256 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B   Llama3.2-3B  --llm_number 0  --n_icl_exs 5 --benchmarks CNN --bs 4


    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp   --out_file out_dire/eval/eval_results.json \
      --max_seq_len 256 --multi_passages 1   --run_name $RUN_NAME    Llama3.2-3B  --llm_number 1  --n_icl_exs 5 --benchmarks CNN --bs 4

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file out_dire/eval/eval_results.json \
      --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B   Llama3.2-3B  --llm_number 0  --n_icl_exs 5 --benchmarks DistractorHotpotQA --bs 1

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file out_dire/eval/eval_results.json \
       --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME   Llama3.2-3B  --llm_number 1  --n_icl_exs 5 --benchmarks DistractorHotpotQA --bs 1
    ;;
esac