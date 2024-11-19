#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-13%4
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID))

# Get the configuration file for this job
CONFIG_FILES=(
config/experiments/llama/hidden_dim2048Llama3.2-3B.yaml
config/experiments/llama/lr0.0005Llama3.2-3B.yaml
config/experiments/llama/n_layers2Llama3.2-3B.yaml
config/experiments/llama/no_embed_lr5e-5Llama3.2-3Bb00319493d3f10e95678.yaml
config/experiments/llama/hidden_dim3072Llama3.2-3B.yaml
config/experiments/llama/n_layers1Llama3.2-3B.yaml
config/experiments/llama/n_layers3Llama3.2-3B.yaml
config/experiments/gemma/hidden_dim2048Gemma7B.yaml
config/experiments/gemma/hidden_dim3072Gemma7B.yaml
config/experiments/gemma/lr0.0005Gemma7B.yaml
config/experiments/gemma/n_layers1Gemma7B.yaml
config/experiments/gemma/n_layers2Gemma7B.yaml
config/experiments/gemma/n_layers3Gemma7B.yaml
config/experiments/gemma/no_embed_lr5e-5Gemma7B7dd9d4cecfbff5795710.yaml

)
# config/experiments/mistral/lr0.0005Mistral7B.yaml
# config/experiments/mistral/lr5e-05Mistral7B.yaml
# config/experiments/mistral/n_layers0Mistral7B.yaml
# config/experiments/mistral/n_layers1Mistral7B.yaml
# config/experiments/mistral/n_layers2Mistral7B.yaml
# config/experiments/mistral/no_embed_bs16_lr5e-5Mistral7B88d0b42410aa4ec12025.yaml

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
    micromamba run -n llm_embed torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train $CONFIG

echo "Finished at: $(date)"

# End of file