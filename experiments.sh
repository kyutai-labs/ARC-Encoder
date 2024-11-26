#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-1
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID))

# Get the configuration file for this job
CONFIG_FILES=(
config/experiments/mistral/eos_Mistral7B_not_causal.yaml
config/experiments/mistral/eos_Mistral7B_causal.yaml 
)
# config/experiments/mistral/2n_trunc_eos_Mistral7B845c88e6bde711a7a36e.yaml   
# config/experiments/mistral/3n_trunc_eos_Mistral7B845c88e6bde711a7a36e.yaml    
# config/experiments/mistral/5n_trunc_eos_Mistral7B845c88e6bde711a7a36e.yaml  
# config/experiments/mistral/no_warmup_Mistral7B.yaml
# config/experiments/mistral/latt_Mistral7B3923ee72a684f4be9a25.yaml    
# config/experiments/mistral/lr_1e-4_lattMistral7B6927e5b48173825489db.yaml 




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