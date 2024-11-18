#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-4%5
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
config/experiments/mistral/no_embeddeb013c0f34ae3c9ca92.yaml  
config/experiments/llama/no_embed7e1d20b548bb45f6dd17.yaml 
config/experiments/mistral/mlp_proj2a8bef70d3a4b650c584.yaml  
config/experiments/mistral/mlp_proj8d4c6c3216b80d72a015.yaml  
config/experiments/mistral/mlp_proj5a1ebec45db30455e197.yaml 
config/experiments/llama/act0a84d43f52dc15053eff.yaml  
config/experiments/mistral/act1a73c20b533bed0e3328.yaml  
config/experiments/mistral/batch_size9c48adf4c78453facc09.yaml  
config/experiments/mistral/learning_rate_7a72633c687dea3b79f6.yaml   
config/experiments/mistral/mlp_projcdbd4db0009898090bd4.yaml  
config/experiments/mistral/act212c9223233d06ebb076.yaml  
config/experiments/mistral/batch_sizec370414b4dc89bdd855c.yaml  
config/experiments/mistral/learning_rate_8ab4d9eef1a27e720cac.yaml  
config/experiments/mistral/mlp_proj7a3a4ade9c98ec81b556.yaml  
config/experiments/mistral/mlp_proje322e1b0b15d757d8f48.yaml  
config/experiments/mistral/act854e742415ddfea70519.yaml  
config/experiments/mistral/mlp_proj212c9223233d06ebb076.yaml  
config/experiments/mistral/mlp_proj7c3574ce8bc21fa7ece4.yaml  
config/experiments/mistral/mlp_proj_hiddim_21f142e308678af2a9b6.yaml  
config/experiments/mistral/batch_size98a7961d71e620a4afbf.yaml  
config/experiments/mistral/learning_rate_2abf8487274cae61b5f0.yaml  
config/experiments/llama/batch_size0c5abfcc3777aae21332.yaml  
config/experiments/llama/learning_rate_3f8b4cf6fb68524c8f2f.yaml  
config/experiments/llama/mlp_proj0a84d43f52dc15053eff.yaml  
config/experiments/llama/mlp_proje63e87fa7bafa6aee405.yaml  
config/experiments/llama/actb12d9d87a02a73f57670.yaml  
config/experiments/llama/batch_sized1a86c71baa3368a7a6c.yaml  
config/experiments/llama/learning_rate_68991676a1697c06cd15.yaml  
config/experiments/llama/mlp_projb72484b6cb1f35b762b4.yaml  
config/experiments/llama/mlp_proj_hiddim29673113f782cb117145.yaml  
config/experiments/llama/acte33fc49cf9005591617a.yaml  
config/experiments/llama/batch_sizef4ec4220946f576affa7.yaml  
config/experiments/llama/learning_rate_ba6baccde12dbf06d347.yaml  
config/experiments/llama/mlp_proje5af7393caa606740ed6.yaml  
)



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