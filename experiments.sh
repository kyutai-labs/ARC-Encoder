#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-7%4
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
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/finetuned_mean_5_notshared_doboth_3B_MLP_Mistral7Bdf219d11eff6d12dc259.yaml  
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/pretrained_5_notshared_doboth_3B_MLP_Mistral7B4d45dee02ab4d1cfe967.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/finetuned_meanpool_crossatt_24_notshared_Mistral7Ba9fe3e5faaea0ca962d8.yaml  
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/pretrained_pooled_crossatt_24_notshared_Mistral7B61c9bdf49d2f4ae7984c.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/finetuned_mean_5_notshared_doboth_no_MLP_Mistral7Bd7789bd189ddc55d7a4e.yaml  
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/pretrained_5_notshared_doboth_no_MLP_Mistral7Bb43dd2408bcf7650e718.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/finetuned_meanpool_crossatt_8_notshared_Mistral7B694c0c8d13a121ed8718.yaml   
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/pretrained_pooled_crossatt_8_notshared_Mistral7B27dd451cc0c8f2e72df3.yaml                               
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