#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-1
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx25
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID)) # Take care if already used

# Get the configuration file for this job
CONFIG_FILES=(
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_1_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/128_SL_FN_Truemean_0_MLP_8_TRUNC_True_CA_16_CAL_False_SKV_False_DB_old_gate_more_params.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_3_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_3_MLP_True_CA_2_CAL_every_True_DBMLM.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_3_MLP_8_TRUNC_True_CA_16_CAL_atend_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_3_MLP_True_CA_16_CAL_atend_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_3_MLP_8_TRUNC_True_CA_2_CAL_every_True_DBCONT.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truelatent_attention_3_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_0_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_3_MLP_True_CA_2_CAL_every_True_DBCONT.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_3_MLP_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truereversed_latent_attention_3_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_1_MLP_RLatt_True_CA_2_CAL_every_True_DB.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_1_MLP_Latt_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_False_0_MLP_True_CA_2_CAL_every_True_DB.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_3_MLP_8_TRUNC_True_CA_2_CAL_every_True_DBMLM.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/LT_FN_Truemean_1_MLP_8_TRUNC_True_CA_2_CAL_every_True_DB.yaml
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

