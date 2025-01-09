#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=19-20
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s02dgx20 #par2dc5-ai-prd-cl02s03dgx30,par2dc5-ai-prd-cl02s04dgx11 #,par2dc5-ai-prd-cl02s04dgx22,par2dc5-ai-prd-cl02s04dgx06,par2dc5-ai-prd-cl02s03dgx21,par2dc5-ai-prd-cl02s04dgx24
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used

# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_no_trained_cont_singpassage_5daaa6bc.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_no_trained_rec_multipassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_no_trained_rec_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_02_singpassage_0f6f2a1a.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_02_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_cont_singpassage_5daaa6bc.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_rec_multipassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_pool_trained_cont_singpassage_5daaa6bc.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_pool_trained_rec_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_1cont_0.2textcont_singpassage_17c38ada.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_1cont_0.5textcont_singpassage_17c38ada.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_cont_singpassage_17c38ada.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_rec_multipassage_0f6f2a1a.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_rec_singpassage_0f6f2a1a.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_rec_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_07_singpassage_0f6f2a1a.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_07_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_both_trained_05_singpassage_0f6f2a1a.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/nopref_pretrain_llm_trained_05_singpassage_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/LT_FN_TrueMEAN_1_MLP_Latt_True_CA_2_CAL_every_True_DB.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/LT_FN_TrueMEAN_1_MLP_RLatt_True_CA_2_CAL_every_True_DB.yaml
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

