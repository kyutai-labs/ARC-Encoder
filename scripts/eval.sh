#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-5 #42
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx20
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used

# Get the configuration file for this job
RUN_NAMES=(
nopref_pretrain_both_trained_rec_singpassage_0f6f2a1a
nopref_pretrain_both_trained_rec_multipassage_0f6f2a1a 
nopref_pretrain_both_trained_cont_singpassage_17c38ada
nopref_pretrain_nollm_trained_cont_singpassage_5darr64
nopref_pretrain_nollm_trained_rec_multipassage_5darr64
nopref_pretrain_nollm_trained_rec_singpassage_5darr64
nopref_pretrain_llm_trained_rec_singpassage_054f63f8 
nopref_pretrain_llm_trained_cont_singpassage_5daaa6bc
nopref_pretrain_pool_trained_cont_singpassage_5daaa6bc
nopref_pretrain_pool_trained_rec_singpassage_054f63f8
)


# Get the specific config file for this array task
RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}

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


echo "Starting evaluation of run $RUN_NAME"

srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/eval_nopref_pretrained.json \
    --n_passages 500 --max_seq_len 128
   
echo "Finished at: $(date)"

# End of file



