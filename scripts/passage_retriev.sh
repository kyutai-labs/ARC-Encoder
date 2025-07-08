#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-19%8
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=retrieval_pisco
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out
#SBATCH --exclude=par2dc5-ai-prd-cl02s01dgx26

# Get number of GPUs allocated to this task, -z checks if CUDA_VISIBLE_DEVICES is empty
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # If CUDA_VISIBLE_DEVICES is not set, try to get from SLURM
    # Replace , with newline and count lines
    N_GPUS=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\n' | wc -l)
else
    # Count GPUs from CUDA_VISIBLE_DEVICES
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Starting job array ${SLURM_ARRAY_TASK_ID} "
echo "Using $N_GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"


case $SLURM_ARRAY_TASK_ID in

*)

    srun --gpus=$N_GPU  \
        micromamba run -p comp_retriev python embed_llm/retrieval/passage_retrieval.py -outpath /lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/true_pisco/pisco_train_${SLURM_ARRAY_TASK_ID}.jsonl \
        -data_path /lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/pisco_subdivision/pisco_train_${SLURM_ARRAY_TASK_ID}.jsonl \

    ;;

esac