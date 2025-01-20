#!/bin/bash

# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/compressed_retrieval
#SBATCH --array=1-640%16
#SBATCH --job-name=create_embeddings
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx16
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/slurm_output/slurm_embeddings_%A_%a.log 

# Command Selection

# case $SLURM_ARRAY_TASK_ID in
#   [1-320]) micromamba run -n comp_retriev python retrieval.py -n_gpus 320 -partition $((SLURM_ARRAY_TASK_ID-1)) -bs 32 ;;
#   *) echo "Could not find command for job $SLURM_ARRAY_TASK_ID" ;;
# esac

case $SLURM_ARRAY_TASK_ID in
  [1-9]|[1-9][0-9]|[1-5][0-9][0-9]|6[0-3][0-9]|640)
    micromamba run -n comp_retriev python embed_llm/retrieval/retrieval.py -n_gpus 640 -partition $((SLURM_ARRAY_TASK_ID-1)) -bs 32 ;;
  *)
    echo "Could not find command for job $SLURM_ARRAY_TASK_ID" ;;
esac