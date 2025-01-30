#!/bin/bash

# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --array=1-640%64
#SBATCH --job-name=create_embeddings
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx14,par2dc5-ai-prd-cl02s01dgx02,par2dc5-ai-prd-cl02s03dgx30,par2dc5-ai-prd-cl02s04dgx20,par2dc5-ai-prd-cl02s01dgx01,par2dc5-ai-prd-cl02s04dgx24,par2dc5-ai-prd-cl02s01dgx09,par2dc5-ai-prd-cl02s04dgx11
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/nvembed/slurm_embeddings_%A_%a.log 

# Command Selection



case $SLURM_ARRAY_TASK_ID in
  [1-9]|[1-9][0-9]|[1-5][0-9][0-9]|6[0-3][0-9]|640)
    micromamba run -n llm_embed  python embed_llm/retrieval/embeddings.py -n_gpus 640 -partition $((SLURM_ARRAY_TASK_ID-1)) -bs 32 ;;
  *)
    echo "Could not find command for job $SLURM_ARRAY_TASK_ID" ;;
esac