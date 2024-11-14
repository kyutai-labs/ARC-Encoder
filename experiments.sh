#!/bin/bash

# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hippolytepilchen/code/compressed_retrieval
#SBATCH --array=1-4
#SBATCH --job-name=incontext_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/slurm_output/slurm_output_%A_%a.log 

# Define hyperparameter values to test
learning_rates=(0.001 0.01 0.1)
batch_sizes=(32 64 128)

# Train model for each combination of hyperparameters
for lr in "${learning_rates[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    python train.py --learning-rate $lr --batch-size $bs
  done
done
