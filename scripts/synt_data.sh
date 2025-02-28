#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-7
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx30
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=synt_data
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/synt_data/embed_llm_%A_%a.out

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


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

    srun --gpus=$N_GPU \
        micromamba run -n  synt_data torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m synthetize_data --output_path /lustre/scwpod02/client/kyutai-interns/hippop/processed_data/crawl/synth_data_p/ \
        --num_gen $SLURM_ARRAY_TASK_ID

    
    ;;

esac

