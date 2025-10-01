#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0
#SBATCH --nodes=1        # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/clean_hp
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/test_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx28



# Set up environment

export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
default_finetuning
)





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

echo "Starting job array ${SLURM_ARRAY_TASK_ID} "
echo "Using $N_GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"
echo "Run name: $RUN_NAME"


case $RUN_NAME in
*)
    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/tests.json  \
                   --max_seq_len 64 --run_name default_finetuning --embed_name Llama3.2-3B  --multi_passages 1 --n_icl_exs 5 

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/tests.json  \
        --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder hp_v2/ 


    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/tests.json  \
        --max_seq_len 64 --run_name $RUN_NAME  --embed_name Llama3.2-3B  --multi_passages 1 --benchmarks DistractorHotpotQA --bs 4 --n_icl_exs 5 --tmp_folder hp_v2/ 


    ;;

esac
