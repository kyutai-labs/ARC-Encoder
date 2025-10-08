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
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx25
# Set up environment

export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
ARC4_Encoder_llama_new_full5shot_test
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
            uv run python -m embed_llm.generation.eval_context_comp   --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
      --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 0  --n_icl_exs 5

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
        --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B   --embed_name Llama3.2-3B   --llm_number 0  

    # srun --gpus=$N_GPU  \
    #         uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
    #    --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME  --embed_name Llama3.2-3B  --llm_number 1  --n_icl_exs 5

    # srun --gpus=$N_GPU  \
    #         uv run python -m embed_llm.generation.eval_context_comp   --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
    #    --eval_trad --run_name $RUN_NAME --embed_name Llama3.2-3B  --llm_number 1  

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp   --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
     --max_seq_len 256 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 0  --n_icl_exs 5 --benchmarks CNN --bs 4


    # srun --gpus=$N_GPU  \
    #         uv run python -m embed_llm.generation.eval_context_comp   --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
    #   --max_seq_len 256 --multi_passages 1   --run_name $RUN_NAME   --embed_name Llama3.2-3B  --llm_number 1  --n_icl_exs 5 --benchmarks CNN --bs 4

    srun --gpus=$N_GPU  \
            uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
      --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 0  --n_icl_exs 5 --benchmarks DistractorHotpotQA --bs 1

    # srun --gpus=$N_GPU  \
    #         uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json \
    #    --max_seq_len 64 --multi_passages 1   --run_name $RUN_NAME  --embed_name Llama3.2-3B  --llm_number 1  --n_icl_exs 5 --benchmarks DistractorHotpotQA --bs 1
    ;;

# *)
#     srun --gpus=$N_GPU  \
#             uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
#                    --max_seq_len 64 --run_name $RUN_NAME --embed_name Llama3.2-3B  --multi_passages 1 --n_icl_exs 5 

#     srun --gpus=$N_GPU  \
#             uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
#                    --max_seq_len 64 --run_name $RUN_NAME --embed_name Llama3.2-3B  --multi_passages 1 --n_icl_exs 5 --benchmarks CNN


#     srun --gpus=$N_GPU  \
#             uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
#         --run_name $RUN_NAME --eval_trad  --embed_name Llama3.2-3B --tmp_folder hp_v2/ 


#     # srun --gpus=$N_GPU  \
#     #         uv run python -m embed_llm.generation.eval_context_comp  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/new_eval.json  \
#     #     --max_seq_len 64 --run_name $RUN_NAME  --embed_name Llama3.2-3B  --multi_passages 1 --benchmarks DistractorHotpotQA --bs 4 --n_icl_exs 5 --tmp_folder hp_v2/ 


#     ;;

esac
