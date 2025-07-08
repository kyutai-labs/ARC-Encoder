#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=2
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/mix_decoder_training
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
multi_decoder_L8_MLP_nc_ft
multi_decoder_L8_nc_ft
multi_decoder_L3_MLP_nc_ft
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


case $RUN_NAME in

*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 501 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 1 
    ;;

*L8*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.1-8B  --llm_number 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B   --embed_name Llama3.1-8B   --llm_number 1 --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.1-8B  --llm_number 2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --embed_name Llama3.1-8B  --llm_number 2 --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2138  --llm_number 1 --compressed_doc_in_icl --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B   --embed_name Llama3.1-8B   --llm_number 1 --compressed_doc_in_icl --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.1-8B  --llm_number 2 --compressed_doc_in_icl --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --embed_name Llama3.1-8B  --llm_number 2 --compressed_doc_in_icl --new_template
    ;;

*)

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B   --embed_name Llama3.2-3B   --llm_number 1 --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B  --llm_number 2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --embed_name Llama3.2-3B  --llm_number 2 --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --llm_number 1 --compressed_doc_in_icl --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B   --embed_name Llama3.2-3B   --llm_number 1 --compressed_doc_in_icl --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B  --llm_number 2 --compressed_doc_in_icl --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --embed_name Llama3.2-3B  --llm_number 2 --compressed_doc_in_icl --new_template
    ;;


esac

