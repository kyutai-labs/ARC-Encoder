#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=3-7%3
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/baselines
#SBATCH --job-name=pisco_eval
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/baselines/pisco/pisco_eval_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
Pisco_8memtoks_mistral
Pisco_8memtoks_llama
Pisco_32memtoks_mistral_v2
Pisco_32memtoks_llama_v2
Pisco_16memtoks_mistral_v2
Pisco_16memtoks_llama_v2
Pisco_8memtoks_mistral_v2
Pisco_8memtoks_llama_v2
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

icae*llama)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/icae/ --compressed_doc_in_icl --n_icl_exs 5  --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/icae/ --compressed_doc_in_icl --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/icae/ --compressed_doc_in_icl --n_icl_exs 5 \
         --benchmarks DistractorHotpotQA  --bs 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/icae/ --compressed_doc_in_icl --bs 4 --europarl

    ;;

icae*mistral)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/icae/ --compressed_doc_in_icl --n_icl_exs 5  --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/icae/ --compressed_doc_in_icl --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/icae/ --compressed_doc_in_icl --n_icl_exs 5 \
         --benchmarks DistractorHotpotQA  --bs 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/icae/ --compressed_doc_in_icl --bs 4 --europarl

    ;;


Pisco*llama*)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 5  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 4 --benchmarks NQ

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 5  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 4 --benchmarks TRIVIAQA

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5 \
         --benchmarks DistractorHotpotQA  --bs 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B --embed_name Llama3.1-8B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 4 --europarl

    ;;

Pisco*mistral*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 5  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 4 --benchmarks NQ

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 5  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5  --bs 4 --benchmarks TRIVIAQA

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5 \
         --benchmarks DistractorHotpotQA  --bs 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 1 --europarl

    ;;

esac
