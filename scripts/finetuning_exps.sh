#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=10
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=pisco_ft_explore
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/baselines/pisco/pisco_as_us%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used



CONFIG_FILES=(
config/experiments/datasets/CP8/CP8_L3B_MLP2_M7B_trunc2_nc_fft_allstar_v6.yaml
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft4_allstar.yaml 
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft4_allstar_v5.yaml 
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft4_allstar_v3.yaml 
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft_allstar.yaml 
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft_allstar_v5.yaml 
config/experiments/ablations/few_llama/final_ft/CP8_L3B_MLP2_L8B_trunc2_nc_fft_allstar_v3.yaml
config/experiments/datasets/CP8/CP8_L3B_MLP2_M7B_trunc2_nc_fft4_allstar_pisco.yaml
config/experiments/datasets/CP8/CP8_L3B_MLP2_M7B_trunc2_nc_fft4_allstar_pisco_v2.yaml
config/experiments/datasets/CP8/CP8_L3B_MLP2_M7B_trunc2_nc_fft4_allstar_pisco_v3.yaml
config/experiments/pisco/Pisco_32memtoks_mistral_as_us.yaml
)


# Modify QA path in train data


# Get the specific config file for this array task
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

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

# Run the actual job, allocate with srun to refresh the context
srun --gpus=$N_GPU \
    torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train $CONFIG

RUN_NAME=$(basename "$CONFIG" .yaml)

echo "Starting evaluation of run $RUN_NAME"


case $RUN_NAME in

*L3B_MLP2_M7B*fft)

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 1 --n_icl_exs 0 --tmp_folder ablations/

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 1 --compressed_doc_in_icl --tmp_folder ablations/

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B  --compressed_doc_in_icl --tmp_folder ablations/
    ;;


*L3B_MLP2_L8B*fft)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B  --embed_name Llama3.2-3B  --multi_passages 1 --compressed_doc_in_icl --tmp_folder ablations/

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B  --compressed_doc_in_icl --tmp_folder ablations/ --llm_name Llama3.1-8B 
    ;;

*L3B_MLP2_M7B*fft*allstar*)

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 1 --compressed_doc_in_icl

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B  --compressed_doc_in_icl

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
    #     --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 2 --compressed_doc_in_icl

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 5 --compressed_doc_in_icl


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 1 --compressed_doc_in_icl --benchmarks DistractorHotpotQA --bs 4

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 10 --compressed_doc_in_icl --benchmarks NarrativeQA_split --bs 1


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

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
    #     --max_samples --max_seq_len 64 --multi_passages 8  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 0 \
    #      --benchmarks NarrativeQA_split  --bs 1

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
    #     --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 1 --europarl --max_seq_len 1400

    ;;

# Attention tmp folder
*L3B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B --tmp_folder ablations/


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder ablations/

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B --tmp_folder ablations/ --compressed_doc_in_icl --n_icl_exs 5


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder ablations/ --compressed_doc_in_icl
    ;;

*L3B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B --tmp_folder ablations/ --llm_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder ablations/ --llm_name Llama3.1-8B

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B --tmp_folder ablations/ --compressed_doc_in_icl --n_icl_exs 5 --llm_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder ablations/ --compressed_doc_in_icl --llm_name Llama3.1-8B
    ;;




*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json \
        --max_samples --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --n_icl_exs 5 \
         --benchmarks DistractorHotpotQA  --bs 1

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_baselines.json  \
        --max_sample  --eval_trad --run_name $RUN_NAME --llm_name mistral_7B --embed_name mistral_7B --tmp_folder baselines/pisco/ --compressed_doc_in_icl --bs 4 --europarl

    ;;

esac