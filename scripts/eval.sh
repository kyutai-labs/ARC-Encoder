#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-9
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out
##SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx19,par2dc5-ai-prd-cl02s04dgx20
# Set up environment

export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
CP8_L3B_MLP2_L8B_trunc10_nc_ft
CP8_L3B_MLP2_L8B_notcausal_ft
CP8_L3B_MLP2_L8B_last_ft
CP8_L3B_MLP2_L8B_fusion_ft
CP8_L3B_MLP2_L8B_trunc10_nc_fft
CP8_L3B_MLP2_L8B_notcausal_fft
CP8_L3B_MLP2_L8B_trunc2_nc_fft
CP8_L3B_MLP2_L8B_trunc8_nc_fft
CP8_L3B_MLP2_L8B_trunc8_nc_ft
CP8_L3B_MLP2_L8B_trunc2_nc_ft
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

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 2 --compressed_doc_in_icl

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
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad 


    ;;

esac
