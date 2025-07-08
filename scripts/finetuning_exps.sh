#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-1
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=data_ft_explore
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/finetuning/embed_llm_%A_%a.out
#SBATCH --dependency=afterany:810981_45


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used




CONFIG_FILES=(
# config/experiments/ablations/ablations_mlp/ft/CP8_L3B_MLP32_M7B_nc_ft.yaml 
# config/experiments/ablations/ablations_mlp/ft/CP8_L3B_MLP16_M7B_nc_ft.yaml 
# config/experiments/ablations/ablations_mlp/ft/CP8_L3B_MLP8_M7B_nc_ft.yaml
# config/experiments/ablations/ablations_mlp/ft/CP8_L3B_MLP4_M7B_nc_ft.yaml 
# config/experiments/datasets/CP4_L3B_MLP2_M7B_20rec_ft_alltstar_nob.yaml 
# config/experiments/datasets/CP4_L3B_MLP2_M7B_20rec_ft_allstar.yaml
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc4_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc2_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc1_nc_ft.yaml
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc4_nc_fft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc2_nc_fft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc1_nc_fft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_notcausal_fft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP8_M7B_trunc1_nc_fft.yaml
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP8_M7B_trunc1_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc21_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc21_nc_fft.yaml # DONE
# config/experiments/ablations/ablations_pooling/ft/CP8_L3B_MLP2_M7B_kmeans_ft.yaml 
# config/experiments/ablations/ablations_pooling/ft/CP8_L3B_MLP2_M7B_last_ft.yaml 
# config/experiments/ablations/ablations_pooling/ft/CP8_L3B_MLP2_M7B_memtoks_ft.yaml
# config/experiments/ablations/ablations_pooling/ft/CP8_L3B_MLP2_M7B_fusion_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc10_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc8_nc_ft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc6_nc_ft.yaml
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_50kft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_60kft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_70kft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_80kft.yaml 
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_90kft.yaml 
# config/experiments/datasets/CP4_L3B_MLP2_M7B_20rec_fft_alltstar_nob.yaml 
# config/experiments/datasets/CP4_L3B_MLP2_M7B_20rec_fft_allstar.yaml
# config/experiments/datasets/CP8_L3B_MLP2_M7B_20rec_fft4_allstar.yaml 
# config/experiments/datasets/CP8_L3B_MLP2_M7B_20rec_fft4_alltstar_nob.yaml
# config/experiments/ablations/ablations_pooling/ft/CP8_L3B_MLP2_M7B_kmeans_ft.yaml
# config/experiments/ablations/ablations_encoder/ft/CP16_L3B_MLP2_M7B_allckpts_very_long_100kft.yaml
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc10_nc_fft.yaml 
# config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc8_nc_fft.yaml
config/experiments/ablations/ablations_encoder/final_ft/CP8_L3B_MLP2_M7B_trunc6_nc_fft.yaml
config/experiments/ablations/ablations_encoder/ft/CP8_L3B_MLP2_M7B_trunc6_nc_ft.yaml
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
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B --tmp_folder ablations/ --compress_doc_in_icl --n_icl_exs 5


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B --tmp_folder ablations/ --compress_doc_in_icl
    ;;
*parll)

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_test_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 1 --compress_doc_in_icl

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_test_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 2 --compress_doc_in_icl

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_test_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME --llm_name mistral_7B  --embed_name Llama3.2-3B  --multi_passages 5 --compress_doc_in_icl

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_test_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME  --llm_name mistral_7B   --embed_name Llama3.2-3B  --multi_passages 2 --together_multi_passages --compress_doc_in_icl


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_test_ft.json \
        --n_passages 500 --max_seq_len 64 --icl_w_document --run_name $RUN_NAME  --llm_name mistral_7B   --embed_name Llama3.2-3B  --multi_passages 5 --together_multi_passages --compress_doc_in_icl
    ;;

*MLP2_M7B_*_interleaved*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --llm_name mistral_7B --embed_name Llama3.2-3B --compressed_doc_in_icl


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name mistral_7B --embed_name Llama3.2-3B --compressed_doc_in_icl --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name mistral_7B --embed_name Llama3.2-3B --compressed_doc_in_icl --benchmark "French" 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name mistral_7B --embed_name Llama3.2-3B --benchmark "French" 
    ;;

*MLP2_L8B_*_interleaved*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --llm_name Llama3.1-8B --embed_name Llama3.2-3B --compressed_doc_in_icl


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name Llama3.1-8B --embed_name Llama3.2-3B --compressed_doc_in_icl --new_template

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name Llama3.1-8B --embed_name Llama3.2-3B --compressed_doc_in_icl --benchmark "French" 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --llm_name Llama3.1-8B --embed_name Llama3.2-3B --benchmark "French" 
    ;;

*L3B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad  --embed_name Llama3.2-3B

    ;;

*L3B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name Llama3.1-8B --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name Llama3.1-8B --embed_name Llama3.2-3B

    ;;


*L8B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name Llama3.1-8B --embed_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name Llama3.1-8B --embed_name Llama3.1-8B

    ;;


    
*L8B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.1-8B

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
