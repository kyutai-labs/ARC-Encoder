#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-4%3
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/baselines 
#SBATCH --job-name=icae_pt
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/baselines/icae/icae_pt_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx01,par2dc5-ai-prd-cl02s04dgx03,par2dc5-ai-prd-cl02s01dgx21,par2dc5-ai-prd-cl02s01dgx05,par2dc5-ai-prd-cl02s01dgx17


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


CONFIG_FILES=(
config/experiments/ablations/new_top/CP8_L3B_MLP2_M7B_best_10k.yaml 
config/experiments/ablations/new_top/CP8_L3B_MLP2_M7B_best_10k_noinsert.yaml 
config/experiments/ablations/new_top/CP8_L3B_MLP2_M7B_best_10k_higher_lr.yaml
config/experiments/ablations/new_top/CP8_L3B_MLP2_M7B_best_40k.yaml 
config/experiments/ablations/new_top/CP8_L3B_MLP2_M7B_best_20k.yaml 
)


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



esac
