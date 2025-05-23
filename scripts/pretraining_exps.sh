#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-5
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=pretrain_rec_tk
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/pretraining/embed_llm_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx15,par2dc5-ai-prd-cl02s04dgx12,par2dc5-ai-prd-cl02s01dgx03,par2dc5-ai-prd-cl02s01dgx04,par2dc5-ai-prd-cl02s03dgx28,par2dc5-ai-prd-cl02s02dgx31,par2dc5-ai-prd-cl02s03dgx04,par2dc5-ai-prd-cl02s01dgx32,par2dc5-ai-prd-cl02s02dgx30,par2dc5-ai-prd-cl02s01dgx24
#SBATCH --dependency=afterany:728454_17

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


CONFIG_FILES=(
config/experiments/rec_sweeps/SA_merge_L4_CR4_decL16_pt_30rec.yaml 
config/experiments/rec_sweeps/SA_merge_L4_CR4_decL16_pt_10rec.yaml 
config/experiments/rec_sweeps/SA_merge_L4_CR4_decL16_pt_5rec.yaml 
config/experiments/rec_sweeps/64memtoks_dec_30rec.yaml 
config/experiments/rec_sweeps/64memtoks_dec_10rec.yaml 
config/experiments/rec_sweeps/64memtoks_dec_5rec.yaml
)
# config/experiments/rec_sweeps/SA_merge_L4_CR4_decL16_pt_20rec_conttok.yaml

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

Matryoshka*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --comp_rate 128

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --comp_rate 128

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --comp_rate 64

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --comp_rate 64

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --comp_rate 32

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --comp_rate 32

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --comp_rate 16

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --comp_rate 16

    ;;
*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --ckpt 9000

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --ckpt 9000

    ;;


esac
