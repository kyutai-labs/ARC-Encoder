#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=1,4 #42
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx30,par2dc5-ai-prd-cl02s04dgx20,par2dc5-ai-prd-cl02s01dgx01,par2dc5-ai-prd-cl02s04dgx24,par2dc5-ai-prd-cl02s01dgx09,par2dc5-ai-prd-cl02s04dgx11
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used

# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_NVwlllm_5noembed_3multi_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_NVwollm_0noembed_3multi_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_alltrained_5noembed_3multi_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_NVwlllm_5noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_NVwollm_0noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_alltrained_5noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_alltrained_1noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_alltrained_10noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_allwollm_0noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_poolllm_5noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_poolwollm_0noembed054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_allwollm_0noembed_3multi_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_poolllm_5noembed_3multi_054f63f8.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/hybrid_poolwollm_0noembed_3multi_054f63f8.yaml
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
    micromamba run -n llm_embed torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train $CONFIG

RUN_NAME=$(basename "$CONFIG" .yaml)

echo "Starting evaluation of run $RUN_NAME"

srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/eval_new_hybrid.json \
    --n_passages 1000 --max_seq_len 64 
   
echo "Finished at: $(date)"

# End of file



