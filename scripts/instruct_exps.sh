#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=7-8
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=instruct_exps
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/instruct/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_1embeds_alpha0_lowlr_newdata.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_1embeds_alpha2_enhcleandata.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_1embeds_alpha2_xRAGdata.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_1embeds_alphafull_xRAGdatatop50.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_1embeds_alphafull_xRAGdata.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha0_lowlr_verif.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha2_1002data.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha2_notempdata.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha2_notempdata_lowlr.yaml
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


case $RUN_NAME in
*3embeds*)

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 2 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 
    ;;
*) 
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME 
    ;;

esac



