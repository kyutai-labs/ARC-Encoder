#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=1-3 
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx25,par2dc5-ai-prd-cl02s04dgx18,par2dc5-ai-prd-cl02s02dgx08,par2dc5-ai-prd-cl02s02dgx14,par2dc5-ai-prd-cl02s01dgx01,par2dc5-ai-prd-cl02s04dgx15,par2dc5-ai-prd-cl02s04dgx31,par2dc5-ai-prd-cl02s03dgx32,par2dc5-ai-prd-cl02s04dgx10,par2dc5-ai-prd-cl02s02dgx04
#SBATCH --job-name=instruct_exps
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/instruct/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used

# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_Hybrid_LLM_False_Emb_False_MaxEmb_3_start_0.0.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha0.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha1_highlr.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/Instruct_mid_3embeds_alpha1.yaml
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
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
    --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 --run_name Hybrid_LLM_False_Emb_False_MaxEmb_3_PNoEmbed_0.0_StartPoint_0.0_16BS

srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
    --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 2 --run_name Hybrid_LLM_False_Emb_False_MaxEmb_3_PNoEmbed_0.0_StartPoint_0.0_16BS

srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_instruct.json \
    --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 --run_name Hybrid_LLM_False_Emb_False_MaxEmb_3_PNoEmbed_0.0_StartPoint_0.0_16BS
   
echo "Finished at: $(date)"

# End of file



