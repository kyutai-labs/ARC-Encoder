#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-1
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx07
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used

# Get the configuration file for this job
RUN_NAMES=(
ToyPretraining_LLM_False_Emb_False_MaxEmb_3_0.5cont_0.1alpha_16BS_tmp
ToyPretraining_LLM_False_Emb_False_MaxEmb_3_0.5cont_16BS
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
*_MaxEmb_1*)
    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 #--reconstruct_seq_len 256 --eval_reconstruction


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 25000 #--reconstruct_seq_len 256 --eval_reconstruction


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 20000 --reconstruct_seq_len 256 --eval_reconstruction
    ;;

*)
    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 3

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 2

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 1

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 25000 --multi_passages 3


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 20000 --reconstruct_seq_len 256 --multi_passages 3
    ;;

esac



