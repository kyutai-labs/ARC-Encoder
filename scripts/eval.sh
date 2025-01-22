#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx14
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used

# Get the configuration file for this job
RUN_NAMES=(
nopref_pretrain_nollm_trained_rec_multipassage_5darr64
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

echo "Starting job array ${SLURM_ARRAY_TASK_ID} with config ${CONFIG}"
echo "Using $N_GPUS GPUs: $CUDA_VISIBLE_DEVICES"
echo "Starting at: $(date)"


echo "Starting evaluation of run $RUN_NAME"

# Get the specific config file for this array task


# NEW OUT PATH /home/hippolytepilchen/code/embed_llm/results
srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/ColBERT/eval_nopref_pretrained.json \
    --n_passages 500 --max_seq_len 64 --benchmarks NQ --multi_passages 3
;;
   
echo "Finished at: $(date)"

# End of file

# case $SLURM_ARRAY_TASK_ID in
#   0)
#     echo "Starting evaluation of run $RUN_NAME"

#     # Get the specific config file for this array task
#     RUN_NAME=${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}

#     # NEW OUT PATH /home/hippolytepilchen/code/embed_llm/results
#     srun --gpus=$N_GPU \
#         micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --eval_reconstruction  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_nopref_pretrained_NVEmbed.json \
#         --n_passages 500 --max_seq_len 128
#     ;;

#   1)
#     srun --gpus=$N_GPU \
#     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --mistral --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/mistral/eval_mistral_QA.json \
#     --n_passages 500 --max_seq_len 128 ;;

#   *)
#     echo "Could not find command for job $SLURM_ARRAY_TASK_ID" ;;
# esac
   



