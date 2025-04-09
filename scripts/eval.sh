#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=4-6
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx16

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
Compr2_L16_Cont_kNormMTA_v2
Compr2_L16_Cont_res_0
NoCompress_MLP_LLM_Cont_L24 # Same
NoCompress_EmbMLP_nocausal_Cont_L24_long # Pb loading
# Div2Compress_MeanSA_MLP_Cont_L16_newrms # No ckpt
# 64Compress_Conv_MLP_Cont_L24_nonorm # No ckpt
No_MLP_Layer0
No_MLP_Layer1
No_MLP_Layer4
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

*)
    srun --gpus=$N_GPU \
         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 

    srun --gpus=$N_GPU \
         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
        --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1  --icl_w_document --compressed_doc_in_icl

    # srun --gpus=$N_GPU \
    #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 9000  --icl_w_document 

    # srun --gpus=$N_GPU \
    #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 4000  --icl_w_document 


    srun --gpus=$N_GPU \
     python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3  --icl_w_document 

    ;;


esac

