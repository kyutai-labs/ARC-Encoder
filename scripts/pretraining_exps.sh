#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-6
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/versatile_compressor
#SBATCH --job-name=pretrain_llama
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/pretraining/embed_llm_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx16,par2dc5-ai-prd-cl02s03dgx20,par2dc5-ai-prd-cl02s02dgx23,par2dc5-ai-prd-cl02s04dgx18,par2dc5-ai-prd-cl02s03dgx21,par2dc5-ai-prd-cl02s03dgx18,par2dc5-ai-prd-cl02s03dgx16
# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used




CONFIG_FILES=(
config/experiments/No_Comp/NC_llama3B_mlp_v2.yaml 
config/experiments/No_Comp/NC_llama3B_rms.yaml 
config/experiments/No_Comp/NC_llama8B_v2.yaml 
config/experiments/No_Comp/NC_llama3B_mlp_nobot_v2.yaml
config/experiments/No_Comp/NC_llama8B_rms.yaml
config/experiments/No_Comp/NC_llama8B_mlp.yaml 
config/experiments/No_Comp/NC_llama8B_mlp_nobot.yaml 
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

*llama3B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name Llama3.2-3B

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --llm_name Llama3.2-3B --ckpt 9000

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name Llama3.2-3B

    ;;

*llama8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name Llama3.1-8B

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --llm_name Llama3.1-8B --ckpt 19000

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/versatile_compressor/results/NVEmbed/eval_llama.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name Llama3.1-8B

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
