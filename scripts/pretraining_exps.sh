#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=11
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2   
#SBATCH --job-name=rec_heavy_pt
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/pretraining/embed_llm_%A_%a.out



# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


CONFIG_FILES=(
config/experiments/heavier_pt/CP8_L3B_MLP2_M7B_20rec.yaml 
config/experiments/heavier_pt/CP8_L3B_MLP2_L8B_20rec.yaml
config/experiments/heavier_pt/CPtrue16_L3B_MLP2_M7B_50rec.yaml
config/experiments/heavier_pt/CP8_L3B_MLP2_M7B_20rec_3interleaved.yaml
config/experiments/heavier_pt/CP8_L3B_MLP2_L8B_20rec_3interleaved.yaml 
config/experiments/heavier_pt/CPtrue16_L3B_MLP2_M7B_20rec_Dist_v2.yaml
config/experiments/heavier_pt/CPtrue16_L3B_MLP2_M7B_20rec_v2.yaml
config/experiments/heavier_pt/CP16_L8B_to_mistral_30rec.yaml 
config/experiments/heavier_pt/CP16_L8B_to_mistral_5rec.yaml
config/experiments/heavier_pt/CPtrue8memtoks_L3B_MLP2_M7B_20rec_v2.yaml
config/experiments/heavier_pt/CPtrue16_L8B_MLP2_M7B_20rec_v2.yaml
config/experiments/heavier_pt/CPtrue16_L3B_MLP2_M7B_5rec.yaml
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

*_to_llama_*)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --llm_name Llama3.1-8B 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --llm_name Llama3.1-8B 

    ;;

*_to_mistral_*)


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --embed_name Llama3.1-8B

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500  --eval_trad --run_name $RUN_NAME --embed_name Llama3.1-8B

    ;;

*L8B_to_mistral*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.1-8B 


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad  --embed_name Llama3.1-8B 

    ;;
*P16_L8B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name mistral_7B --embed_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name mistral_7B --embed_name Llama3.1-8B

    ;;

*P16_M7B_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   

    ;;

*P16_M7B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name mistral_7B --llm_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --embed_name mistral_7B --llm_name Llama3.1-8B

    ;;

*llama3Benc*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B

    ;;

*L3B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad    --embed_name Llama3.2-3B

    ;;

*L3B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --llm_name Llama3.1-8B --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --llm_name Llama3.1-8B --embed_name Llama3.2-3B

    ;;

*L3B_MLP2_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad   --embed_name Llama3.2-3B

    ;;


*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pretraining.json  \
        --n_passages 500 --run_name $RUN_NAME --eval_trad 


    ;;


esac
