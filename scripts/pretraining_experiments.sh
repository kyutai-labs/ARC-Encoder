#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-10%8
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=distill_pretrain_embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_embmid_MaxEmb_1_50cont_0alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_embmid_MaxEmb_1_50cont_2alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_embmid_MaxEmb_3_50cont_0alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_embmid_MaxEmb_3_50cont_2alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_1_50cont_0alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_1_50cont_2alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_3_50cont_01alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_3_50cont_0alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_3_50cont_2alpha_08tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/DistillTraining_mid_MaxEmb_3_50cont_2alpha_1tmp.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/ToyInstruct_LLM_False_Emb_False_MaxEmb_3_alpha_2.yaml
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
*_MaxEmb_1*)


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --reconstruct_seq_len 256


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 25000


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 20000 
    ;;

*)
    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --reconstruct_seq_len 256 --multi_passages 3

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000  --multi_passages 2


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 1



    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 25000  --multi_passages 3


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    --n_passages 500 --max_seq_len 64 --ckpt 20000  --multi_passages 3
    ;;

esac

# End of file



