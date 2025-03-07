#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-4
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=instruct_embed_llm
#SBATCH --dependency=afterany:671979_3
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx15,par2dc5-ai-prd-cl02s03dgx14,par2dc5-ai-prd-cl02s04dgx25,par2dc5-ai-prd-cl02s04dgx05,par2dc5-ai-prd-cl02s01dgx29
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/instruct/embed_llm_%A_%a.out

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used



# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Compress_32_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Compress_4_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Compress_all_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Compress_nothing_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Distractor20_Instruct.yaml # TODO
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_CA_Rec_Distractor20_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolRlatEmbed_CA_Cont_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolEmbed_CA_02Cont_Dist_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolEmbed_CA_Rec_Distractor_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolRlatEmbed_CA_Cont_2comp_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolRlatEmbed_CA_Cont_64comp_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolRlatEmbed_CA_Cont_8comp_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Instruct_v2.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Instruct_v2_new_data.yaml
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
*xRAG1*)


    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 
    ;;

*SQUAD*)


    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 
    
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 5 
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 4 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 2 

    ;;

*)
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 5 
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 4 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 2 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 --ckpt 9500

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 3 --ckpt 9000
    ;;


esac














