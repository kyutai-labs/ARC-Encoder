#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=3-4
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx31,par2dc5-ai-prd-cl02s01dgx08,par2dc5-ai-prd-cl02s04dgx11,par2dc5-ai-prd-cl02s03dgx22
#SBATCH --job-name=instruct_embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/instruct/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used

# Get the configuration file for this job
CONFIG_FILES=(
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Instruct_notall.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Rec_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_pref_Cont_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_pref_Rec_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Rec_Instruct_further_embeds.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_CA_Cont_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_CA_Rec_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Cont_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG5_atlas_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG5_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_wiki_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_atlas_Instruct.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG5_atlas_Instruct.yaml # Done
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_Instruct.yaml # Done
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Distill_Instruct.yaml #WIP
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_pref_Cont_Distill_Instruct.yaml  #WIP
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG5_Instruct.yaml #WIP
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_wiki_Instruct.yaml #WIP
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_pref_Cont_Distill_Instruct.yaml # WIP
# /home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainPoolEmbed_CA_Rec_Instruct.yaml #WIP
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TraincausalEmbed_CA_Rec_Instruct.yaml 
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainPoolEmbed_CA_Cont_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_pref_Rec_xRAG1_atlas_Instruct.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainEmbed_CA_Cont_Instruct_SQUAD_only.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/NVEmbed_CA_Rec_Instruct_SQUAD_only.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TraincausalEmbed_CA_Cont_Instruct.yaml # TODO Later
/home/hippolytepilchen/code/embed_llm/config/experiments/train_configs/TrainCausalPoolEmbed_CA_Cont_Instruct.yaml # SAME
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



