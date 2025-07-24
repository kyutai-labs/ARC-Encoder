#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-3
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2   
#SBATCH --job-name=long_explor_pt
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/ablations/pt_long_llm_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s02dgx27,par2dc5-ai-prd-cl02s03dgx27,par2dc5-ai-prd-cl02s04dgx09,par2dc5-ai-prd-cl02s03dgx21,par2dc5-ai-prd-cl02s02dgx23,par2dc5-ai-prd-cl02s03dgx11

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used


CONFIG_FILES=(
config/experiments/ablations/ablations_pooling/CP8_default_last.yaml 
config/experiments/ablations/ablations_pooling/CP8_default_memtoks.yaml 
config/experiments/ablations/ablations_pooling/CP8_default_kmeans.yaml 
config/experiments/ablations/ablations_pooling/CP8_default_fusion.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_LoRA.yaml 
config/experiments/ablations/ablations_mlp/CP8_default_MLP32.yaml 
config/experiments/ablations/ablations_mlp/CP8_default_MLP16.yaml 
config/experiments/ablations/ablations_mlp/CP8_default_MLP8.yaml 
config/experiments/ablations/ablations_mlp/CP8_default_MLP4.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_wcaus.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_trunc21.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_trunc14.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_trunc4.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_notoks.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_trunc1.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_evry2.yaml 
config/experiments/ablations/ablations_encoder/CP8_default_evry1.yaml
config/experiments/ablations/ablations_reconstruction_cp/CP32_default.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP16_default.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP8_default_nointer.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP8_default_100rec.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP8_default_50rec.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP4_default.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP8_default_0rec.yaml 
config/experiments/ablations/ablations_reconstruction_cp/CP2_default.yaml 
config/experiments/ablations/ablations_pairs/CP8_L3B_MLP2_L8B_default.yaml
config/experiments/ablations/ablations_pairs/CP8_L8B_M7B_default.yaml 
config/experiments/ablations/ablations_pairs/CP8_L8B_L8B_default.yaml 
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

RUN_NAME=$(basename "$CONFIG" .yaml
)

echo "Starting evaluation of run $RUN_NAME"


case $RUN_NAME in

*L3B_MLP2_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B --llm_name Llama3.1-8B


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --embed_name Llama3.2-3B --llm_name Llama3.1-8B

    ;;

*L3B_*_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.2-3B  --tmp_folder 'ablations/'


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --embed_name Llama3.2-3B  --tmp_folder 'ablations/'

    ;;

*L8B_MLP2_L8B* | *L8B_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.1-8B --llm_name Llama3.1-8B --tmp_folder 'ablations/'


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --embed_name Llama3.1-8B --llm_name Llama3.1-8B --tmp_folder 'ablations/'

    ;;

*L8B_MLP2_M7B* | *L8B_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME  --embed_name Llama3.1-8B --tmp_folder 'ablations/'


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --embed_name Llama3.1-8B --tmp_folder 'ablations/'

    ;;

*M7B_MLP2_L8B* | *M7B_L8B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --llm_name Llama3.1-8B --tmp_folder 'ablations/'


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad --llm_name Llama3.1-8B --tmp_folder 'ablations/'

    ;;

*M7B_MLP2_M7B* | *M7B_M7B*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME   --tmp_folder 'ablations/'


    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_pt_ablations.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad  --tmp_folder 'ablations/'

    ;;


esac
