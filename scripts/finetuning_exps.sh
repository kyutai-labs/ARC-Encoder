#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=8-18%3
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=fine_tuning_comp
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/finetuning/embed_llm_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx22,par2dc5-ai-prd-cl02s01dgx28,par2dc5-ai-prd-cl02s04dgx31,par2dc5-ai-prd-cl02s04dgx27

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used



CONFIG_FILES=(
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_NQ.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_TRIVIAQA.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_mix_QA.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_mix_QA_lang.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_mix_lang.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_fr_de.yaml 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_en_fr.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_TRIVIAQA.yaml # Work 
config/experiments/datasets/SA_merge_L4_CR4_decL16_pt_ft_en_de.yaml # Need to redo
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_NQ.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_mix_QA.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_mix_QA_lang.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_mix_lang.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_fr_de.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_en_fr.yaml
config/experiments/datasets/SA_merge_L4_CR4_decL16_ft_en_de.yaml
config/experiments/fine-tuning/meanSA_full_llm_comp4_squad.yaml
config/experiments/mem_toks/various/128memtoks_nodec_rec_squad_to64.yaml 
config/experiments/mem_toks/various/64memtoks_nodec_rec_squad_to32.yaml
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



*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad  --fine_tuned

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_ft.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad 

    ;;


esac
