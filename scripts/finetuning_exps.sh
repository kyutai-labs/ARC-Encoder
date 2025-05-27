#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-18%4
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=fine_tuning_comp
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/finetuning/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used



CONFIG_FILES=(
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/128memtoks_nodec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/128memtoks_nodec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/128memtoks_dec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/128memtoks_dec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/64memtoks_nodec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/16memtoks_nodec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/64memtoks_dec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/32memtoks_nodec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/32memtoks_dec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/16memtoks_dec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/8memtoks_nodec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/8memtoks_dec_rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/128memtoks_nodec_rec_squad_to64.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/64memtoks_nodec_rec_squad_to32.yaml
config/experiments/rec_sweeps/ft/64memtoks_dec_30rec_squad.yaml 
config/experiments/rec_sweeps/ft/64memtoks_dec_10rec_squad.yaml 
config/experiments/rec_sweeps/ft/64memtoks_dec_5rec_squad.yaml
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/32memtoks_dec_rec_TS
/home/hippolytepilchen/code/hp_v2/config/experiments/mem_toks/corrected/32memtoks_nodec_rec_TS
)


# config/experiments/mem_toks/ft/64memtoks_nodec_conttok_TS.yaml 
# config/experiments/mem_toks/ft/64memtoks_nodec_conttok_squad.yaml 
# config/experiments/mem_toks/ft/4memtoks_nodec_squad.yaml 
# config/experiments/mem_toks/ft/4memtoks_nodec_rec_TS.yaml
# config/experiments/mem_toks/ft/64memtoks_dec_conttok_TS.yaml 
# config/experiments/mem_toks/ft/64memtoks_dec_conttok_squad.yaml
# config/experiments/mem_toks/ft/64memtoks_dec_TS.yaml 
# config/experiments/mem_toks/ft/64memtoks_dec_squad.yaml
# config/experiments/rec_sweeps/ft/SA_merge_L4_CR4_decL16_pt_5rec_conttok_TS.yaml
# config/experiments/rec_sweeps/ft/SA_merge_L4_CR4_decL16_pt_5rec_conttok_squad.yaml
# config/experiments/rec_sweeps/ft/SA_merge_L4_CR4_pt_5rec_squad.yaml 
# config/experiments/rec_sweeps/ft/SA_merge_L4_CR4_pt_5rec_conttok_squad.yaml


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
*squad*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_corrected_memtoks.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_corrected_memtoks.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_corrected_memtoks.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad  --fine_tuned

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_corrected_memtoks.json \
        --n_passages 500 --run_name $RUN_NAME --eval_trad 

    ;;


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
