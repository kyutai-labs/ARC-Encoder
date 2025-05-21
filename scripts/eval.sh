#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=eval_models_trad
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used



# Get the configuration file for this job
RUN_NAMES=(
64memtoks_nodec_rec
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

*istral*)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/mistral/eval_mistral_translate.json \
        --n_passages 500  --mistral --eval_trad

    ;;
*)
#    srun --gpus=$N_GPU  \
#             python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_fine_tuning.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --benchmarks SQUAD

#     srun --gpus=$N_GPU  \
#             python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_fine_tuning.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --benchmarks SQUAD

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --comp_rate -2

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --comp_rate -2

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --comp_rate -4

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --comp_rate -4

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --comp_rate -8

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --comp_rate -8

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --comp_rate -16

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --comp_rate -16

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --comp_rate -32

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_other_cr.json \
    #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5 --comp_rate -32

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_trad.json \
        --n_passages 500  --run_name $RUN_NAME --eval_trad

    ;;




# *squ*)
#     srun --gpus=$N_GPU  \
#             python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_fine_tuning.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0

#     srun --gpus=$N_GPU  \
#             python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_fine_tuning.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 5

#     ;;

# *)
#     srun --gpus=$N_GPU \
#          python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 

#     srun --gpus=$N_GPU \
#         python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#         --n_passages 500 --max_seq_len 64   --multi_passages 1  --icl_w_document --n_icl_exs 5 
#     # srun --gpus=$N_GPU \
#     #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool_revtemp.json \
#     #     --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME --n_icl_exs 0 --reversed_template

#     # srun --gpus=$N_GPU \
#     #     python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool_revtemp.json \
#     #     --n_passages 500 --max_seq_len 64   --multi_passages 1  --icl_w_document --n_icl_exs 5 --reversed_template 
#     ;;

# *)
#     srun --gpus=$N_GPU \
#          python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#         --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 

#     # srun --gpus=$N_GPU \
#     #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#     #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1  --icl_w_document --compressed_doc_in_icl

#     # srun --gpus=$N_GPU \
#     #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#     #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 9000  --icl_w_document 

#     # srun --gpus=$N_GPU \
#     #      python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#     #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 4000  --icl_w_document 


#     # srun --gpus=$N_GPU \
#     #  python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json \
#     # --n_passages 500 --max_seq_len 64   --multi_passages 3  --icl_w_document 

#     ;;


#   python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/eval_embedpool.json --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name 4P_TruncL_16_TrainL_4_decL0_newinit
esac

