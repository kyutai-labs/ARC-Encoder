#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=4-7
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx04
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out



# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used

# Get the configuration file for this job
RUN_NAMES=(
NoCompress_MLP_Cont_L8
NoCompress_MLP_Cont_L4
NoCompress_MLP_Cont_L16_res0
Div2Compress_MeanSA_MLP_Cont_L16
NTP_24_8
NTP_1_31
NTP_8_24
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

*Instr*)
  srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 --icl_before_pref --llmemb_icl_w_context


    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
        --n_passages 500 --max_seq_len 64 --instruct_name $RUN_NAME --multi_passages 1 --ckpt 9000 --icl_before_pref --llmemb_icl_w_context

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3 --icl_before_pref --llmemb_icl_w_context
    ;;
*)
  srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
        --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --icl_before_pref --llmemb_icl_w_context


    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
        --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 9000 --icl_before_pref --llmemb_icl_w_context
    
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
        --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 4000 --icl_before_pref --llmemb_icl_w_context

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_dissect.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3 --icl_before_pref --llmemb_icl_w_context
    ;;
    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64   --multi_passages 3

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64 --multi_passages 1

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64   --multi_passages 5

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64   --multi_passages 4


    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64   --multi_passages 2

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    # --n_passages 500 --max_seq_len 64 --ckpt 15000 --multi_passages 3

    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus_clean.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 3
    
    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus_clean.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 2
    
    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus_clean.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 1


    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 3

    
    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 1

    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    #     --n_passages 500 --max_seq_len 64  --multi_passages 2
    


    
    
    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    #     --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 3 --benchmarks HotpotQA

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 3

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 5
    
    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64 --ckpt 30000 --multi_passages 4


    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64  --reconstruct_seq_len 256 --multi_passages 3

    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64   --multi_passages 2


    # srun --gpus=$N_GPU \
    # micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_hybrid_focus.json \
    # --n_passages 500 --max_seq_len 64  --multi_passages 1
    # ;;

esac

