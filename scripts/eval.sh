#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=12-59
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s01dgx09,par2dc5-ai-prd-cl02s04dgx27,par2dc5-ai-prd-cl02s04dgx23,par2dc5-ai-prd-cl02s01dgx04,par2dc5-ai-prd-cl02s01dgx23,par2dc5-ai-prd-cl02s02dgx04,par2dc5-ai-prd-cl02s01dgx06

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used


# Get the configuration file for this job
RUN_NAMES=(
Div2Compress_MeanSA_MLP_Cont_L16_nonorm_SL128
64Compress_MeanSA_MLP_Cont_L16_nonorm_SL128
64Compress_MeanSA_MLP_Cont_L16_nonorm
NoCompress_MLP_Cont_L16_newrms
NoCompress_MLP_Cont_L24_nonorm_SL128
64Compress_MeanSA_MLP_Cont_L24_nonorm
Div2Compress_MeanSA_MLP_Cont_L16_nonorm
Div2Compress_MeanSA_MLP_Cont_L24_nonorm
NoCompress_MLP_Cont_L16_nonorm_SL128
NoCompress_MLP_Cont_L24_nonorm
64Compress_Conv_MLP_Cont_L16_nonorm_SL128
64Compress_Mean_MLP_Cont_L16_nonorm_SL128
NoCompress_MLP_Cont_L16 # New job
NoCompress_MLP_Cont_L24
Div2Compress_Mean_MLP_Cont_L16_newrms
Div2Compress_Mean_MLP_Cont_L16_nonorm
Compr2_L16_Cont_res
Compr2_L16_lastSA_Cont
Compr2_L16_Cont
Div2Compress_MeanSA_MLP_Cont_L24_newrms
NoCompress_MLP_Cont_L24_newrms_rSL
Div2Compress_MeanSA_MLP_Cont_L24_oldrms
Compr2_L16_Cont_MTA
Compr2_L16_Cont_2NormMTA
Compr2_L16_Cont_MTAConv
Compr2_L16_Cont_MTAConv_2Norm
Compr2_L16_Cont_dist
Div2Compress_MeanSA_MLP_Cont_L16_oldrms
Compr2_L16_Cont_2NormSA
Compr2_L16_Cont_adapt
Compr2_L16_80Cont
NoCompress_MLP_Cont_L16_newrms_rSL_v2
NoCompress_MLP_Cont_L24_newrms_rSL_v2
Compr2_L16_Cont_MTA_v2
Compr2_L16_Cont_res_8
Compr2_L16_Cont_res_2Norm
64Compress_SVDSA_MLP_Cont_L24
64Compress_Conv_MLP_Cont_L24
NoCompress_MLP_Rec_L4
NoCompress_MLP_Rec_L8
NoCompress_MLP_Rec_L16
NoCompress_MLP_Cont_L24_SL512_distill
NoCompress_MLP_Cont_L8
NoCompress_MLP_Cont_L4
NoCompress_MLP_Cont_L16_res0
Div2Compress_MeanSA_MLP_Cont_L16
NoCompress_MLP_LLM_Cont_L24
NoCompress_MLP_Cont_L24_SL128
NoCompress_MLP_Cont_L4_nonorm
NoCompress_MLP_Cont_L8_nonorm
Div2Compress_MeanSA_MLP_Rec_L16
NoCompress_MLP_Cont_L16_nonorm
NoCompress_EmbMLP_nocausal_Cont_L24_long
64Compress_Mean_MLP_Cont_L24_nonorm
64Compress_Mean_MLP_Cont_L16_nonorm
64Compress_Conv_MLP_Cont_L16_nonorm
64Compress_Mean_MLP_Cont_L16_newrms
64Compress_Conv_MLP_Cont_L24_nonorm
Div2Compress_MeanSA_MLP_Cont_L16_newrms
64Compress_Conv_MLP_Cont_L16_newrms
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

*)
    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
        --n_passages 500 --max_seq_len 64 --multi_passages 1  --icl_w_document --run_name $RUN_NAME 

    srun --gpus=$N_GPU \
        micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
        --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1  --icl_w_document --compressed_doc_in_icl

    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 9000  --icl_w_document 

    # srun --gpus=$N_GPU \
    #     micromamba run -n llm_embed python embed_llm/generation/evaluation.py  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    #     --n_passages 500 --max_seq_len 64 --run_name $RUN_NAME --multi_passages 1 --ckpt 4000  --icl_w_document 


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_true_dissect.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3  --icl_w_document 

    ;;


esac

