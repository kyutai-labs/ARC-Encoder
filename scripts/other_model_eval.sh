#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=2
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/eval_dissect_%A_%a.out

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


case $SLURM_ARRAY_TASK_ID in


0)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64  --wo_embeds --llm_name mistralai/Mistral-7B-v0.3 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 2  --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 3  --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 4  --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 5  --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample   --eval_trad  --wo_embeds --new_template --llm_name mistralai/Mistral-7B-v0.3 

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample   --eval_trad  --wo_embeds --new_template --europarl --llm_name mistralai/Mistral-7B-v0.3 
    ;;


1)
    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64  --wo_embeds --llm_name meta-llama/Meta-Llama-3-8B 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 2  --wo_embeds --query_w_context --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 3  --wo_embeds --query_w_context --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 4  --wo_embeds --query_w_context --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 5  --wo_embeds --query_w_context --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample    --eval_trad  --wo_embeds --new_template --llm_name meta-llama/Meta-Llama-3-8B 

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample    --eval_trad  --wo_embeds --new_template --europarl --llm_name meta-llama/Meta-Llama-3-8B 
    ;;

2)

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 1  --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 2  --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 3   --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document  --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 4  --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64    --multi_passages 5  --llm_name meta-llama/Meta-Llama-3-8B --icl_w_document --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample    --eval_trad  --new_template --llm_name meta-llama/Meta-Llama-3-8B  --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample   --eval_trad   --new_template --europarl --llm_name meta-llama/Meta-Llama-3-8B  --compressed_doc_in_icl --compressor_name microsoft/phi-2 --use_llmlingua2
    ;;


esac

