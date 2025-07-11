#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=10-11,4
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/hp_v2
#SBATCH --job-name=europarl_llama_multpass
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/baseline_eval_%A_%a.out
#SBATCH --nodelist=par2dc5-ai-prd-cl02s03dgx10

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
        --max_sample --max_seq_len 64  --wo_embeds --llm_name meta-llama/Llama-3.1-8B  --benchmarks NarrativeQA --bs 1   --n_icl_exs 0 --query_w_context --max_doc_len 32768

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64  --wo_embeds --llm_name mistralai/Mistral-7B-v0.3  --benchmarks NarrativeQA --bs 1  --n_icl_exs 0 --query_w_context --max_doc_len 32768

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
    #     --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA --bs 4 --n_icl_exs 0 --max_doc_len 6144

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
    #     --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name meta-llama/Llama-3.1-8B  --icl_w_document --benchmarks NarrativeQA --bs 1 --n_icl_exs 0 --max_doc_len 16384

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
    #     --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA --bs 1 --n_icl_exs 0 --max_doc_len 16384

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
    #     --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name meta-llama/Llama-3.1-8B  --icl_w_document --benchmarks NarrativeQA --bs 1 --n_icl_exs 0 --max_doc_len 32768

    # srun --gpus=$N_GPU  \
    #         python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
    #     --max_sample --max_seq_len 64   --multi_passages 1 --wo_embeds --query_w_context --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA --bs 1 --n_icl_exs 0 --max_doc_len 32768


    ;;


1)
   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 2.0 --max_seq_len 64 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 4.0 --max_seq_len 64 

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 8.0 --max_seq_len 64 

    ;;

2)
   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 2.0 --max_seq_len 64 

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 4.0 --max_seq_len 64 

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 8.0 --max_seq_len 64 
    ;;

3)
   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 2.0 --max_seq_len 1400 --europarl --bs 4

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 4.0 --max_seq_len 1400 --europarl --bs 4

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name mistralai/Mistral-7B-v0.3  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 8.0 --max_seq_len 1400 --europarl --bs 4



    ;;

4)


   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 2.0 --max_seq_len 1400 --europarl --bs 4

    srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 4.0 --max_seq_len 1400 --europarl --bs 4

   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json \
        --max_sample    --eval_trad   --llm_name meta-llama/Llama-3.1-8B  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2  --comp_rate 8.0 --max_seq_len 1400 --europarl --bs 4



    ;;

5)

#    srun --gpus=$N_GPU  \
#             python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json  \
#         --max_sample   --eval_trad    --europarl --bs 4 --llm_name mistralai/Mistral-7B-v0.3  --wo_embeds --max_seq_len 1400


   srun --gpus=$N_GPU  \
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json  \
        --max_sample   --eval_trad   --europarl --bs 4 --llm_name meta-llama/Llama-3.1-8B  --wo_embeds --max_seq_len 1400
    ;;




7)
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 2.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 2.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name meta-llama/Llama-3.1-8B --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    ;;
8)
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 4.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 4.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name meta-llama/Llama-3.1-8B --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    ;;

9)
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 8.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name mistralai/Mistral-7B-v0.3 --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_LLMLingua_paperfinal.json --comp_rate 8.0 \
        --max_sample --max_seq_len 64   --multi_passages 10  --compressed_doc_in_icl --compressor_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank --use_llmlingua2 --llm_name meta-llama/Llama-3.1-8B --icl_w_document --benchmarks NarrativeQA_split  --bs 1 --n_icl_exs 0
    ;;

10)
# All for excel with base models Mistral
    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 3 --wo_embeds  --llm_name meta-llama/Llama-3.1-8B  --benchmarks NQ  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8

    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 5 --wo_embeds  --llm_name meta-llama/Llama-3.1-8B  --benchmarks NQ  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8


    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 3 --wo_embeds  --llm_name meta-llama/Llama-3.1-8B  --benchmarks TRIVIAQA  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8

    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 5 --wo_embeds  --llm_name meta-llama/Llama-3.1-8B  --benchmarks TRIVIAQA  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8

    ;;

11)
# All for excel with base models Llama-3.1

    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 3 --wo_embeds  --llm_name mistralai/Mistral-7B-v0.3  --benchmarks NQ  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8

    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 5 --wo_embeds  --llm_name mistralai/Mistral-7B-v0.3  --benchmarks NQ  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8


    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 3 --wo_embeds  --llm_name mistralai/Mistral-7B-v0.3 --benchmarks TRIVIAQA  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8

    srun --gpus=$N_GPU  \ 
            python embed_llm/generation/evaluate_other_models.py  --out_file /home/hippolytepilchen/code/hp_v2/results/NVEmbed/other_models/eval_RAG_QA_paperfinal.json \
        --max_sample --max_seq_len 64   --multi_passages 5 --wo_embeds  --llm_name mistralai/Mistral-7B-v0.3  --benchmarks TRIVIAQA  --n_icl_exs 5 --icl_w_document --query_w_context --bs 8


    ;;

esac


