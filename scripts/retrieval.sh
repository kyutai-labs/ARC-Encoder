#!/bin/bash

# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --array=0
#SBATCH --job-name=retrieve_passages
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx07,par2dc5-ai-prd-cl02s04dgx14
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/retrieval/slurm_embeddings_%A_%a.log 

# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID )) # Take care if already used

# Get the configuration file for this job
DATA_PATHS=(
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_open_data/train.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/commonsense_qa.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_data_old/train.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/triviaqa_data/train.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/freebase_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/msmarco_qa.jsonl 
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/web_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/wiki_qa_good_answer.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/wiki_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/yahoo_qa.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_open_data/eval.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/nq_data_old/test.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/datasets/Question_Answering/triviaqa_data/test.jsonl
)

OUT_PATHS=(
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/nq_open_data.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/commonsense_qa.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/nq_data_old.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/triviaqa_data.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/freebase_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/msmarco_qa.jsonl 
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/web_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/wiki_qa_good_answer.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/wiki_qa.jsonl  
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/instruct_data/QA_w_retrieved_passages_NVEmbed/yahoo_qa.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/nq_open_data.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/nq_data_old.jsonl
/lustre/scwpod02/client/kyutai-interns/hippop/processed_data/eval_QA_NVEmbed/triviaqa_data.jsonl
)




# Get the specific config file for this array task
DATA_PATH=${DATA_PATHS[$SLURM_ARRAY_TASK_ID]}
OUT_PATH=${OUT_PATHS[$SLURM_ARRAY_TASK_ID]}



srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/retrieval/passage_retrieval.py -outpath $OUT_PATH -data_path $DATA_PATH  -bs 64
   
echo "Finished at: $(date)"

# End of file



