#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-32%24
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=par2dc5-ai-prd-cl02s04dgx22,par2dc5-ai-prd-cl02s04dgx21,par2dc5-ai-prd-cl02s04dgx18,par2dc5-ai-prd-cl02s04dgx14,par2dc5-ai-prd-cl02s04dgx11,par2dc5-ai-prd-cl02s02dgx26
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=eval_models
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/eval/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID - 100)) # Take care if already used

# Get the configuration file for this job
RUN_NAMES=(
NVEmbed_CA_Rec_shared
Hybrid_LLM_False_Emb_False_MaxEmb_3_PNoEmbed_0.0_StartPoint_0.0_16BS
NVEmbed_CA_Cont
NVEmbed_CA_Cont_distill_2alpha_1tmp
NVEmbed_CA_Hybrid0
NVEmbed_CA_Rec
NVEmbed_CA_Rec_shared
NVEmbed_pref_Cont
NVEmbed_pref_Cont_distill_2alpha_1tmp
NVEmbed_pref_Hybrid0
NVEmbed_pref_Rec
NVEmbed_pref_Rec_xRAG1
NVEmbed_pref_Rec_xRAG1_atlas
TrainEmbed_CA_Cont
TrainEmbed_CA_Cont_distill_2alpha_1tmp
TrainEmbed_CA_Rec
TrainEmbed_pref_Cont
TrainEmbed_pref_Cont_distill_2alpha_1tmp
TrainEmbed_pref_Rec
NVEmbed_pref_Rec_xRAG5_atlas
NVEmbed_pref_Rec_xRAG5
ToyPretraining_LLM_False_Emb_False_MaxEmb_3_fullrec_16BS
Instruct_embmid_3embeds_alpha2_v
NVEmbed_CA_Cont_Instruct
NVEmbed_CA_Rec_Instruct
NVEmbed_pref_Cont_Instruct
NVEmbed_pref_Rec_Instruct
TrainEmbed_CA_Cont_Instruct
TrainEmbed_CA_Cont_Instruct_notall
TrainEmbed_CA_Rec_Instruct
TrainEmbed_CA_Rec_Instruct_further_embeds
TrainEmbed_pref_Cont_Instruct
TrainEmbed_pref_Rec_Instruct
# DistillTraining_mid_MaxEmb_3_50cont_01alpha_1tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_fullrec_16BS
# DistillTraining_embmid_MaxEmb_3_50cont_2alpha_1tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_fullcont_16BS_alternativeCA
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_0.5cont_16BS
# DistillTraining_mid_MaxEmb_3_50cont_0alpha_1tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_0.5cont_1alpha_16BS_tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_1_fullcont_2alpha_16BS_tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_1_fullcont_2alpha_16BS_alternativeCA_0.5tmp
# DistillTraining_mid_MaxEmb_3_50cont_2alpha_1tmp
# ToyPretraining_LLM_False_Emb_True_MaxEmb_1_fullcont_2alpha_16BS_0.5tmp
# ToyPretraining_LLM_False_Emb_True_MaxEmb_1_fullcont_2alpha_16BS_alternativeCA_0.5tmp
# DistillTraining_mid_MaxEmb_3_50cont_2alpha_08tmp
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_fullcont_16BS_beginCA
# ToyPretraining_LLM_False_Emb_False_MaxEmb_3_fullcont_16BS
# ToyPretraining_LLM_False_Emb_False_MaxEmb_1_0.2cont_0alpha_16BS_tmp
# DistillTraining_mid_MaxEmb_1_50cont_0alpha_1tmp
# DistillTraining_embmid_MaxEmb_3_50cont_0alpha_1tmp
# DistillTraining_embmid_MaxEmb_1_50cont_2alpha_1tmp
# ToyInstruct_LLM_False_Emb_False_MaxEmb_3_alpha_2
# DistillTraining_embmid_MaxEmb_1_50cont_0alpha_1tmp
# ToyPretraining_LLM_False_Emb_True_MaxEmb_1_0.2cont_16BS
# ToyPretraining_LLM_False_Emb_True_MaxEmb_1_pure_reconstruct_16BS
# DistillTraining_mid_MaxEmb_1_50cont_2alpha_1tmp
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
*nstruc*)

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64 --multi_passages 1

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 5

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 4


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --instruct_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 2

    ;;
*)

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 3

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64 --multi_passages 1

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 5

    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 4


    srun --gpus=$N_GPU \
    micromamba run -n llm_embed python embed_llm/generation/evaluation.py --run_name $RUN_NAME  --out_file /home/hippolytepilchen/code/embed_llm/results/NVEmbed/eval_final_multi.json \
    --n_passages 500 --max_seq_len 64   --multi_passages 2

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
    ;;

esac

