#!/bin/bash

# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/hippolytepilchen/code/compressed_retrieval
#SBATCH --array=1-4
#SBATCH --job-name=incontext_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/slurm_output/slurm_output_%A_%a.log 

# Command Selection
case $SLURM_ARRAY_TASK_ID in


1)  

    for id_ex in 1 2 3 4 5; do
        micromamba run -n comp_retriev python \
            test_incontext.py -n_ex 5 -model_name "Mistral7B_base" -name_exp "Mistral7B_base_n_ex_$id_ex"  -id_ex $id_ex -n_test 10 -id_start 1
    done
    ;;
2)
    for id_ex in 1 2 3 4 5; do
        micromamba run -n comp_retriev python \
            test_incontext.py -n_ex 5 -model_name "Qwen7B_base" -name_exp "Qwen7B_base_n_ex_$id_ex"  -id_ex $id_ex -n_test 10 -id_start 1
    done

    ;;
3)

    for id_ex in 1 2 3 4 5; do
        micromamba run -n comp_retriev python \
            test_incontext.py -n_ex 5 -model_name "Llama8B_base" -name_exp "Llama8B_base_n_ex_$id_ex"  -id_ex $id_ex -n_test 10 -id_start 1
    done

    ;;

4)

    for id_ex in 1 2 3 4 5; do
        micromamba run -n comp_retriev python \
            test_incontext.py -n_ex 5 -model_name "Gemma7B" -name_exp "Gemma7B_n_ex_$id_ex"  -id_ex $id_ex -n_test 10 -id_start 1
    done

    ;;
*)
echo "Could not a find command for job $SLURM_ARRAY_TASK_ID"
;;
esac
