#!/bin/bash
# SBATCH options
#SBATCH --partition=kyutai
#SBATCH --array=0-3
#SBATCH --nodes=1         # Request single node
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --chdir=/home/hippolytepilchen/code/embed_llm
#SBATCH --job-name=embed_llm
#SBATCH --output=/lustre/scwpod02/client/kyutai-interns/hippop/experiments/embed_llm_out/embed_llm_%A_%a.out


# Set up environment
export MASTER_PORT=$((29500 + $SLURM_ARRAY_TASK_ID))

# Get the configuration file for this job
CONFIG_FILES=(
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/cross_att_half_last_layersMistral7Bdbbb7faebb2f32cf20e9.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/cross_att_5_last_layersMistral7Bdbbb7faebb2f32cf20e9.yaml
/home/hippolytepilchen/code/embed_llm/config/experiments/mistral/cross_att_0.75_last_layersMistral7Bdbbb7faebb2f32cf20e9.yaml
 )
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_16t_Mistral7B7bc7dcc2ba28873eda96.yaml  
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/continuation_1e-4_Mistral7B20ed0018b2a84fba09c4.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/latt_Mistral7B_causal.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/latt_Mistral7B_not_causal.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/mean_Mistral7B_not_causal_trunc2.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/mean_Mistral7B_not_causal_trunc6.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/mean_Mistral7B_not_causal_trunc3.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/mean_Mistral7B_not_causal.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/continuation_Mistral7B20ed0018b2a84fba09c4.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/eos_Mistral7B_causal.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/eos_Mistral7B_not_causal.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_512t_Mistral7B20ed0018b2a84fba09c4.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_256t_Mistral7Be9ffc00fa42bedbc50d0.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_128t_Mistral7B226729d875c65b331ef8.yaml 
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_64t_Mistral7B9bbea1b3b8dc23079b04.yaml
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_32t_Mistral7Bccbc3f29d69bd124c6cf.yaml   
# /home/hippolytepilchen/code/embed_llm/config/experiments/mistral/SL_16t_Mistral7B7bc7dcc2ba28873eda96.yaml  

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
    micromamba run -n llm_embed torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train $CONFIG

echo "Finished at: $(date)"

# End of file