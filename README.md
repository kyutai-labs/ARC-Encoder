# Compressed Retrieval Augmented LLM

## Overview
In this repository, you will find all the necessary tools to train and infer with our Compressed Retrieval Augmented LLM. This system demonstrates improved performance on knowledge-intensive tasks compared to standard LLMs by retrieving and leveraging insightful context. The retrieved passages are compressed to retain only the most useful information, minimizing computational costs associated with large context sizes. This balance ensures efficiency and enhanced results.

## Folder Structure

### `embed_llm/data`
Contains scripts for loading, tokenizing, and formatting datasets. Supports various training tasks, including continuation, reconstruction, hybrid tasks, and instruction tuning.

### `embed_llm/generation`
Includes evaluation scripts to measure reconstruction abilities and QA performance. Facilitates testing and benchmarking for generation-related tasks.

### `embed_llm/models`
Houses the modules defining the Compressed Retrieval Augmented LLM. The `wrapped_models_training.py` file contains functions for loading pretrained models and fine-tuning them using LoRA under the Fully Sharded Data Parallel framework for efficient GPU parallelization.

### `embed_llm/monitoring`
Provides functions to monitor training progress, track metrics, and log relevant information for analysis.

### `embed_llm/retrieval`
Contains scripts for embedding and retrieving data from the desired database using the [NVEmbedv2](https://arxiv.org/abs/2405.17428) model. Modifications are included to retrieve embeddings from NVEmbedv2 before pooling, enabling more flexible operations.

### `embed_llm/training`
Includes files necessary for training the pipeline on multiple GPUs in parallel. Features evaluation and checkpointing capabilities to ensure robustness and reproducibility.

## Getting Started

### Environment Setup
First, create an environment:

```bash
micromamba create -n embed_llm python=3.10 -c conda-forge
micromamba activate embed_llm
pip install -U -r requirements.txt
```

### Training
Once you have access to multiple GPUs, launch a training session by configuring your experiment in a YAML file located in the `config` folder. Include your [Weights & Biases](https://wandb.ai/site/) credentials in the YAML file to enable training monitoring.

```bash
srun --gpus=$N_GPU micromamba run -n embed_llm torchrun --nproc-per-node $N_GPUS --master_port $MASTER_PORT -m train config/experiments/debug_pretrain.yaml
```

### Evaluation & Generation
Currently, evaluation and generation are supported on two GPUs:

```bash
srun --gpus=2 micromamba run -n embed_llm python embed_llm/generation/evaluation.py --run_name $RUN_NAME --eval_reconstruction --out_file results/eval_QA_reconstruct.jsonl \
--n_passages 500 --max_seq_len 64 --ckpt 30000 --reconstruct_seq_len 256 --multi_passages 3
```

Alternatively, you can use the `test_generation.ipynb` notebook for evaluation and generation tasks.

## License
This project is licensed under [LICENSE_NAME]. Refer to the LICENSE file for more details.
