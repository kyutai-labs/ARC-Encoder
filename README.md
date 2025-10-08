#  Adaptable text Representations Compressor (ARC)

[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/<your-paper-id>)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/HippolyteP/ARC_finetuning)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository contains the code to reproduce most of the experiments from the paper [**BLABLA**](bla).  
You can pretrain and fine-tune your own **ARC-Encoder**, or directly use our released checkpoints to fine-tune on specific datasets.  

---

## 🗂️ Table of Contents
- [Installation](#-installation)
- [ Preparing Datasets](#-preparing-datasets)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Folder Structure](#-folder-structure)
- [Acknowledgments](#-acknowledgments)

---


## Installation

### 1️⃣ Clone this repository
```sh
git clone ...
```
***Once cloned set import paths at `embed_llm/__init__.py`.*** Then, export these as environment variables since they are useful in the config files. 

### 2️⃣ Install all required dependencies:
We recommend using [`uv`](https://docs.astral.sh/uv/) to manage the environment.
It's about 10x faster than `pip` and has a bunch of other benefits too.
Once you've installed `uv`, no explicit package installation is required:
just prefix every command with `uv run` (e.g. train using `uv run torchrun ...`).
This will automatically install the necessary dependencies based on `pyproject.toml`.

#### Installing without `uv`

If you prefer working with `pip`, and handling the install manually, you will need at least Python 3.10. 
We still advise using a virtual environment,
which can be created using [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
[virtualenv](https://virtualenv.pypa.io/en/latest/).
Then, run:

```sh
cd moshi-finetune
pip install -e .
```

### 3️⃣ Load backbone models

Create a directory <MODEL_PATH> where you’ll store the backbone models for your ARC-Encoder and decoder.
For LLaMA models, register on the [LLaMa downloads](https://www.llama.com/llama-downloads/)  page to obtain URLs.

```
# For Mistral 7B
wget   https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar -P <MODEL_PATH>/mistral_7B

# For Llama3.2 3B
wget   url -P <MODEL_PATH>/Llama3.2-3B

# For Llama3.1 8B
wget   url -P <MODEL_PATH>/Llama3.1-8B

# For Llama2 7B Chat
wget   https://huggingface.co/meta-llama/Llama-2-7b-chat/resolve/main/consolidated.00.pth? -P <MODEL_PATH>/Llama2-7B-Chat
wget https://huggingface.co/meta-llama/Llama-2-7b-chat/resolve/main/tokenizer.model? -P <MODEL_PATH>/Llama2-7B-Chat
echo '{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000}' > <MODEL_PATH>/Llama2-7B-Chat/params.json

# For Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/model.safetensors? -P <MODEL_PATH>/Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/tokenizer.json? -P <MODEL_PATH>/Olmo7B
echo '{"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 50304}' > <MODEL_PATH>/Olmo7B/params.json
```


## Prepare datasets

For fine-tuning, load our Hugging Face dataset:
👉 [ARC Finetuning Dataset](https://huggingface.co/datasets/HippolyteP/ARC_finetuning). 

Use the `load_datasets.ipynb` notebook to load the evaluation datasets. 

Then,  run the following scripts to prepare evaluation datasets:
- `retrieval/embeddings.py` — GPU recommended, creates embeddings for Wikipedia text chunks.
- `retrieval/passage_retrieval.py`— CPU fine, retrieves passages for Natural Questions and TriviaQA, as described in the paper.


## Load pre-train ARC-Encoders
Pretrained ARC-Encoders will soon be released and available on HuggingFace, stay tuned. 

| Models                | Specificities                                       | 
| :-------------------- | :-------------------------------------------------- | 
| [ARC<sub>8</sub>-Encoder<sup>L</sup>](link)| Trained on 6.5B tokens on Llama3.1-8B base specifically with a pooling factor (PF) of 8                                 |  
| [ARC<sub>8</sub>-Encoder<sup>M</sup>](link)| Trained on 6.5B tokens on Mistral-7B base specifically with a PF of 8                    |  
| [ARC<sub>8</sub>-Encoder<sup>multi</sup>](link)|    Trained by sampling among these two decoders using 6.5B tokens for each one of them with a PF of 8                       |  



## Start training 

To pretrain or fine-tune your ARC-Encoder, review configuration examples in `config/`,
create your own YAML file, and launch training with:

```
uv run torchrun --nproc-per-node <number of gpus> -m train config/<your .yaml config file>
```


## Evaluation
Evaluate your models on QA benchmarks in two ways:
Use arguments from `embed_llm/generation/eval_context_comp.py`
Or run the complete evaluation as in the paper using `scripts/eval.sh`
Example command:
```
uv run python -m embed_llm.generation.eval_context_comp \
  --out_file eval.json \
  --max_seq_len 64 \
  --run_name <your_experiment_name> \
  --llm_name Llama3.1-8B \
  --embed_name Llama3.2-3B \
  --llm_number 0 \
  --n_icl_exs 5

```

## Folder Structure
| Folder                 | Description                                                                                                                                                                                                                                                            |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed_llm/data`       | Scripts for loading, tokenizing, and formatting datasets (continuation, reconstruction, fine-tuning).                                                                                                                                                                  |
| `embed_llm/generation` | Evaluation scripts for downstream tasks, including [LLMLingua2](https://arxiv.org/abs/2403.12968). Also includes long-context evaluations with [CEPED](https://arxiv.org/abs/2402.16617) and [LLaMA-2-7B-32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K). |
| `embed_llm/models`     | Modules defining ARC-Encoders and their paired decoder-only LLMs. Includes `wrapped_models_training.py` for **FSDP** training.                                                                                                                                         |
| `embed_llm/monitoring` | Tools for tracking metrics, progress, and logging during training.                                                                                                                                                                                                     |
| `embed_llm/retrieval`  | Scripts for embedding/retrieval using [NVEmbedv2](https://arxiv.org/abs/2405.17428).                                                                                                                                                                                   |
| `embed_llm/training`   | Utilities for distributed multi-GPU training.                                                                                                                                                                                                                          |
| `scripts`              | SLURM job launch scripts (evaluation/training) and dataset synthesis scripts under `synt_data/`.                                                                                                                                                                       |
##   Acknowledgments
This project uses code from:
- [mistral-finetune](https://github.com/mistralai/mistral-finetune)  (Apache License 2.0)
- [FID](https://github.com/facebookresearch/FiD) (Attribution-NonCommercial 4.0 International)