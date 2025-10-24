#  Adaptable text Representations Compressor (ARC-Encoder)

[![Paper](https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.20535)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/kyutai/ARC_finetuning)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)


This repository contains the code to reproduce most of the experiments from the [paper](https://arxiv.org/abs/2510.20535) *ARC-Encoder: learning compressed text representations for large language models*.  
You can pretrain and fine-tune your own **ARC-Encoder**, or directly use our released checkpoints to fine-tune on specific datasets.  

---

## 🗂️ Table of Contents
- [Installation](#-installation)
- [Preparing Datasets](#-preparing-datasets)
- [Folder Structure](#-folder-structure)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Acknowledgments](#-acknowledgments)

---


## Installation

### 1️⃣ Clone this repository
```sh
git clone git@github.com:kyutai-labs/ARC-Encoder.git
```
***Once the repository is cloned, configure the directory paths in `embed_llm/__init__.py`***. Ensure that each path ends with a `/` to indicate a directory.
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
cd ARC-Encoder
pip install -e .
```

## Load models

### Pre-train ARC-Encoders
Pretrained ARC-Encoders are available on HuggingFace !!

| Models                | Specificities                                       | 
| :-------------------- | :-------------------------------------------------- | 
| [ARC<sub>8</sub>-Encoder<sup>L</sup>](https://huggingface.co/kyutai/ARC8_Encoder_Llama)| Trained on 2.6B tokens on Llama 3.1 8B base specifically with a pooling factor (PF) of 8.                                 |  
| [ARC<sub>8</sub>-Encoder<sup>M</sup>](https://huggingface.co/kyutai/ARC8_Encoder_Mistral)| Trained on 2.6B tokens on Mistral 7B base specifically with a PF of 8.                    |  
| [ARC<sub>8</sub>-Encoder<sup>multi</sup>](https://huggingface.co/kyutai/ARC8_Encoder_multi)|    Trained by sampling among these two decoders trained on 2.6B tokens with a PF of 8.                       |  

First, please use the following code to load them and format the folders accurately in your `<TMP_PATH>`, you just need to perform it once per model:
```python
from embed_llm.models.augmented_model import load_and_save_released_models

# ARC8_Encoder_multi, ARC8_Encoder_Llama or ARC8_Encoder_Mistral
load_and_save_released_models(ARC8_Encoder_Llama, hf_token=<HF_TOKEN>)
```
***Remark:*** This code snipet load from HF the model and then create at `<TMP_PATH>` the appropriate folder containing the checkpoint and additional necessary files to perform finetuning or evaluation with this codebase. To reduce the occupied memory space you can then delete the model from you HF cache. 

### Backbones
Create a directory `<MODEL_PATH>` to store the backbone models used by your ARC-Encoders and decoders.  
To reproduce the basic experiments with our released pretrained ARC-Encoders, you will need one of the following decoders: **Mistral 7B** or **Llama 3.1 8B**.  

For LLaMA models, register on the [LLaMA downloads](https://www.llama.com/llama-downloads/) page to obtain the download URLs.  
Ensure that the configuration file inside each model directory is named `params.json`, as it specifies the architectural parameters.  

If you are using pretrained ARC-Encoders, you may skip loading the Llama 3.2 3B weights, but the `params.json` and `tokenizer.model` files are still required.


```sh
# For Llama 3.2 3B, 
wget   url -P <MODEL_PATH>/Llama3.2-3B

# Depending on the decoder you want to test on

# For Mistral 7B
wget   https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar -P <MODEL_PATH>/mistral_7B

# For Llama 3.1 8B
wget   url -P <MODEL_PATH>/Llama3.1-8B
```

For additional experiments: 
```sh
# For Llama 2 7B Chat

wget   https://huggingface.co/meta-llama/Llama-2-7b-chat/resolve/main/consolidated.00.pth? -P <MODEL_PATH>/Llama2-7B-Chat
wget https://huggingface.co/meta-llama/Llama-2-7b-chat/resolve/main/tokenizer.model? -P <MODEL_PATH>/

echo '{"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000}' > <MODEL_PATH>/Llama2-7B-Chat/params.json

# For Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/model.safetensors? -P <MODEL_PATH>/Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/tokenizer.json? -P <MODEL_PATH>/Olmo7B
echo '{"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 50304}' > <MODEL_PATH>/Olmo7B/params.json
```


## Prepare datasets

Use the `load_datasets.ipynb` notebook to load the fine-tuning and evaluation datasets. 

For fine-tuning, load our Hugging Face dataset:
👉 [ARC Finetuning Dataset](https://huggingface.co/datasets/kyutai/ARC_finetuning). 

Then,  run the following scripts to prepare evaluation datasets still using `uv run python` (make sure to first set `DATA_PATH`):
- `retrieval/embeddings.py` — GPU recommended, creates embeddings for Wikipedia text chunks.
- `retrieval/passage_retrieval.py`— CPU fine, retrieves passages for Natural Questions and TriviaQA, as described in the paper.


## Folder Structure

| Folder                 | Description                                                                                                                                                                                                                                                            |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embed_llm/data`       | Scripts for loading, tokenizing, and formatting datasets (continuation, reconstruction, fine-tuning).                                                                                                                                                                  |
| `embed_llm/generation` | Evaluation scripts on downstream tasks for our ARC-Encoders and other baselines, including [LLMLingua2](https://arxiv.org/abs/2403.12968) and base models with or without retrieved contexts. Also includes long-context evaluations with [CEPED](https://arxiv.org/abs/2402.16617) and [LLaMA-2-7B-32k](https://huggingface.co/togethercomputer/LLaMA-2-7B-32K) in the `long_context/` folder. |
| `embed_llm/models`     | Modules defining ARC-Encoders and their paired decoder-only LLMs. Includes `wrapped_models_training.py` for **FSDP** training. `augmented_model.py` houses the pipeline wrapper which enables to load LLMs and the encoder, to perform a training forward as well as to generate text by formating data and encoding the text to compress. `enhanced_transformer.py` consists in the backbone architecture for either the decoder (with `forward` and `generate` functions) or the encoder (with `forward_embedder` function), both are initialized from the same module. `generate.py` implements the prefilling and decoding stage with KV-cache of our pipeline. Please note that compressed text representations extracted from the encoder are alternatively called **embeddings** or **comp_repr**.                                                                                                                                         |
| `embed_llm/monitoring` | Tools for tracking metrics, progress, and logging during training.                                                                                                                                                                                                  |
| `embed_llm/training`   | Utilities for distributed multi-GPU training.                                                                                                                                                                                                                          |
| `retrieval/`  | Scripts for embedding/retrieval using [NVEmbedv2](https://arxiv.org/abs/2405.17428).                                                                                                                                                                                   |
| `scripts/`              | SLURM job launch scripts (evaluation/training) and dataset synthesis scripts under `synt_data/`.      


**Remarks:** for each backbone model, a `params.json` file is required to define the configurations used to instantiate both the encoder and the decoder. Each trained ARC-Encoder checkpoint follows the structure described below, allowing flexible fine-tuning of individual components from different checkpoints. The evaluation scripts only require the experiment name and will automatically load and evaluate the latest checkpoint.

```
<Experiment Name>/
├── args.yaml # Contains full train arguments, only important if using a frozen encoder and training the MLP to monitor which initial ckpt was used for the encoder
├── ... # Metrics and monitoring files
└── checkpoints/
    ├── checkpoint_010000/ 
    ├── ...
    └── checkpoint_030000/ 
        ├── params.json # Always important to configure the architecture
        ├── embedder/consolidated.safetensors # Encoder checkpoint
        └── bridge_module/consolidated.safetensors # MLP projector checkpoint
```



## Start training 

To pretrain or fine-tune your ARC-Encoder, review configuration examples in `configs/`,
create your own YAML file, and launch training with:

```python
uv run torchrun --nproc-per-node <number of gpus> -m train configs/<your .yaml config file>
```


## Evaluation
Evaluate your models on QA benchmarks in two ways:
Use arguments from `embed_llm/generation/eval_context_comp.py`
Or run the complete evaluation as in the paper using `scripts/eval.sh`
Example command:
```python

uv run python -m embed_llm.generation.eval_context_comp \
  --out_file eval.json \
  --max_seq_len 64 \
  --run_name <your_experiment_name> \
  --llm_name Llama3.1-8B \
  --llm_number 0 \ # If ARC-Encoder_multi, to target if you want the first one trained on (for HF models Llama3.1-8B) or the second one
  --n_icl_exs 5

```


## License

Our code is released under the Apache License 2.0, except for the `retrieval/` directory, entirely independent from the other parts, which is licensed under the Attribution-NonCommercial 4.0 International license.


##   Acknowledgments and Citation
This project makes use of code snippets from:
- [mistral-finetune](https://github.com/mistralai/mistral-finetune)  (Apache License 2.0)
- [FID](https://github.com/facebookresearch/FiD) (Attribution-NonCommercial 4.0 International)


If you use ARC-Encoders for any of your projects please cite:

```bibtex
@misc{pilchen2025arcencoderlearningcompressedtext,
      title={ARC-Encoder: learning compressed text representations for large language models}, 
      author={Hippolyte Pilchen and Edouard Grave and Patrick Pérez},
      year={2025},
      eprint={2510.20535},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.20535}, 
}
```