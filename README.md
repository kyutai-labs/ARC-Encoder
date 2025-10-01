# Adaptable text Representations Compressor

This repository contains the code to reproduce most of the experiments of the paper \[BLABLA](bla). You can pretrain and fine-tune your own ARC-Encoder. Using the pretrained checkpoints we released, you can easily fine-tune an ARC-Encoder on specific datasets. 


## 📥 Installation

### 1️⃣ Clone this repository
```sh
git clone ...
```
Once cloned set import paths at `embed_llm/__init__.py`.

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

In the directory (<MODEL_PATH>) where you want to store the different backbone models for your ARC-Encoder and decoder, for Llama models you need to register on this website [Llama downloads](https://www.llama.com/llama-downloads/) and get the download url:
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


## 📚 Prepare datasets

For the fine-tuning dataset, please directly load our HuggingFace dataset at [BLABLA](blabla). Use the `load_datasets.ipynb` notebook to load the evaluation datasets. Then, you should run `retrieval/embeddings.py` (on GPUs if possible) to create embeddings of the Wikipedia text chunks and `retrieval/passage_retrieval.py` (you can stay on CPUs) to retrieve passages from Wikipedia for Natural Question and TRIVIAQA as done in the paper.




## 🏋️ Start training


FOR LONG CONTEXT PT TASK WITH CHUNKED CONTINUATION NOT IMPLEMENTED
### ⚙️ Customizing training configuration

## 🔮 Evaluation



## Acknowledgments

This project uses code from [mistral-finetune](https://github.com/mistralai/mistral-finetune) licensed under the Apache License 2.0 and [FID](https://github.com/facebookresearch/FiD) licensed under Attribution-NonCommercial 4.0 International.
