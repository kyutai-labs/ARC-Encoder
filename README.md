# Adaptable text Representations Compressor

This repository contains the code to reproduce most of the experiments of the paper \[BLABLA](bla). You can pretrain and fine-tune your own ARC-Encoder. Using the pretrained checkpoints we released, you can easily fine-tune an ARC-Encoder on specific datasets. 


## 📥 Installation

### 1️⃣ Clone this repository
```sh
git clone ...
```

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
In the directory (<PREFIX_PATH>) where you want to store the different backbone models for your ARC-Encoder and decoder, for Llama models you need to register on this website [Llama downloads](https://www.llama.com/llama-downloads/) and get the download url:
```
# For Mistral 7B
wget   https://models.mistralcdn.com/mistral-7b-v0-1/mistral-7B-v0.1.tar -P <PREFIX_PATH>/mistral_7B

# For Llama3.2 3B
wget   url -P <PREFIX_PATH>/Llama3.2-3B

# For Llama3.1 8B
wget   url -P <PREFIX_PATH>/Llama3.1-8B

# For Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/model.safetensors? -P <PREFIX_PATH>/Olmo7B
wget https://huggingface.co/allenai/OLMo-7B/resolve/main/tokenizer.json? -P <PREFIX_PATH>/Olmo7B
echo '{"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 50304}' > <PREFIX_PATH>/Olmo7B/params.json
```


## 📚 Prepare datasets


## 🏋️ Start training

### ⚙️ Customizing training configuration

## 🔮 Evaluation



## Acknowledgments

This project uses code from [mistral-finetune](https://github.com/mistralai/mistral-finetune) licensed under the Apache License 2.0.
