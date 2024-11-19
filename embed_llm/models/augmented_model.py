import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Tuple, Any, Optional, Sequence
from functools import partial
import safetensors.torch
import json

from embed_llm.models.embedding_modules import MLP_project
from embed_llm.retrieval.embeddings import encode_text, get_embedder
from embed_llm.data.data_loader import Batch
from embed_llm.models.args import MLPProjectArgs, get_model_config

# Mistral specifics
from embed_llm.models.mistral.transformer import Transformer as MistralTransformer
from embed_llm.models.mistral.tokenizer import load_tokenizer as load_mistraltokenizer
from embed_llm.models.mistral.generate import generate as mistral_generate

# Gemma specifics
from embed_llm.models.gemma.model import GemmaForCausalLM, set_default_tensor_type

# Llama specifics
from embed_llm.models.llama.model import Transformer as LlamaTransformer
from embed_llm.models.llama.generation import Llama


Models = Union[LlamaTransformer, MistralTransformer, GemmaForCausalLM]

# Check for infinite or NaN values in your input
def check_data(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print("Found NaN or Inf in input data")
        return True
    return False


def pad_and_convert_to_tensor(
    x: List[int],
    y: List[int],
    sizes: List[int],
    y_mask: Union[List[bool], None],
    seq_len: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    final_x = (
        torch.ones((len(sizes), seq_len), dtype=torch.long).cuda(non_blocking=True)
        * pad_id
    )
    final_y = (
        torch.ones((len(sizes), seq_len), dtype=torch.long).cuda(non_blocking=True)
        * pad_id
    )
    final_mask = (
        torch.zeros((len(sizes), seq_len), dtype=torch.bool).cuda(non_blocking=True)
        if y_mask is not None
        else None
    )
    # Pad the input and output sequences
    ind = 0
    for i, size in enumerate(sizes):
        final_x[i, :size] = torch.tensor(x[ind:ind+size]).cuda(non_blocking=True)
        final_y[i, :size] = torch.tensor(y[ind:ind+size]).cuda(non_blocking=True)
        if y_mask is not None:
            final_mask[i, :size] = torch.tensor(y_mask[ind:ind+size]).cuda(non_blocking=True)
        ind += size
    return final_x, final_y, final_mask


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        llm_name: str,
        mlp_project_args: MLPProjectArgs,
        llm: Models,
        param_dtype: torch.dtype = torch.bfloat16,
        max_seq_len: Optional[int] = None,
        w_embeds: bool = True,
    ):
        super().__init__()
        self.add_module("llm", llm)
        self.llm_name = llm_name.lower()
        self.max_seq_len = max_seq_len
        self.w_embeds = w_embeds
        self.mlp_project_args = mlp_project_args

        if "mistral" in self.llm_name:
            self.forward = self.forward_mistral

        elif "gemma" in self.llm_name:
            self.forward = self.forward_gemma

        elif "llama" in self.llm_name:
            self.forward = self.forward_llama
            
        if mlp_project_args.n_layers > 0 and w_embeds:
            self.mlp_project = MLP_project(args=mlp_project_args, dtype=param_dtype)
        else:
            self.mlp_project = None
     


    def forward_mistral(
        self, 
        x: torch.Tensor, 
        seqlens: List[int], 
        embeddings: Optional[torch.Tensor] = None, 
    ) -> torch.Tensor:

        if self.mlp_project is not None:
            embeddings = self.mlp_project(embeddings)
            # check_data(embeddings)
            
        return self.llm.forward(input_ids=x, embeddings=embeddings, seqlens=seqlens)

    def forward_llama(
        self,
        x: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        seqlens: Optional[List[int]] = None,
    ) -> torch.Tensor:

        if self.mlp_project is not None:
            embeddings = self.mlp_project(embeddings)
        return self.llm.forward(tokens=x, embeddings=embeddings)

    def forward_gemma(
        self,
        x: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        seqlens: Optional[List[int]] = None,
    ) -> torch.Tensor:

        if self.mlp_project is not None:
            embeddings = self.mlp_project(embeddings)

        if self.w_embeds and embeddings is not None:
            att_mask = torch.full((self.max_seq_len + 1, self.max_seq_len + 1), float("-inf")).cuda(
                non_blocking=True
            )
        else:
            att_mask = torch.full((self.max_seq_len, self.max_seq_len), float("-inf")).cuda(
                non_blocking=True
            )
        att_mask = torch.triu(att_mask, diagonal=1)

        return self.llm.forward(
            input_token_ids=x, embeddings=embeddings, mask=att_mask, is_training=True
        )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        llm_name: str,
        mlp_project_args: MLPProjectArgs,
        embed_model_name: str,
        embedding_model: Any,
        param_dtype: torch.dtype = torch.bfloat16,
        tokenizer: Any = None,
        w_embeds: bool = True,
        pad_token_id: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.embed_model_name = embed_model_name
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.param_dtype = param_dtype
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.llm_name = llm_name.lower()
        self.mlp_project_args = mlp_project_args
        self.w_embeds = w_embeds
        self.model = None
        self.generate = None

    def get_model(self, llm: Any) -> nn.Module:
        return EmbedAugModel(
            llm_name=self.llm_name,
            mlp_project_args=self.mlp_project_args,
            llm=llm,
            max_seq_len=self.max_seq_len,
            param_dtype=self.param_dtype,
            w_embeds=self.w_embeds,
        )

    def store_model(self, model: nn.Module):
        self.model = model


    def prepare_forward(self, batch: Batch):

        if self.w_embeds:
            with torch.no_grad():
                embeddings = encode_text(
                    batch.texts,
                    self.embed_model_name,
                    self.embedding_model,
                    query_embedding=False,
                    device="cuda",
                ).type(self.param_dtype).detach()
        else:
            embeddings = None

        if "mistral" in self.llm_name:
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

        elif "llama" in self.llm_name or "gemma" in self.llm_name:
            x, y, y_mask = pad_and_convert_to_tensor(
                x=batch.x,
                y=batch.y,
                sizes=batch.sizes,
                y_mask=batch.y_mask,
                seq_len=self.max_seq_len,
                pad_id=self.pad_token_id,
            )

        seqlens = batch.sizes
        return x, y, y_mask, seqlens, embeddings

    # TODO Modifier pour que Llama et Gemma accepte format checpointing
    @staticmethod
    def load_inference_model(
        llm_path: str,
        ckpt_path: str,
        device: str,
        model_name: str,
        embed_model_name: str,
        max_batch_size: Optional[int] = 4,
        max_seq_len: Optional[int] = 512,
        variant: Optional[str] = None,
        w_embed: bool = True,
    ):
        lora_path = ckpt_path + '/' + model_name.lower() + '/consolidated/lora.safetensors'
        mlp_path = ckpt_path + '/' + 'MLP_projector'
        
        embedding_model = get_embedder(embed_model_name, device_map = device)
        
        if "mistral" in model_name.lower():
            llm = MistralTransformer.from_folder(
                folder = Path(llm_path), 
                max_batch_size = max_batch_size, 
                device =  device
            )
            # load LoRA
            llm.load_lora(Path(lora_path))

            llm.eval()
            tokenizer = load_mistraltokenizer(
                Path(llm_path)
            ).instruct_tokenizer.tokenizer
            
        elif "gemma" in model_name.lower():
            assert variant is not None, "Variant must be specified for Gemma"
            gemma_config = get_model_config(variant)

            gemma_config.dtype = "float32" if args.device == "cpu" else "float16"

            # Create the model and load the weights.
            device = torch.device(args.device)
            with set_default_tensor_type(gemma_config.get_dtype()):
                llm = GemmaForCausalLM(gemma_config)
                llm.load_weights(model_path=llm_path, lora_path=lora_path)

        elif "llama" in model_name.lower():
            llama = Llama.build(
                ckpt_dir=llm_path,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                tokenizer_path=str(Path(llm_path) / "tokenizer.model"),
                device = device
            )
            llm = llama.model

            # load LoRA
            llm.load_lora(Path(lora_path))

            llm.eval()
            tokenizer = llama.tokenizer
        else:
            raise NotImplementedError("Model not yet implemented")

        with open(Path(mlp_path) / "params.json", "r") as f:
            args = json.loads(f.read())
        mlp_project_args = MLPProjectArgs(**args)

        augmented_pipeline = EmbedAugPipeline(
            llm_name = model_name,
            mlp_project_args = mlp_project_args,
            embed_model_name = embed_model_name,
            embedding_model = embedding_model,
            tokenizer = tokenizer,
            max_seq_len=max_seq_len,
            pad_token_id=tokenizer.pad_id,
            w_embeds=w_embed,
        )
        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))
        
        if mlp_project_args.n_layers > 0:
            augmented_pipeline.model.mlp_project.load_state_dict(safetensors.torch.load_file(mlp_path+'/lora.safetensors')
            )

        augmented_pipeline.model = augmented_pipeline.model.to(device)
        augmented_pipeline.model.eval()

        if "mistral" in model_name.lower():
            augmented_pipeline.generate = partial(augmented_pipeline.generate_mistral, device = device)
        elif "llama" in model_name.lower():
            augmented_pipeline.generate = partial(
                augmented_pipeline.generate_llama, llama_model=Llama(llm, tokenizer), device = device
            )
        elif "gemma" in model_name.lower():
            augmented_pipeline.generate = partial(
                augmented_pipeline.generate_gemma, device=device
            )

        return augmented_pipeline

    @torch.inference_mode()
    def generate_gemma(
        self,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: Union[float, None] = 0.95,
        w_embeds: Optional[bool] = None,
        ):
        
        w_embeds = w_embeds if not None else self.w_embeds
        if w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None
        self.model.llm.w_embeds = w_embeds if w_embeds is not None else self.w_embeds
        return self.model.llm.generate(
            prompts=prompts,
            embeddings=embeddings,
            device=device,
            output_len=max_tokens,
            temperature=temperature,
            tokenizer=self.tokenizer,
        )

    @torch.inference_mode()
    def generate_mistral(
        self,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: float = 0.6,
        w_embeds: Optional[bool] = None,
    ):
        w_embeds = w_embeds if not None else self.w_embeds
        if w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None

        encoded_prompts = [
            self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts
        ]
        eos_id = self.tokenizer.eos_id
        self.model.llm.w_embeds = w_embeds if w_embeds is not None else self.w_embeds
        generated_tokens, logprobs = mistral_generate(
            encoded_prompts=encoded_prompts,
            embeddings=embeddings,
            model=self.model.llm,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
        )
        produced_text = [
            self.tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]
        
        final_texts = []
        for text in produced_text:
            if '\n\n' in text:
                text = text.split('\n\n')[0]
            final_texts.append(text)
        return final_texts

    @torch.inference_mode()
    def generate_llama(
        self,
        llama_model: Llama,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: float = 0.6,
        w_embeds: Optional[bool] = None,
    ):
        w_embeds = w_embeds if not None else self.w_embeds
        if w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None
        self.model.llm.w_embeds = w_embeds if w_embeds is not None else self.w_embeds
        prompt_tokens = llama_model.tokenizer.encode_batch(s = prompts, bos = True, eos = False)
        out_tokens, logprobs = llama_model.generate(
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            max_gen_len=max_tokens,
            temperature=temperature,
            logprobs=True,
        )
        produced_text = llama_model.tokenizer.decode_batch(out_tokens)
        final_texts = []
        for text in produced_text:
            if '\n\n' in text:
                text = text.split('\n\n')[0]
            final_texts.append(text)
        return final_texts
