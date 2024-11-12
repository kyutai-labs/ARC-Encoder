import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Tuple, Any, Optional, Sequence
from functools import partial
import json

from embed_llm.models.embedding_modules import MLP_project
from embed_llm.retrieval.embeddings import encode_text
from embed_llm.data.data_loader import Batch
from embed_llm.models.args import MLPProjectArgs

# Mistral specifics
from embed_llm.models.mistral.transformer import Transformer as MistralTransformer
from embed_llm.models.mistral.tokenizer import load_tokenizer as load_mistraltokenizer
from embed_llm.models.mistral.generate import generate as mistral_generate

# Gemma specifics
from embed_llm.models.gemma.model import GemmaForCausalLM

# Llama specifics
from embed_llm.models.llama.model import Transformer as LlamaTransformer
from embed_llm.models.llama.generation import Llama


Models = Union[LlamaTransformer, MistralTransformer, GemmaForCausalLM]


def pad_and_convert_to_tensor(
        x: List[int], y: List[int],
        sizes: List[int], y_mask: List[bool],
        seq_len: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    final_x = torch.ones((len(sizes), seq_len), dtype=torch.long).cuda(
        non_blocking=True) * pad_id
    final_y = torch.ones((len(sizes), seq_len), dtype=torch.long).cuda(
        non_blocking=True) * pad_id
    final_mask = torch.zeros((len(sizes), seq_len),
                             dtype=torch.bool).cuda(non_blocking=True)
    # Pad the input and output sequences
    for i, size in enumerate(sizes):
        final_x[i, :size] = torch.tensor(x[i]).cuda()
        final_y[i, :size] = torch.tensor(y[i]).cuda()
        final_mask[i, :size] = torch.tensor(y_mask[i]).cuda()

    return final_x, final_y, final_mask


class EmbedAugModel(nn.Module):
    def __init__(self, model_name: str, mlp_project_args: MLPProjectArgs, llm: Models,
                 embed_model_name: str, embedding_model: Any, norm_wo_embeds: bool = False,
                 tokenizer: Any = None,
                 pad_token_id: Optional[int] = None, max_seq_len: Optional[int] = None):
        super().__init__()

        self.llm = llm
        self.embed_model_name = embed_model_name
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.llm_name = model_name.lower()

        if mlp_project_args.n_layers > 0:
            self.mlp_project = MLP_project(mlp_project_args)

        if 'mistral' in self.llm_name:
            self.forward = partial(self.forward_mistral,
                                   norm_wo_embeds=norm_wo_embeds)

        elif 'gemma' in self.llm_name:
            self.forward = partial(self.forward_gemma, pad_token_id=pad_token_id,
                                   norm_wo_embeds=norm_wo_embeds, max_seq_len=max_seq_len)

        elif 'llama' in self.llm_name:
            self.forward = partial(self.forward_llama, pad_token_id=pad_token_id,
                                   norm_wo_embeds=norm_wo_embeds, max_seq_len=max_seq_len)

        self.generate = None

    def forward_mistral(self, batch: Batch,  norm_wo_embeds: bool):

        x = torch.from_numpy(batch.x).cuda(non_blocking=True)
        y = torch.from_numpy(batch.y).cuda(non_blocking=True)
        y_mask = (
            torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
            if batch.y_mask is not None
            else None
        )
        with torch.no_grad():
            embeddings = encode_text(batch.texts, self.embed_model_name,
                                     self.embedding_model, query_embedding=False, device='cuda').cuda()

        if hasattr(self, 'mlp_project'):
            embeddings = self.mlp_project(embeddings)

        return self.llm(input_ids=x,
                        embeddings=embeddings,
                        seqlens=batch.sizes,
                        norm_wo_embeds=norm_wo_embeds,
                        ), y.detach(), y_mask.detach()

    def forward_llama(self, batch: Batch, max_seq_len: int, pad_token_id: int, norm_wo_embeds: bool):

        with torch.no_grad():
            embeddings = encode_text(batch.texts, self.embed_model_name,
                                     self.embedding_model, query_embedding=False, device='cuda').cuda()

        if hasattr(self, 'mlp_project'):
            embeddings = self.mlp_project(embeddings)

        x, y, y_mask = pad_and_convert_to_tensor(
            x=batch.x,
            y=batch.y,
            sizes=batch.sizes,
            y_mask=batch.y_mask,
            seq_len=max_seq_len,
            pad_id=pad_token_id,
        )
        return self.llm(tokens=x,
                        embeddings=embeddings,
                        norm_wo_embeds=norm_wo_embeds,
                        training=True
                        ), y.detach(), y_mask.detach()

    def forward_gemma(self, batch: Batch, max_seq_len: int, pad_token_id: int, norm_wo_embeds: bool):
        with torch.no_grad():
            embeddings = encode_text(batch.texts, self.embed_model_name,
                                     self.embedding_model, query_embedding=False, device='cuda').cuda()

        if hasattr(self, 'mlp_project'):
            embeddings = self.mlp_project(embeddings)

        x, y, y_mask = pad_and_convert_to_tensor(
            x=batch.x,
            y=batch.y,
            sizes=batch.sizes,
            y_mask=batch.y_mask,
            seq_len=max_seq_len,
            pad_id=pad_token_id,
        )

        att_mask = torch.full((max_seq_len, max_seq_len),
                              float("-inf")).cuda(non_blocking=True)
        att_mask = torch.triu(att_mask, diagonal=1)

        return self.llm(input_token_ids=x,
                        embeddings=embeddings,
                        mask=att_mask,
                        norm_wo_embeds=norm_wo_embeds,
                        ), y.detach(), y_mask.detach()

    # TODO Modifier pour que Llama et Gemma accepte format checpointing
    @staticmethod
    def load_inference_model(
            llm_path: str,
            mlp_path: str,
            device: str,
            model_name: str,
            max_batch_size: int,
            max_seq_len: int,
            embed_model_name: str,
            embedding_model: Any,
            norm_wo_embeds: bool = False,
            variant: Optional[str] = None):

        if 'mistral' in model_name.lower():
            model = MistralTransformer.from_folder(
                Path(llm_path), max_batch_size, max_seq_len, device, variant)
            model.eval()
            tokenizer = load_mistraltokenizer(
                Path(llm_path)).instruct_tokenizer.tokenizer
        elif 'gemma' in model_name.lower():
            # Implement a loading function for Gemma according to the format for checkpointing
            raise NotImplementedError('Model not yet implemented')
        elif 'llama' in model_name.lower():
            llm = Llama.build(
                ckpt_dir=llm_path,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                tokenizer_path=str(Path(llm_path) / 'tokenizer.model')
            )
            model = llm.model
            model.eval()
            tokenizer = llm.tokenizer
        else:
            raise NotImplementedError('Model not yet implemented')

        if mlp_path is not None:
            with open(Path(mlp_path) / "params.json", "r") as f:
                args = json.loads(f.read())
            mlp_project_args = MLPProjectArgs(**args)

            augmented_model = EmbedAugModel(
                model_name, mlp_project_args, model, embed_model_name, embedding_model, norm_wo_embeds, tokenizer)
            augmented_model.mlp_project.load_state_dict(
                torch.load(Path(mlp_path) / "model.pt"))

        augmented_model.eval()
        if 'mistral' in model_name.lower():
            augmented_model.generate = augmented_model.generate_mistral
        elif 'llama' in model_name.lower():
            augmented_model.generate = partial(
                augmented_model.generate_llama, llama_model=Llama(llm, tokenizer))
        elif 'gemma' in model_name.lower():
            augmented_model.generate = partial(
                augmented_model.generate_gemma, device=device)

        return augmented_model

    @torch.inference_mode()
    def generate_gemma(
            self,
            prompts: Union[str, Sequence[str]],
            text_conditioning:  Union[str, Sequence[str]],
            device: Any,
            max_tokens: int = 100,
            temperature: Union[float, None] = 0.95):

        embeddings = encode_text(text_conditioning, self.embed_model_name,
                                 self.embedding_model, query_embedding=False, device='cuda')

        return self.llm.generate(
            prompts=prompts,
            embeddings=embeddings,
            device=device,
            output_len=max_tokens,
            temperature=temperature,
        )

    @torch.inference_mode()
    def generate_mistral(self,
                         prompts: Union[str, Sequence[str]],
                         text_conditioning:  Union[str, Sequence[str]],
                         max_tokens: int = 100,
                         temperature: float = 0.6,
                         ):

        embeddings = encode_text(text_conditioning, self.embed_model_name,
                                 self.embedding_model, query_embedding=False, device='cuda')
        encoded_prompts = [self.tokenizer.encode(
            prompt, bos=True, eos=False) for prompt in prompts]
        eos_id = self.tokenizer.eos_token_id

        generated_tokens, logprobs = mistral_generate(
            encoded_prompts=encoded_prompts,
            embeddings=embeddings,
            model=self.llm,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
        )
        produced_text = [self.tokenizer.decode(
            generated_tokens[i]) for i in range(len(generated_tokens))]
        return produced_text

    @torch.inference_mode()
    def generate_llama(self,
                       llama_model: Llama,
                       prompts: Union[str, Sequence[str]],
                       text_conditioning:  Union[str, Sequence[str]],
                       max_tokens: int,
                       temperature: float = 0.6):

        embeddings = encode_text(text_conditioning, self.embed_model_name,
                                 self.embedding_model, query_embedding=False, device='cuda')
        prompt_tokens = llama_model.tokenizer.encode_batch(prompts)
        out_tokens = llama_model.generate(
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            max_gen_len=max_tokens,
            temperature=temperature,
        )
        return llama_model.tokenizer.decode_batch(out_tokens)
