from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)
from dataclasses import fields
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import Cache
from PIL import Image
from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

from .modeling_llada import LLaDAModelLM
from .sampling import cosine_schedule, mask_by_random_topk
from transformers import PretrainedConfig

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

class MMadaConfig(PretrainedConfig):
    model_type = "mmada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        allowed_keys = [
            "vocab_size",
            "llm_vocab_size",
            "llm_model_path",
            "codebook_size",
            "num_vq_tokens",
            "num_new_special_tokens",
            "gradient_checkpointing",
            "new_vocab_size",
        ]

        for key in allowed_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])



class MMadaModelLM(LLaDAModelLM):
    config_class = MMadaConfig
    base_model_prefix = "model"
    def __init__(self, config: MMadaConfig, *args, **kwargs):
        print(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

        # # resize token embeddings
        # print(f"Resizing token embeddings to {config.new_vocab_size}")
        # self.resize_token_embeddings(config.new_vocab_size)

    @torch.no_grad()
    def interleave_generate(
        self,
        input_ids: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        text_cfg: float = 0.0,
        image_cfg: float = 3.5,
        noise_schedule: Callable = cosine_schedule,
        text_steps: int = 100,
        image_steps: int = 100,
        reserved_token_mapping: Dict = None,
        generator: torch.Generator = None,
        config=None,
        remasking = "low_confidence",
        text_temperature: float = 0.0,
        image_temperature: float = 1.0,
        **kwargs,
    ):
        
        input_ids = input_ids.unsqueeze(0)
        uncond_input_ids = uncond_input_ids.unsqueeze(0)
        # print(f"input_ids shape: {input_ids.shape}")
        # making a output ids with all masks
        uni_prompting = kwargs.get("uni_prompting", None)

        output_interleave_ids = torch.cat([
            torch.full((input_ids.shape[0], 1), reserved_token_mapping['<|soi|>']).to(input_ids.device),
            torch.full((input_ids.shape[0], config.model.mmada.num_vq_tokens), self.config.mask_token_id).to(input_ids.device),
            torch.full((input_ids.shape[0], 1), reserved_token_mapping['<|eoi|>']).to(input_ids.device),
            torch.full((input_ids.shape[0], 1), uni_prompting.text_tokenizer.bos_token_id).to(input_ids.device),
            torch.full((input_ids.shape[0], config.dataset.preprocessing.max_seq_length-1), self.config.mask_token_id).to(input_ids.device),
        ], dim=1)
        # print(f"output_interleave_ids shape: {output_interleave_ids.shape}")

        # combine input_ids and output_interleave_ids
        combined_input_ids = torch.cat([input_ids, output_interleave_ids], dim=1)
        # print(f"combined_input_ids shape: {combined_input_ids.shape}")

        # forward process  
        text_masked_indices = combined_input_ids[:, - config.dataset.preprocessing.max_seq_length:] == self.config.mask_token_id
        # print(f"text_masked_indices shape: {text_masked_indices.shape}")

        num_transfer_tokens = get_num_transfer_tokens(text_masked_indices, text_steps)

        image_generation_step_index = torch.linspace( text_steps // 4, text_steps - 1 , image_steps).round().int().to(input_ids.device)


        for i in range(text_steps):
            text_masked_indices = combined_input_ids[:, - config.dataset.preprocessing.max_seq_length:] == self.config.mask_token_id 

            if text_cfg or image_cfg:
                combined_uncond_input_ids = torch.cat([
                    uncond_input_ids,
                    combined_input_ids[:, input_ids.shape[1]:],
                ], dim=1)
                # print(f"combined uncond_input_ids shape: {combined_uncond_input_ids.shape}")

                logits = self(torch.cat([combined_input_ids, combined_uncond_input_ids], dim=0)).logits
                # print(f"logits shape: {logits.shape}")

                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # print(f"cond_logits shape: {cond_logits.shape}, uncond_logits shape: {uncond_logits.shape}")
                logits = cond_logits + text_cfg * (uncond_logits - cond_logits)

            else:
                raise ValueError("text_cfg and image_cfg cannot be both 0")

            text_logits = logits[:, - config.dataset.preprocessing.max_seq_length:]
            logits_with_noise = add_gumbel_noise(text_logits, temperature=text_temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            # print(x0.shape)

            if remasking == 'low_confidence':
                p = F.softmax(text_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # x0_p[:, - config.dataset.preprocessing.max_seq_length:] = -np.inf
            # print(f"text_masked_indices shape: {text_masked_indices.shape}, {x0.shape}, {combined_input_ids[:, - config.dataset.preprocessing.max_seq_length:].shape}")
            x0 = torch.where(text_masked_indices, x0, combined_input_ids[:,  - config.dataset.preprocessing.max_seq_length:])
            confidence = torch.where(text_masked_indices, x0_p, -np.inf)
            # print(confidence.shape)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            # print(f"transfer_index shape: {transfer_index.shape}, {transfer_index}")
            
            combined_input_ids[:,  - config.dataset.preprocessing.max_seq_length:][transfer_index] = x0[transfer_index]

            if i in image_generation_step_index: 
                num_vq_tokens = config.model.mmada.num_vq_tokens 
                input_ids_minus_lm_vocab_size = combined_input_ids[:, input_ids.shape[1] + 1: input_ids.shape[1] + 1 + num_vq_tokens].clone()
                input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == self.config.mask_token_id, self.config.mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer))

                image_logits = (1 + image_cfg) * cond_logits[:, input_ids.shape[1] + 1: input_ids.shape[1] + 1 + num_vq_tokens, len(uni_prompting.text_tokenizer):len(uni_prompting.text_tokenizer) + config.model.mmada.codebook_size] - image_cfg * uncond_logits[:, input_ids.shape[1] + 1: input_ids.shape[1] + 1 + num_vq_tokens, len(uni_prompting.text_tokenizer):len(uni_prompting.text_tokenizer) + config.model.mmada.codebook_size]

                # print(f"image_logits shape: {image_logits.shape}")

                probs = image_logits.softmax(dim=-1)
                sampled = probs.reshape(-1, image_logits.size(-1))
                sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*image_logits.shape[:-1]) # 1, 1024

                unknown_map = input_ids_minus_lm_vocab_size == self.config.mask_token_id
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

                ratio = 1.0 * (i + 1) / text_steps
                # print(f"ratio: {ratio}")
                mask_ratio = noise_schedule(torch.tensor(ratio))
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)

                selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
                mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
                mask_len = torch.max(torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len))

                temperature = image_temperature * (1.0 - ratio)
                masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
                combined_input_ids[:, input_ids.shape[1] + 1: input_ids.shape[1] + 1 + num_vq_tokens] = torch.where(masking, self.config.mask_token_id, sampled_ids + len(uni_prompting.text_tokenizer))

                input_ids_minus_lm_vocab_size = torch.where(masking, self.config.mask_token_id, sampled_ids)


        return_image_ids = sampled_ids
        return_text_ids = combined_input_ids[:, - config.dataset.preprocessing.max_seq_length:]
        # print(f"return_image_ids shape: {return_image_ids.shape}, return_text_ids shape: {return_text_ids.shape}")

        return return_image_ids, return_text_ids
                
                
                
                
                




                
                
                

        

    @torch.no_grad()
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            # print(attention_mask.shape,uncond_attention_mask.shape )
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cfg_attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (cfg_attention_mask[:, :, None] & cfg_attention_mask[:, None, :]).bool().unsqueeze(1)
                # print(f"attention bias shape: {attention_bias.shape}")
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids
    
    def forward_process(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            p_mask_lm=None,
            p_mask_mmu=None,
            answer_lengths=None,
            t2i_masks=None,
            answer_lengths_lm=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        if batch_size_t2i == 0:
            loss_t2i = torch.tensor(0.0, device=input_ids.device)
        else:
            # t2i loss
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
        
        # llada loss  
        masked_indices = input_ids == self.config.mask_token_id 
        masked_indices_lm = masked_indices[batch_size_t2i:batch_size_t2i + batch_size_lm]
        # if masked_indices_lm.numel() > 0:
        #     mask_counts = torch.sum(masked_indices_lm, dim=1)
        #     logging.info(f"[LM mask nums]: {mask_counts.cpu()}.")
        # else:
        #     logging.info("[LM mask nums] no LM sample.")
        masked_indices_mmu = masked_indices[-batch_size_mmu:]
        p_mask_lm = p_mask_lm.to(masked_indices_lm.device)
        p_mask_mmu = p_mask_mmu.to(masked_indices_mmu.device)       
        answer_lengths = answer_lengths.to(masked_indices_mmu.device) 
        loss_lm = F.cross_entropy(
            logits[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1, self.output_size),
            labels[batch_size_t2i:batch_size_t2i + batch_size_lm][masked_indices_lm].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_lm[masked_indices_lm]
        # print(f"logits lm shape: {logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape}")
        # loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])

        # # llm loss 
        # answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
        # loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        
        if answer_lengths_lm is not None:
            answer_lengths_lm = answer_lengths_lm.to(masked_indices_lm.device)
            loss_lm = torch.sum(loss_lm / answer_lengths_lm[masked_indices_lm]) / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0])  
        else:
            loss_lm = loss_lm.sum() / (logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[0] * logits[batch_size_t2i:batch_size_t2i + batch_size_lm].shape[1])


        loss_mmu = F.cross_entropy(
            logits[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1, self.output_size),
            labels[-batch_size_mmu:][masked_indices_mmu].contiguous().view(-1), ignore_index=-100, reduction='none'
            )/p_mask_mmu[masked_indices_mmu]
        loss_mmu = torch.sum(loss_mmu/answer_lengths[masked_indices_mmu]) / (logits[-batch_size_mmu:].shape[0])
        
        return logits, loss_t2i, loss_lm, loss_mmu


    def forward_process_separate(
        self,
        input_ids_t2i=None,
        labels_t2i=None,
        t2i_masks=None,
        input_ids_lm=None,
        labels_lm=None,
        p_mask_lm=None,
        attention_mask_lm=None,
        answer_lengths_lm=None,
        input_ids_mmu=None,
        labels_mmu=None,
        p_mask_mmu=None,
        attention_mask_mmu=None,
        answer_lengths_mmu=None,
        max_seq_length=128,
    ):
        # prepare sizes and defaults
        bs_t2i = input_ids_t2i.size(0) if input_ids_t2i is not None else 0
        bs_lm   = input_ids_lm.size(0)   if input_ids_lm   is not None else 0
        bs_mmu  = input_ids_mmu.size(0)  if input_ids_mmu  is not None else 0
        loss_t2i = input_ids_t2i.new_zeros(()) if bs_t2i>0 else torch.tensor(0., device=self.device)
        loss_lm   = input_ids_lm.new_zeros(())   if bs_lm>0   else torch.tensor(0., device=self.device)
        loss_mmu  = input_ids_mmu.new_zeros(())  if bs_mmu>0  else torch.tensor(0., device=self.device)
        logits_parts = []

        # t2i
        if bs_t2i > 0:
            att = (t2i_masks[:,:,None] & t2i_masks[:,None,:]).bool().unsqueeze(1)
            att = att.to(input_ids_t2i.device)
            logit = self(input_ids_t2i, attention_bias=att).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            loss_t2i = F.cross_entropy(
                logit[:, max_seq_length+1:].reshape(-1, self.output_size),
                labels_t2i[:, max_seq_length+1:].reshape(-1),
                ignore_index=-100,
            )

        # lm
        if bs_lm > 0:
            L = input_ids_lm.size(1)
            att = attention_mask_lm.bool()
            att = att.unsqueeze(1).unsqueeze(-1) & att.unsqueeze(1).unsqueeze(-2)
            att = att.to(input_ids_lm.device)
            logit = self(input_ids_lm, attention_bias=att).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            mask = input_ids_lm == self.config.mask_token_id
            p = p_mask_lm.to(mask.device)
            al = answer_lengths_lm.to(mask.device) if answer_lengths_lm is not None else None
            raw = F.cross_entropy(
                logit[mask].reshape(-1, self.output_size),
                labels_lm[mask].reshape(-1),
                ignore_index=-100, reduction='none'
            ) / p[mask]
            if al is not None:
                loss_lm = torch.sum(raw / al[mask]) / bs_lm
            else:
                loss_lm = raw.sum() / (bs_lm * L)

        # mmu
        if bs_mmu > 0:
            L = input_ids_mmu.size(1)
            att = attention_mask_mmu.bool()
            att = att.unsqueeze(1).unsqueeze(-1) & att.unsqueeze(1).unsqueeze(-2)
            att = att.to(input_ids_mmu.device)
            logit = self(input_ids_mmu, attention_bias=att).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            mask = input_ids_mmu == self.config.mask_token_id
            p = p_mask_mmu.to(mask.device)
            al = answer_lengths_mmu.to(mask.device)
            raw = F.cross_entropy(
                logit[mask].reshape(-1, self.output_size),
                labels_mmu[mask].reshape(-1),
                ignore_index=-100, reduction='none'
            ) / p[mask]
            loss_mmu = torch.sum(raw / al[mask]) / bs_mmu

        # concat logits
        # logits = torch.cat(logits_parts, dim=0) if logits_parts else None
        logits = None
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_process_separate_full_attn(
        self,
        input_ids_t2i=None,
        labels_t2i=None,
        t2i_masks=None,
        input_ids_lm=None,
        labels_lm=None,
        p_mask_lm=None,
        attention_mask_lm=None,
        answer_lengths_lm=None,
        input_ids_mmu=None,
        labels_mmu=None,
        p_mask_mmu=None,
        attention_mask_mmu=None,
        answer_lengths_mmu=None,
        max_seq_length=128,
    ):
        # prepare sizes and defaults
        bs_t2i = input_ids_t2i.size(0) if input_ids_t2i is not None else 0
        bs_lm   = input_ids_lm.size(0)   if input_ids_lm   is not None else 0
        bs_mmu  = input_ids_mmu.size(0)  if input_ids_mmu  is not None else 0
        loss_t2i = input_ids_t2i.new_zeros(()) if bs_t2i>0 else torch.tensor(0., device=self.device)
        loss_lm   = input_ids_lm.new_zeros(())   if bs_lm>0   else torch.tensor(0., device=self.device)
        loss_mmu  = input_ids_mmu.new_zeros(())  if bs_mmu>0  else torch.tensor(0., device=self.device)
        logits_parts = []

        # t2i
        if bs_t2i > 0:
            logit = self(input_ids_t2i).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            loss_t2i = F.cross_entropy(
                logit[:, :].reshape(-1, self.output_size),
                labels_t2i[:, :].reshape(-1),
                ignore_index=-100,
            )

        # lm
        if bs_lm > 0:
            L = input_ids_lm.size(1)
            logit = self(input_ids_lm).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            mask = input_ids_lm == self.config.mask_token_id
            p = p_mask_lm.to(mask.device)
            al = answer_lengths_lm.to(mask.device) if answer_lengths_lm is not None else None
            raw = F.cross_entropy(
                logit[mask].reshape(-1, self.output_size),
                labels_lm[mask].reshape(-1),
                ignore_index=-100, reduction='none'
            ) / p[mask]
            if al is not None:
                loss_lm = torch.sum(raw / al[mask]) / bs_lm
            else:
                loss_lm = raw.sum() / (bs_lm * L)

        # mmu
        if bs_mmu > 0:
            L = input_ids_mmu.size(1)
            logit = self(input_ids_mmu).logits
            logits_parts.append(logit); self.output_size = logit.size(-1)
            mask = input_ids_mmu == self.config.mask_token_id
            p = p_mask_mmu.to(mask.device)
            al = answer_lengths_mmu.to(mask.device)
            raw = F.cross_entropy(
                logit[mask].reshape(-1, self.output_size),
                labels_mmu[mask].reshape(-1),
                ignore_index=-100, reduction='none'
            ) / p[mask]
            loss_mmu = torch.sum(raw / al[mask]) / bs_mmu

        # concat logits
        # logits = torch.cat(logits_parts, dim=0) if logits_parts else None
        logits = None
        return logits, loss_t2i, loss_lm, loss_mmu

    def forward_t2i(
            self,
            input_ids, 
            labels,
            batch_size_t2i=0,
            max_seq_length=128,
            t2i_masks=None
            ):
        # attention bias, True for batch_size, 1, seq_len, seq_len  
        attention_bias = torch.ones(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
        attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
        attention_bias[:batch_size_t2i] = attention_bias_t2i
        logits = self(input_ids, attention_bias=attention_bias).logits 
        # logits = self(input_ids).logits
        self.output_size = logits.shape[-1]

        # print(f"logits shape: {logits.shape}") B, 359, vocab_size

        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )
        
        return loss_t2i





    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        # print(f"num_blocks: {num_blocks}, steps: {steps}")
        # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            # num_transfer_tokens = get_num_transfer_tokens(prompt_index, steps)
            # print(f"num_transfer_tokens: {num_transfer_tokens}, num_transfer_tokens.shape: {num_transfer_tokens.shape}")
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                

        return x

    @torch.no_grad()
    def mmu_generate_fast(self, idx=None, input_embeddings=None, max_new_tokens=128, steps=128,block_length=128, temperature=0.0, top_k=None, eot_token=None, cfg_scale=0.0, remasking='low_confidence', mask_id=126336, attention_mask=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        if attention_mask is not None and 0.0 in attention_mask:
            attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
            # print(f"attention_bias: {attention_bias}")
        else:
            attention_bias = None
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        batch_size = idx.shape[0]
        x = torch.full((batch_size, idx.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(self.device)
        x[:, :idx.shape[1]] = idx.clone()
        prompt_index = (x != mask_id)
        
        
        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        for num_block in range(num_blocks):
            block_mask_index = (x[:, idx.shape[1] + num_block * block_length: idx.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            for i in range(steps):
                mask_index = (x == mask_id) 
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self(x, attention_bias=attention_bias).logits
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, idx.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            if eot_token is not None:
                last_token_index_in_current_block = idx.shape[1] + (num_block + 1) * block_length - 1
                if last_token_index_in_current_block < x.shape[1]:
                    tokens_at_block_end = x[:, last_token_index_in_current_block]
                    if torch.all(tokens_at_block_end == eot_token):
                        break
        return x

    @torch.no_grad()
    def t2i_generate_decoding_stepwise(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            uncond_attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            seq_len=1024,
            mask_token_id = 126336,
            resolution = 512,
            codebook_size = 8192,
            vq_model = None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """

        # begin with all image token ids masked
        # 计算有多少个mask token
        mask_count = (input_ids == mask_token_id).sum().item()
        num_vq_tokens = seq_len
        num_new_special_tokens = 0
        uni_prompting = kwargs.get("uni_prompting", None)
        # print(f"config.model.mmada.llm_vocab_size: {config.model.mmada.llm_vocab_size}, {len(uni_prompting.text_tokenizer)}")
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id, mask_token_id, input_ids_minus_lm_vocab_size - len(uni_prompting.text_tokenizer) - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :resolution + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, resolution + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                attention_mask = torch.cat([attention_mask, uncond_attention_mask], dim=0)
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(model_input, attention_bias=attention_bias).logits 
                # print(f"logits.shape: {logits.shape}")
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]
            else:
                attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
                logits = self(input_ids, attention_bias=attention_bias).logits
                logits = logits[:, -(num_vq_tokens + 1):-1, len(uni_prompting.text_tokenizer) + num_new_special_tokens: len(uni_prompting.text_tokenizer) + num_new_special_tokens + codebook_size]

            # logits: 1, 1024, 8192
            # print(f"logits.shape: {logits.shape}")
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            # print(f"probs: {probs}, probs.shape: {probs.shape}, sampled: {sampled}, sampled.shape: {sampled.shape}")
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1]) # 1, 1024

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            # print(f"unknown_map.sum(dim=-1, keepdim=True): {unknown_map.sum(dim=-1, keepdim=True)}")
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            current_image_vq_indices = sampled_ids.clone()
            # print(f"current_image_vq_indices: {current_image_vq_indices}")
            current_image_vq_indices = torch.clamp(current_image_vq_indices, 0, 8192 - 1)
            current_image = vq_model.decode_code(current_image_vq_indices)
            images = torch.clamp((current_image + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = Image.fromarray(images[0]) 
            yield pil_images, f"Step {step + 1}/{timesteps}"
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # print(f"mask_len: {mask_len}, mask_len.shape: {mask_len.shape}")
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + len(uni_prompting.text_tokenizer)
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
            

        return sampled_ids
    

AutoConfig.register("mmada", MMadaConfig)
AutoModelForCausalLM.register(MMadaConfig, MMadaModelLM)
AutoModel.register(MMadaConfig, MMadaModelLM)
