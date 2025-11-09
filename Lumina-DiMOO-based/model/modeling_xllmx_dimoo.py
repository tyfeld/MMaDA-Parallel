import functools
import logging
import math
from typing import List, Dict, Tuple, Optional
import torch.nn.functional as F
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig
from .modeling_llada import LLaDAModelLM
from .configuration_llada import LLaDAConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

__all__ = ["LLaDAForMultiModalGeneration"]


def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1
    return attention_mask


class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    
    IMAGE_START_TOKEN = 126349
    IMAGE_END_TOKEN = 126350
    ANSWER_START_TOKEN = 126354
    ANSWER_END_TOKEN = 126355
    BREAKLINE_TOKEN = 126084
    MASK_TOKEN = 126336
    PAD_TOKEN = 126339
    
    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        print(f"Initializing LLaDAForMultiModalGeneration with config: {config}")
        super().__init__(config, *args, **kwargs)
        self._debug_step = 0
    
    def forward(
        self, 
        input_ids=None, 
        labels=None, 
        infer=False, 
        use_cache=False, 
        return_dict=False,
        compute_separate_losses=True,
        t=None,
        text_coeff=1.0,
        image_coeff=1.0,
    ):
        if infer:
            input_ids = input_ids.tolist()
        
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids]
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)
        attention_bias = (attention_mask[:, :, None] & attention_mask[:, None, :]).bool().unsqueeze(1)
        
        output = LLaDAModelLM.forward(
            self, 
            input_ids=input_ids, 
            attention_bias=attention_bias, 
            use_cache=use_cache
        )
        
        if infer:
            return output
        
        if labels is None:
            if return_dict:
                return {'logits': output.logits}
            else:
                return output.logits
        
        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)
        
        logits = output.logits
        batch_size = logits.shape[0]
        
        unscaled_loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]), 
            labels.contiguous().view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, -1)
        
        valid_mask = (labels != -100)
        
        if valid_mask.sum() > 0:
            interleave_loss = unscaled_loss[valid_mask].mean()
        else:
            interleave_loss = torch.tensor(0.0, device=self.device)
        
        if compute_separate_losses:
            self._debug_step += 1
            debug_this_step = (self._debug_step <= 3)
            
            if debug_this_step:
                print(f"\n{'='*80}")
                print(f"DEBUG Step {self._debug_step}")
                print(f"{'='*80}")
            
            text_loss_list = []
            image_loss_list = []
            
            for b in range(batch_size):
                answer_start_positions = (input_ids[b] == self.ANSWER_START_TOKEN).nonzero(as_tuple=True)[0]
                
                if len(answer_start_positions) == 0:
                    continue
                
                answer_start = answer_start_positions[0].item()
                
                answer_end_in_search = (input_ids[b, answer_start:] == self.ANSWER_END_TOKEN).nonzero(as_tuple=True)[0]
                if len(answer_end_in_search) > 0:
                    answer_end = answer_start + answer_end_in_search[0].item()
                else:
                    answer_end = original_lengths[b]
                
                answer_region_input = input_ids[b, answer_start:answer_end]
                image_start_in_answer = (answer_region_input == self.IMAGE_START_TOKEN).nonzero(as_tuple=True)[0]
                
                if len(image_start_in_answer) > 0:
                    image_start_pos = answer_start + image_start_in_answer[0].item()
                    image_end_search = input_ids[b, image_start_pos:]
                    image_end_in_search = (image_end_search == self.IMAGE_END_TOKEN).nonzero(as_tuple=True)[0]
                    
                    if len(image_end_in_search) > 0 :
                        image_end_pos = image_start_pos + image_end_in_search[0].item()
                        
                        for pos in range(image_start_pos + 1, image_end_pos):
                            if input_ids[b, pos] != self.BREAKLINE_TOKEN:
                                image_loss_list.append(unscaled_loss[b, pos])
                        
                        for pos in range(image_end_pos + 1, answer_end):
                            if labels[b, pos] != -100:
                                text_loss_list.append(unscaled_loss[b, pos])
                else:
                    for pos in range(answer_start + 1, answer_end):
                        if labels[b, pos] != -100:
                            text_loss_list.append(unscaled_loss[b, pos])
            
            if debug_this_step:
                print(f"Total text_loss_list length: {len(text_loss_list)}")
                print(f"Total image_loss_list length: {len(image_loss_list)}")
                if len(text_loss_list) > 0:
                    non_zero_text = [l.item() for l in text_loss_list if l.item() > 0]
                    print(f"Non-zero text losses count: {len(non_zero_text)}/{len(text_loss_list)}")
                    print(f"Sample non-zero text losses: {non_zero_text[:5]}")
                if len(image_loss_list) > 0:
                    non_zero_image = [l.item() for l in image_loss_list if l.item() > 0]
                    print(f"Non-zero image losses count: {len(non_zero_image)}/{len(image_loss_list)}")
                    print(f"Sample non-zero image losses: {non_zero_image[:5]}")
                print(f"{'='*80}\n")
            
            if len(text_loss_list) > 0:
                text_loss = torch.stack(text_loss_list).mean()
            else:
                text_loss = torch.tensor(0.0, device=self.device)
            
            if len(image_loss_list) > 0:
                image_loss = torch.stack(image_loss_list).mean()
            else:
                image_loss = torch.tensor(0.0, device=self.device)
            
            if t is not None and len(text_loss_list) > 0:
                text_loss = text_loss / t.mean().clamp(min=0.01)
            
            if return_dict:
                return {
                    'logits': logits,
                    'loss': interleave_loss,
                    'interleave_loss': interleave_loss,
                    'text_loss': text_loss,
                    'image_loss': image_loss,
                    'labels': labels,
                }
            else:
                return interleave_loss, {
                    'text_loss': text_loss,
                    'image_loss': image_loss,
                    'interleave_loss': interleave_loss,
                }
        else:
            if return_dict:
                return {'logits': logits, 'loss': interleave_loss, 'labels': labels}
            else:
                return interleave_loss
    
    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules


    def get_checkpointing_wrap_module_list(self) -> List:
        return list(self.model.transformer.blocks)