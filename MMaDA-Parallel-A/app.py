import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import torch
import math
from PIL import Image
from transformers import AutoTokenizer
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    decode_vq_to_image, calculate_vq_params,
    generate_crop_size_list, var_center_crop, add_break_line,
    encode_img_with_breaks, encode_img_with_paint
)
from utils.prompt_utils import generate_text_image_to_text_image_prompt
import torch.nn.functional as F

MODEL = None
TOKENIZER = None
VQVAE = None
DEVICE = None
CURRENT_MODEL_PATH = None

SPECIAL_TOKENS = {
    "mask_token": 126336,
    "newline_token": 126084,
    "image_token_offset": 126356,
    "answer_start": 126354,
    "answer_end": 126355,
    "boi": 126349,
    "eoi": 126350,
    "uncondition": 126351
}

SYSTEM_PROMPT = "Generate an image applying the following editing instruction based on the original image."

def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)

def add_gumbel_noise(logits, temperature=1.0, generator=None):
    if temperature == 0:
        return logits
    
    if generator is not None:
        uniform_noise = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator)
    else:
        uniform_noise = torch.rand_like(logits)
    
    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
    return logits + temperature * gumbel_noise

def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    if generator is not None:
        noise = torch.randn(probs.shape, dtype=probs.dtype, device=probs.device, generator=generator)
    else:
        noise = torch.randn_like(probs)
    
    confidence = torch.log(probs + 1e-10) + temperature * noise
    sorted_confidence, sorted_indices = torch.sort(confidence, dim=-1, descending=False)
    
    if isinstance(mask_len, torch.Tensor):
        mask_len_clamped = torch.clamp(mask_len, 0, probs.shape[-1] - 1)
        mask_len_clamped = mask_len_clamped.long().squeeze(-1)
    else:
        mask_len_clamped = int(mask_len)
    
    if isinstance(mask_len_clamped, torch.Tensor):
        batch = probs.shape[0]
        masking = torch.zeros_like(probs, dtype=torch.bool, device=probs.device)
        for b in range(batch):
            k = mask_len_clamped[b].item()
            if k <= 0:
                continue
            low_idx = sorted_indices[b, :k]
            masking[b, low_idx] = True
    else:
        k = mask_len_clamped
        if k <= 0:
            masking = torch.zeros_like(probs, dtype=torch.bool, device=probs.device)
        else:
            low_idx = sorted_indices[:, :k]
            masking = torch.zeros_like(probs, dtype=torch.bool, device=probs.device)
            batch = probs.shape[0]
            for b in range(batch):
                masking[b, low_idx[b]] = True
    
    return masking

def get_num_transfer_tokens(text_masked_indices, text_steps):
    batch_size = text_masked_indices.shape[0]
    initial_masks = text_masked_indices.sum(dim=1)
    
    num_transfer = torch.zeros(batch_size, text_steps, dtype=torch.long, device=text_masked_indices.device)
    
    for b in range(batch_size):
        total_masks = initial_masks[b].item()
        remaining = total_masks
        
        for step in range(text_steps):
            ratio = (step + 1) / text_steps
            target_remaining = int(total_masks * (1 - ratio))
            tokens_to_unmask = max(0, remaining - target_remaining)
            num_transfer[b, step] = tokens_to_unmask
            remaining -= tokens_to_unmask
    
    return num_transfer

@torch.no_grad()
def decode_text_with_masks(combined_input_ids, text_start, text_end, tokenizer, mask_token):
    text_ids = combined_input_ids[0, text_start:text_end].cpu().tolist()
    
    result_parts = []
    consecutive_masks = 0
    
    for token_id in text_ids:
        if token_id == mask_token:
            consecutive_masks += 1
        else:
            if consecutive_masks > 0:
                if consecutive_masks <= 10:
                    result_parts.append("▓" * consecutive_masks)
                else:
                    result_parts.append(f"▓▓▓▓▓[...{consecutive_masks - 5} more]")
                consecutive_masks = 0
            
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if token_text.strip() or token_text in [' ', '\n', '\t']:
                    result_parts.append(token_text)
            except:
                result_parts.append(f"[{token_id}]")
    
    if consecutive_masks > 0:
        if consecutive_masks <= 10:
            result_parts.append("▓" * consecutive_masks)
        else:
            result_parts.append(f"▓▓▓▓▓[...{consecutive_masks - 5} more]")
    
    return "".join(result_parts)

@torch.no_grad()
def generate_ti2ti_stepwise(
    model, input_ids, text_start, text_end, image_start, seq_len, newline_every,
    text_steps=100, temperature=1.0, text_temperature=0.7, cfg_scale=0.0, cfg_img=4.0,
    uncon_text=None, uncon_image=None, tokenizer=None, remasking='low_confidence',
    noise_schedule=cosine_schedule, generator=None, text_vocab_size=126356,
    codebook_size=8192, vqvae=None, image_height=512, image_width=512,
):
    device = input_ids.device
    MASK_TOKEN = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    
    combined_input_ids = input_ids.clone()
    num_vq_tokens = seq_len
    total_image_len = seq_len + seq_len // newline_every
    image_end = image_start + total_image_len
    
    text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN
    num_transfer_tokens = get_num_transfer_tokens(text_masked_indices, text_steps)
    
    image_generation_step_indices = torch.linspace(
        0, text_steps - 1, int(text_steps * 0.3)
    ).round().int().tolist()
    
    image_position_mapping = []
    for i in range(image_start, image_end):
        if combined_input_ids[0, i] != NEW_LINE:
            image_position_mapping.append(i)
    
    batch_size = combined_input_ids.shape[0]
    initial_text_display = decode_text_with_masks(combined_input_ids, text_start, text_end, tokenizer, MASK_TOKEN)
    last_generated_image = None
    
    yield 0, initial_text_display, None, f"Step 0/{text_steps}"
    
    for step in range(text_steps):
        cond_logits = model(combined_input_ids, infer=True, use_cache=False).logits
        
        text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN
        
        if text_masked_indices.sum() > 0:
            text_logits = cond_logits[:, text_start:text_end, :]
            logits_with_noise = add_gumbel_noise(text_logits, temperature=text_temperature, generator=generator)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if remasking == 'low_confidence':
                p = F.softmax(text_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                if generator is not None:
                    x0_p = torch.rand(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                x0_p = torch.ones_like(x0, dtype=torch.float)
            
            x0 = torch.where(text_masked_indices, x0, combined_input_ids[:, text_start:text_end])
            confidence = torch.where(text_masked_indices, x0_p, float('-inf'))
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, step].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            
            combined_input_ids[:, text_start:text_end][transfer_index] = x0[transfer_index]
        
        if step in image_generation_step_indices:
            vq_tokens_list = []
            mask_positions = []
            for idx, pos in enumerate(image_position_mapping):
                token = combined_input_ids[0, pos].item()
                if token == MASK_TOKEN:
                    vq_tokens_list.append(-1)
                    mask_positions.append(idx)
                else:
                    vq_token = token - text_vocab_size
                    vq_token = max(0, min(vq_token, codebook_size - 1))
                    vq_tokens_list.append(vq_token)
            
            vq_tokens_tensor = torch.tensor(vq_tokens_list, device=device).unsqueeze(0)
            unknown_map = vq_tokens_tensor == -1
            
            cond_image_logits_list = []
            for pos in image_position_mapping:
                cond_image_logits_list.append(
                    cond_logits[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size]
                )
            cond_vq_logits = torch.cat(cond_image_logits_list, dim=1)
            
            if (cfg_scale > 0.0 and uncon_text is not None) or (cfg_img > 0.0 and uncon_image is not None):
                if uncon_text is None:
                    combined_uncond_text = combined_input_ids.clone()
                else:
                    combined_uncond_text = combined_input_ids.clone()
                    prefix_len = uncon_text.shape[1]
                    combined_uncond_text[:, :prefix_len] = uncon_text.to(device)
                
                if uncon_image is None:
                    combined_uncond_img = combined_input_ids.clone()
                else:
                    combined_uncond_img = combined_input_ids.clone()
                    prefix_len_img = uncon_image.shape[1]
                    combined_uncond_img[:, :prefix_len_img] = uncon_image.to(device)
                
                uncond_text_logits_full = model(combined_uncond_text, infer=True, use_cache=False).logits
                uncond_img_logits_full = model(combined_uncond_img, infer=True, use_cache=False).logits
                
                uncond_text_vq_list = []
                uncond_img_vq_list = []
                for pos in image_position_mapping:
                    uncond_text_vq_list.append(
                        uncond_text_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size]
                    )
                    uncond_img_vq_list.append(
                        uncond_img_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size]
                    )
                
                uncond_text_vq_logits = torch.cat(uncond_text_vq_list, dim=1)
                uncond_img_vq_logits = torch.cat(uncond_img_vq_list, dim=1)
            else:
                uncond_text_vq_logits = torch.zeros_like(cond_vq_logits)
                uncond_img_vq_logits = torch.zeros_like(cond_vq_logits)
            
            image_logits = cond_vq_logits
            if cfg_scale != 0.0:
                image_logits = image_logits + cfg_scale * (cond_vq_logits - uncond_text_vq_logits)
            if cfg_img != 0.0:
                image_logits = image_logits + cfg_img * (cond_vq_logits - uncond_img_vq_logits)
            
            probs = F.softmax(image_logits, dim=-1)
            
            if temperature == 0:
                sampled_ids = probs.argmax(dim=-1)
            else:
                sampled = probs.reshape(-1, image_logits.size(-1))
                if generator is not None:
                    sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*image_logits.shape[:-1])
                else:
                    sampled_ids = torch.multinomial(sampled, 1)[:, 0].view(*image_logits.shape[:-1])
            
            sampled_ids = torch.where(unknown_map, sampled_ids, vq_tokens_tensor)
            sampled_ids = torch.clamp(sampled_ids, 0, codebook_size - 1)
            
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
            high_val = torch.finfo(selected_probs.dtype).max
            selected_probs = torch.where(unknown_map, selected_probs, high_val)
            
            ratio = 1.0 * (step + 1) / text_steps
            mask_ratio = noise_schedule(torch.tensor(ratio, device=device))
            unknown_counts = unknown_map.sum(dim=-1, keepdim=True)
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(device)
            mask_len = torch.max(torch.tensor([1], device=device), torch.min(unknown_counts - 1, mask_len.to(device).long()))
            if mask_len.ndim == 1:
                mask_len = mask_len.unsqueeze(1)
            
            img_temp = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, img_temp, generator=generator)
            final_vq_tokens = torch.where(masking, torch.tensor(-1, device=device), sampled_ids)
            
            for idx, pos in enumerate(image_position_mapping):
                v = final_vq_tokens[0, idx].item()
                if v == -1:
                    combined_input_ids[0, pos] = MASK_TOKEN
                else:
                    combined_input_ids[0, pos] = int(v + text_vocab_size)
            
            try:
                decoded_image = decode_vq_to_image(
                    sampled_ids, None, None, image_height, image_width, vqvae
                )
                
                masked_positions_bool = masking[0]
                if masked_positions_bool.sum() > 0:
                    from PIL import ImageDraw
                    decoded_image = decoded_image.copy()
                    draw = ImageDraw.Draw(decoded_image, 'RGBA')
                    
                    vae_scale = 2 ** (len(VQVAE.config.block_out_channels) - 1)
                    token_h = image_height // vae_scale
                    token_w = image_width // vae_scale
                    pixel_h = image_height // token_h
                    pixel_w = image_width // token_w
                    
                    masked_indices = torch.where(masked_positions_bool)[0].cpu().tolist()
                    for masked_idx in masked_indices:
                        token_row = masked_idx // token_w
                        token_col = masked_idx % token_w
                        
                        y1 = token_row * pixel_h
                        x1 = token_col * pixel_w
                        y2 = y1 + pixel_h
                        x2 = x1 + pixel_w
                        
                        draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 120))
                
                last_generated_image = decoded_image
            except Exception as e:
                pass
        
        text_display = decode_text_with_masks(combined_input_ids, text_start, text_end, tokenizer, MASK_TOKEN)
        text_masks_remaining = (combined_input_ids[:, text_start:text_end] == MASK_TOKEN).sum().item()
        text_progress = (1 - text_masks_remaining / (text_end - text_start)) * 100
        
        status_msg = f"Step {step + 1}/{text_steps} | Text: {text_progress:.1f}%"
        if step in image_generation_step_indices:
            image_masks_remaining = sum(1 for pos in image_position_mapping if combined_input_ids[0, pos] == MASK_TOKEN)
            image_progress = (1 - image_masks_remaining / num_vq_tokens) * 100
            status_msg += f" | Image: {image_progress:.1f}%"
        
        if step % 5 == 0 or step in image_generation_step_indices or step == text_steps - 1:
            yield step + 1, text_display, last_generated_image, status_msg
    
    final_text_display = decode_text_with_masks(combined_input_ids, text_start, text_end, tokenizer, MASK_TOKEN)
    
    if last_generated_image is not None:
        final_image = last_generated_image
    else:
        final_vq_tokens = []
        final_mask_positions = []
        for idx, pos in enumerate(image_position_mapping):
            token = combined_input_ids[0, pos].item()
            if token != MASK_TOKEN:
                vq_token = token - text_vocab_size
                vq_token = max(0, min(vq_token, codebook_size - 1))
                final_vq_tokens.append(vq_token)
            else:
                final_vq_tokens.append(codebook_size // 2)
                final_mask_positions.append(idx)
        
        vq_tensor = torch.tensor(final_vq_tokens, dtype=torch.long, device=device).unsqueeze(0)
        final_image = decode_vq_to_image(vq_tensor, None, None, image_height, image_width, vqvae)
        
        if final_mask_positions:
            from PIL import ImageDraw
            final_image = final_image.copy()
            draw = ImageDraw.Draw(final_image, 'RGBA')
            
            vae_scale = 2 ** (len(VQVAE.config.block_out_channels) - 1)
            token_h = image_height // vae_scale
            token_w = image_width // vae_scale
            pixel_h = image_height // token_h
            pixel_w = image_width // token_w
            
            for masked_idx in final_mask_positions:
                token_row = masked_idx // token_w
                token_col = masked_idx % token_w
                
                y1 = token_row * pixel_h
                x1 = token_col * pixel_w
                y2 = y1 + pixel_h
                x2 = x1 + pixel_w
                
                draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 120))
    
    yield text_steps, final_text_display, final_image, "✓ Complete"

def load_model_and_vae(model_path, vae_path):
    global MODEL, TOKENIZER, VQVAE, DEVICE, CURRENT_MODEL_PATH
    
    if MODEL is not None and CURRENT_MODEL_PATH == model_path:
        return f"Model already loaded: {model_path}"
    
    try:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        MODEL = LLaDAForMultiModalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        MODEL.eval()
        
        from diffusers import VQModel
        VQVAE = VQModel.from_pretrained(vae_path, subfolder="vqvae").to(DEVICE)
        
        CURRENT_MODEL_PATH = model_path
        
        return f"✓ Model loaded | Device: {DEVICE}"
    except Exception as e:
        MODEL = None
        TOKENIZER = None
        VQVAE = None
        CURRENT_MODEL_PATH = None
        return f"✗ Failed: {str(e)}"

def generate_wrapper(
    input_image, prompt_text, model_path, vae_path, height, width,
    text_steps, text_gen_length, text_block_length, cfg_scale, cfg_img,
    temperature, text_temperature, remasking_strategy, painting_mode,
    mask_h_ratio, mask_w_ratio, seed,
):
    global MODEL, TOKENIZER, VQVAE, DEVICE
    
    if MODEL is None or TOKENIZER is None or VQVAE is None:
        load_status = load_model_and_vae(model_path, vae_path)
        if "Failed" in load_status:
            yield "", None, load_status
            return
    
    if input_image is None:
        yield "", None, "✗ No input image"
        return
    
    if seed != 0:
        torch.manual_seed(seed)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    else:
        generator = None
    
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]
    
    try:
        input_prompt, uncon_text = generate_text_image_to_text_image_prompt(
            prompt_text, SYSTEM_PROMPT
        )
        
        prompt_ids = TOKENIZER(input_prompt)["input_ids"]
        uncon_text_ids = TOKENIZER(uncon_text)["input_ids"]
        
        img = input_image.convert("RGB")
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        img = var_center_crop(img, crop_size_list=crop_size_list)
        
        input_img_token = encode_img_with_breaks(img, VQVAE)
        
        con_input_list = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
        uncon_input_image = prompt_ids
        
        vae_scale = 2 ** (len(VQVAE.config.block_out_channels) - 1)
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
            height, width, vae_scale
        )
        
        text_mask_tokens = [MASK] * text_gen_length
        
        if painting_mode:
            img_mask_token, img_vis = encode_img_with_paint(
                img, vqvae=VQVAE, mask_h_ratio=mask_h_ratio, 
                mask_w_ratio=mask_w_ratio, mask_mode=painting_mode
            )
        else:
            img_mask_token = add_break_line(
                [MASK] * seq_len, token_grid_height, token_grid_width, 
                new_number=NEW_LINE
            )
        
        end_token_ids = TOKENIZER("</answer>", add_special_tokens=False).input_ids
        pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + text_mask_tokens + end_token_ids
        
        code_start = len(con_input_list)
        image_start = len(con_input_list) + 2
        image_end = image_start + len(img_mask_token)
        text_start = image_end + 1
        text_end = text_start + text_gen_length
        
        full_input_ids = con_input_list + pred_token
        con_input = torch.tensor(full_input_ids, device=DEVICE).unsqueeze(0)
        uncon_input_text_tensor = torch.tensor(uncon_input_text, device=DEVICE).unsqueeze(0)
        uncon_input_image_tensor = torch.tensor(uncon_input_image, device=DEVICE).unsqueeze(0)
        
        config = MODEL.config
        text_vocab_size = getattr(config, 'text_vocab_size', 126356)
        codebook_size = getattr(config, 'codebook_size', 8192)
        
        for step, text_display, image, status in generate_ti2ti_stepwise(
            model=MODEL, input_ids=con_input, text_start=text_start, text_end=text_end,
            image_start=image_start, seq_len=seq_len, newline_every=newline_every,
            text_steps=text_steps, temperature=temperature, text_temperature=text_temperature,
            cfg_scale=cfg_scale, cfg_img=cfg_img, uncon_text=uncon_input_text_tensor,
            uncon_image=uncon_input_image_tensor, tokenizer=TOKENIZER,
            remasking=remasking_strategy, noise_schedule=cosine_schedule,
            generator=generator, text_vocab_size=text_vocab_size,
            codebook_size=codebook_size, vqvae=VQVAE,
            image_height=height, image_width=width,
        ):
            yield text_display, image, status
    
    except Exception as e:
        import traceback
        yield "", None, f"✗ Error: {str(e)}"

css_styles = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
    max-width: 1400px !important;
    margin: auto;
}
.gr-button-primary {
    background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
}
.gr-button-primary:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4) !important;
}
.output-markdown {
    min-height: 400px !important;
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 12px !important;
    background: #fafafa !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
}
.output-markdown .prose,
.output-markdown .prose * {
    font-size: 10px !important;
    line-height: 1.4 !important;
}
.output-markdown h1 {
    font-size: 1.4em !important;
    margin-top: 0.8em !important;
    margin-bottom: 0.4em !important;
    color: #333 !important;
}
.output-markdown h2 {
    font-size: 1.2em !important;
    margin-top: 0.8em !important;
    margin-bottom: 0.4em !important;
    color: #333 !important;
}
.output-markdown h3 {
    font-size: 1.1em !important;
    margin-top: 0.8em !important;
    margin-bottom: 0.4em !important;
    color: #333 !important;
}
.output-markdown code {
    background: #f0f0f0 !important;
    padding: 2px 4px !important;
    border-radius: 3px !important;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
    font-size: 12px !important;
}
.output-markdown pre {
    background: #f5f5f5 !important;
    padding: 8px !important;
    border-radius: 5px !important;
    overflow-x: auto !important;
    font-size: 12px !important;
}
.output-markdown ul, .output-markdown ol {
    padding-left: 18px !important;
    margin: 8px 0 !important;
}
.output-markdown li {
    margin: 4px 0 !important;
}
.output-markdown p {
    margin: 6px 0 !important;
}
.output-markdown strong {
    font-weight: 600 !important;
}
footer {display: none !important}
"""

with gr.Blocks(css=css_styles, theme=gr.themes.Soft(primary_hue="purple")) as demo:
    gr.Markdown(
        """
        # 🎨 MMaDA-Parallel: Text+Image to Text+Image Generation
        
        Real-time parallel generation with step-by-step visualization.
        
        **Github:** [tyfeld/MMaDA-Parallel-A](https://github.com/tyfeld/MMaDA-Parallel-A)
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            
            input_image = gr.Image(type="pil", label="Input Image")
            prompt_text = gr.Textbox(
                label="Editing Instruction",
                lines=3,
                value="Make the sky more dramatic with sunset colors",
                placeholder="Enter your editing instruction..."
            )
            
            with gr.Accordion("Model", open=False):
                model_path = gr.Textbox(
                    label="Model Path",
                    value="tyfeld/MMaDA-Parallel-A",
                    info="HuggingFace path or local directory"
                )
                vae_path = gr.Textbox(
                    label="VAE Path",
                    value="tyfeld/MMaDA-Parallel-A",
                    info="VQ-VAE checkpoint path"
                )
            
            with gr.Accordion("Parameters", open=False):
                with gr.Row():
                    height = gr.Slider(256, 768, value=512, step=64, label="Height")
                    width = gr.Slider(256, 768, value=512, step=64, label="Width")
                
                text_steps = gr.Slider(32, 512, value=128, step=32, label="Steps")
                text_gen_length = gr.Slider(64, 512, value=256, step=32, label="Text Length")
                text_block_length = gr.Slider(16, 128, value=32, step=16, label="Block Length")
                
                with gr.Row():
                    cfg_scale = gr.Slider(0, 5, value=2.5, step=0.5, label="Text CFG")
                    cfg_img = gr.Slider(0, 8, value=4.0, step=0.5, label="Image CFG")
                
                with gr.Row():
                    temperature = gr.Slider(0, 2, value=1.0, step=0.1, label="Image Temp")
                    text_temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="Text Temp")
                
                remasking_strategy = gr.Dropdown(
                    choices=["low_confidence", "random"],
                    value="low_confidence",
                    label="Remasking"
                )
                
                seed = gr.Slider(0, 10000, value=0, step=1, label="Seed (0=random)")
            
            with gr.Accordion("Painting Mode", open=False):
                painting_mode = gr.Dropdown(
                    choices=[None, "inpainting", "outpainting"],
                    value=None,
                    label="Mode"
                )
                with gr.Row():
                    mask_h_ratio = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Mask H")
                    mask_w_ratio = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Mask W")
            
            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### Output")
            
            status_text = gr.Textbox(label="Status", lines=2, interactive=False)
            
            with gr.Row():
                with gr.Column(scale=1.2):
                    output_text = gr.Markdown(
                        value="*Waiting...*",
                        label="Generated Text (▓ = masked)",
                        show_label=True,
                        container=True,
                        elem_classes=["output-markdown"]
                    )
                
                with gr.Column(scale=1):
                    output_image = gr.Image(label="Generated Image", type="pil", interactive=False)
    
    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            input_image, prompt_text, model_path, vae_path,
            height, width, text_steps, text_gen_length, text_block_length,
            cfg_scale, cfg_img, temperature, text_temperature,
            remasking_strategy, painting_mode, mask_h_ratio, mask_w_ratio, seed
        ],
        outputs=[output_text, output_image, status_text]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MMaDA-Parallel Gradio Demo")
    parser.add_argument("--model_path", type=str, default="tyfeld/MMaDA-Parallel-A")
    parser.add_argument("--vae_path", type=str, default="tyfeld/MMaDA-Parallel-A")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    print("Loading model...")
    load_status = load_model_and_vae(args.model_path, args.vae_path)
    print(load_status)
    
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
