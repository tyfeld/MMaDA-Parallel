import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np


def add_gumbel_noise(logits, temperature=1.0, generator=None):
    """Add Gumbel noise to logits for sampling"""
    if temperature == 0:
        return logits

    if generator is not None:
        uniform_noise = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator)
    else:
        uniform_noise = torch.rand_like(logits)

    gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)

    return logits + temperature * gumbel_noise


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    """
    Mask tokens by random top-k selection based on confidence
    probs: [batch, L] confidence scores (higher = more confident)
    mask_len: tensor shape [batch, 1] or scalar, number of tokens to keep masked (lowest-confidence)
    returns: boolean mask [batch, L] True where token should REMAIN masked
    """
    if generator is not None:
        noise = torch.randn(probs.shape, dtype=probs.dtype, device=probs.device, generator=generator)
    else:
        noise = torch.randn_like(probs)

    # Add small noise to jitter confidences according to temperature
    confidence = torch.log(probs + 1e-10) + temperature * noise  # higher = more confident

    # We want to mask lowest-confidence tokens -> find cutoff
    sorted_confidence, sorted_indices = torch.sort(confidence, dim=-1, descending=False)  # ascending

    # mask_len may be float or tensor; ensure integer per-batch
    if isinstance(mask_len, torch.Tensor):
        mask_len_clamped = torch.clamp(mask_len, 0, probs.shape[-1] - 1)
        mask_len_clamped = mask_len_clamped.long().squeeze(-1)  # shape [batch]
    else:
        mask_len_clamped = int(mask_len)

    # Build boolean mask: True for tokens to KEEP masked (lowest confidence)
    if isinstance(mask_len_clamped, torch.Tensor):
        batch = probs.shape[0]
        masking = torch.zeros_like(probs, dtype=torch.bool, device=probs.device)
        for b in range(batch):
            k = mask_len_clamped[b].item()
            if k <= 0:
                continue
            low_idx = sorted_indices[b, :k]  # indices of lowest k confidences
            masking[b, low_idx] = True
    else:
        # scalar k
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


def cosine_schedule(t):
    """Cosine noise schedule"""
    return torch.cos(t * math.pi / 2)


def get_num_transfer_tokens(text_masked_indices, text_steps):
    """
    Calculate number of tokens to unmask at each step
    Returns: [batch_size, text_steps]
    """
    batch_size = text_masked_indices.shape[0]
    initial_masks = text_masked_indices.sum(dim=1)  # [batch_size]

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


def generate_ti2ti(
    model,
    input_ids,
    text_start,
    text_end,
    image_start,
    seq_len,
    newline_every,
    text_steps=100,
    text_gen_length=256,
    text_block_length=64,
    timesteps=100,
    temperature=1.0,
    text_temperature=0.7,
    cfg_scale=0.0,
    cfg_img=4.0,
    uncon_text=None,
    uncon_image=None,
    tokenizer=None,
    remasking='low_confidence',
    noise_schedule=cosine_schedule,
    generator=None,
    text_vocab_size=126356,
    codebook_size=8192,
):
    """
    Generate text and image jointly with interleaved generation.
    Text generation uses cond logits only (text_cfg assumed 0).
    Image generation (at scheduled steps) uses two CFGs:
       - uncond_text (if provided) : guidance that relates to text part
       - uncond_image (if provided): guidance that relates to image part
    """

    device = input_ids.device
    MASK_TOKEN = 126336
    NEW_LINE = 126084

    # Clone input for modification
    combined_input_ids = input_ids.clone()

    # Calculate total image region length (including newlines)
    num_vq_tokens = seq_len
    total_image_len = seq_len + seq_len // newline_every
    image_end = image_start + total_image_len

    print(f"Interleaved generation: {text_steps} total steps")
    print(f"  - Text generation range: [{text_start}, {text_end})")
    print(f"  - Image generation range: [{image_start}, {image_end}) (total {total_image_len} including newlines)")
    print(f"  - VQ tokens: {num_vq_tokens}")

    # Calculate number of tokens to unmask at each step for text
    text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN
    num_transfer_tokens = get_num_transfer_tokens(text_masked_indices, text_steps)

    # Schedule: when to perform image generation steps
    image_generation_step_indices = torch.linspace(
        text_steps // 4, text_steps - 1, timesteps
    ).round().int().tolist()

    print(f"  - Image generation at steps: {image_generation_step_indices[:5]}...{image_generation_step_indices[-5:]}")

    # Build position mapping for image (excluding newlines)
    image_position_mapping = []
    for i in range(image_start, image_end):
        if combined_input_ids[0, i] != NEW_LINE:
            image_position_mapping.append(i)

    assert len(image_position_mapping) == num_vq_tokens, f"Expected {num_vq_tokens} VQ tokens, got {len(image_position_mapping)}"

    batch_size = combined_input_ids.shape[0]

    # ========== Interleaved Generation Loop ==========
    for step in tqdm(range(text_steps), desc="Interleaved generation"):

        # ===== Forward pass: compute conditional logits once per step =====
        with torch.no_grad():
            cond_logits = model(combined_input_ids, infer=True, use_cache=False).logits  # [B, L, V]

        # ===== Text Generation Step (no CFG for text; use cond_logits directly) =====
        text_masked_indices = combined_input_ids[:, text_start:text_end] == MASK_TOKEN

        if text_masked_indices.sum() > 0:
            # Extract text logits from cond (no guidance)
            text_logits = cond_logits[:, text_start:text_end, :]

            # Apply temperature & gumbel
            logits_with_noise = add_gumbel_noise(text_logits, temperature=text_temperature, generator=generator)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, text_len]

            # Compute confidence for remasking
            if remasking == 'low_confidence':
                p = F.softmax(text_logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, text_len]
            elif remasking == 'random':
                if generator is not None:
                    x0_p = torch.rand(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
                else:
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # keep already-unmasked tokens
            x0 = torch.where(text_masked_indices, x0, combined_input_ids[:, text_start:text_end])
            confidence = torch.where(text_masked_indices, x0_p, -np.inf)

            # Select tokens to unmask based on confidence (top-k per batch element)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, step].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            # Unmask selected tokens into combined_input_ids
            # Note: transfer_index is [B, text_len] boolean; place into full combined_input_ids
            combined_input_ids[:, text_start:text_end][transfer_index] = x0[transfer_index]

        # ===== Image Generation Step (scheduled) =====
        if step in image_generation_step_indices:
            # Build vq token list from current combined_input_ids (placeholder -1 for masked)
            vq_tokens_list = []
            for pos in image_position_mapping:
                token = combined_input_ids[0, pos].item()
                if token == MASK_TOKEN:
                    vq_tokens_list.append(-1)
                else:
                    vq_token = token - text_vocab_size
                    vq_token = max(0, min(vq_token, codebook_size - 1))
                    vq_tokens_list.append(vq_token)

            vq_tokens_tensor = torch.tensor(vq_tokens_list, device=device).unsqueeze(0)  # [1, num_vq_tokens]
            unknown_map = vq_tokens_tensor == -1  # True where masked

            # Extract cond_vq_logits from cond_logits (for VQ positions and vocab offset)
            cond_image_logits_list = []
            for pos in image_position_mapping:
                cond_image_logits_list.append(cond_logits[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])
            cond_vq_logits = torch.cat(cond_image_logits_list, dim=1)  # [B, num_vq_tokens, codebook_size]

            # Prepare uncond logits only when needed (for image CFG)
            # Create combined_uncond_text and combined_uncond_img by replacing prefix with uncon_text/uncon_image
            if (cfg_scale > 0.0 and uncon_text is not None) or (cfg_img > 0.0 and uncon_image is not None):
                # clone base input
                # IMPORTANT: uncon_text/uncon_image expected to be on the same device or will be moved
                # If uncon_text / uncon_image is None, create copies to avoid errors
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

                # Forward for unconds
                with torch.no_grad():
                    uncond_text_logits_full = model(combined_uncond_text, infer=True, use_cache=False).logits
                    uncond_img_logits_full = model(combined_uncond_img, infer=True, use_cache=False).logits

                # Extract VQ ranges for each image position
                uncond_text_vq_list = []
                uncond_img_vq_list = []
                for pos in image_position_mapping:
                    uncond_text_vq_list.append(uncond_text_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])
                    uncond_img_vq_list.append(uncond_img_logits_full[:, pos:pos+1, text_vocab_size:text_vocab_size+codebook_size])

                uncond_text_vq_logits = torch.cat(uncond_text_vq_list, dim=1)  # [B, num_vq_tokens, codebook_size]
                uncond_img_vq_logits = torch.cat(uncond_img_vq_list, dim=1)    # [B, num_vq_tokens, codebook_size]
            else:
                # no unconds provided or scales are zero -> set uncond logits to zeros so (cond - 0) works if used
                uncond_text_vq_logits = torch.zeros_like(cond_vq_logits)
                uncond_img_vq_logits = torch.zeros_like(cond_vq_logits)

            # Compose guided image logits:
            # image_logits = cond_vq + cfg_scale * (cond_vq - uncond_text_vq) + cfg_img * (cond_vq - uncond_img_vq)
            if cfg_scale == 0.0 and cfg_img == 0.0:
                image_logits = cond_vq_logits
            else:
                image_logits = cond_vq_logits
                if cfg_scale != 0.0:
                    image_logits = image_logits + cfg_scale * (cond_vq_logits - uncond_text_vq_logits)
                if cfg_img != 0.0:
                    image_logits = image_logits + cfg_img * (cond_vq_logits - uncond_img_vq_logits)

            # Sample from image_logits
            probs = F.softmax(image_logits, dim=-1)  # [B, num_vq, codebook]

            if temperature == 0:
                sampled_ids = probs.argmax(dim=-1)
            else:
                # flatten batch*num_vq x vocab for multinomial
                sampled = probs.reshape(-1, image_logits.size(-1))
                if generator is not None:
                    sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*image_logits.shape[:-1])
                else:
                    sampled_ids = torch.multinomial(sampled, 1)[:, 0].view(*image_logits.shape[:-1])

            # Keep already-unmasked tokens unchanged
            sampled_ids = torch.where(unknown_map, sampled_ids, vq_tokens_tensor)

            # Clamp safety
            sampled_ids = torch.clamp(sampled_ids, 0, codebook_size - 1)

            # Confidence for sampled tokens
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)  # [B, num_vq]

            # If token was previously unmasked, give it very high confidence so we don't remask it
            high_val = torch.finfo(selected_probs.dtype).max
            selected_probs = torch.where(unknown_map, selected_probs, high_val)

            # Masking ratio and mask_len calculation
            ratio = 1.0 * (step + 1) / text_steps
            mask_ratio = noise_schedule(torch.tensor(ratio, device=device))
            # compute how many tokens to keep masked (lowest confidences)
            unknown_counts = unknown_map.sum(dim=-1, keepdim=True)  # [B,1]
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(device)  # shape [1,] maybe
            # clamp mask_len to [1, unknown_counts-1]
            mask_len = torch.max(torch.tensor([1], device=device), torch.min(unknown_counts - 1, mask_len.to(device).long()))
            # ensure shape [B,1]
            if mask_len.ndim == 1:
                mask_len = mask_len.unsqueeze(1)

            # temperature decay for image sampling (optional)
            img_temp = temperature * (1.0 - ratio)

            # masking boolean: True where should remain masked
            masking = mask_by_random_topk(mask_len, selected_probs, img_temp, generator=generator)

            # final_vq_tokens: -1 means remain masked, else sampled id
            final_vq_tokens = torch.where(masking, torch.tensor(-1, device=device), sampled_ids)

            # Write back into combined_input_ids (convert vq id -> full vocab id by adding offset)
            for idx, pos in enumerate(image_position_mapping):
                v = final_vq_tokens[0, idx].item()
                if v == -1:
                    combined_input_ids[0, pos] = MASK_TOKEN
                else:
                    combined_input_ids[0, pos] = int(v + text_vocab_size)

    # ===== Extract final results =====
    # Extract text tokens
    text_tokens = combined_input_ids[0, text_start:text_end].cpu().tolist()
    text_tokens = [t for t in text_tokens if t != MASK_TOKEN]
    generated_text = tokenizer.decode(text_tokens, skip_special_tokens=True) if tokenizer is not None else text_tokens

    # Extract image VQ tokens
    image_tokens = []
    for pos in image_position_mapping:
        token = combined_input_ids[0, pos].item()
        if token != MASK_TOKEN:
            vq_token = token - text_vocab_size
            vq_token = max(0, min(vq_token, codebook_size - 1))
            image_tokens.append(vq_token)
        else:
            # still masked -> sample randomly
            image_tokens.append(int(torch.randint(0, codebook_size, (1,)).item()))

    print(f"Interleaved generation complete.")
    print(f"  - Generated text: {len(text_tokens)} tokens")
    print(f"  - Generated image: {len(image_tokens)} VQ tokens (range [0, {codebook_size}))")

    return image_tokens, generated_text
