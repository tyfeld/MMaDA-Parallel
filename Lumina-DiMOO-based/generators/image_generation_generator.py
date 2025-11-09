# -*- coding: utf-8 -*-
"""
Image generation generator (with optional debug prints/saving)
"""
import torch
import math
import os
import numpy as np
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from model import LLaDAForMultiModalGeneration


@torch.no_grad()
def generate_image(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    uncon_ids: torch.LongTensor = None,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
    debug: bool = True,
    debug_log_dir: Optional[str] = None,
    max_print_tokens: int = 100
) -> torch.LongTensor:
    """
    MaskGit parallel decoding to generate VQ tokens

    Added debug=True to print shapes and token samples per step. Optional debug_log_dir to save numpy dumps.

    Args:
        debug: when True, print detailed info each step.
        debug_log_dir: directory to save per-step npy dumps (x, vq_mask, logits, sampled_full)
        max_print_tokens: maximum number of tokens/logits to print for arrays (prevents terminal spam)
    """

    if debug and debug_log_dir:
        os.makedirs(debug_log_dir, exist_ok=True)

    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    x = prompt.clone()

    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    if isinstance(model, LLaDAForMultiModalGeneration):
        model.caching(use_cache)
    else:  # DDP
        model.module.caching(use_cache)

    warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    for step in range(timesteps):
        if not use_cache or step <= warmup_step or (step-warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    # Infer text vocabulary size
    if text_vocab_size is None:
        # call with a minimal input to get logits size
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    if debug:
        print("=== generate_image debug start ===")
        print(f"device={device}, seq_len={seq_len}, code_start={code_start}, codebook_size={codebook_size}")
        print(f"text_vocab_size={text_vocab_size}, vocab_offset={vocab_offset}")
        print(f"Initial x.shape={x.shape}, initial unknown_cnt={int(unknown_cnt.item())}")
        print("==================================")

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            if debug:
                print(f"[step {step}] All tokens filled, breaking early.")
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_cache and step and refresh_steps[step]:
            if isinstance(model, LLaDAForMultiModalGeneration):
                model.empty_cache()
            else:  # DDP
                model.module.empty_cache()

        if debug:
            print(f"\n--- step {step} ---")
            print(f"unknown_cnt={int(unknown_cnt.item())}, keep_n={int(keep_n.item())}, refresh_step={bool(refresh_steps[step])}")
            print(f"x.shape={x.shape}, vq_mask.sum()={int(vq_mask.sum().item())}")
            # print a slice of tokens around code_start for visibility if code_start is set
            if code_start is not None:
                cs = code_start
                sample_slice = x[0, cs:cs+min(50, x.shape[1]-cs)].detach().cpu().numpy().tolist()
                print(f"x tokens at code_start (first 50): {sample_slice[:min(len(sample_slice), max_print_tokens)]}")
        
        # Forward pass (with/without CFG)
        if cfg_scale > 0:
            # build uncond sequence
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
            uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)

            # conditional logits
            cond_out = model(x, infer=True, use_cache=use_cache)
            cond_logits = cond_out.logits[..., vocab_offset : vocab_offset + codebook_size]
            if debug:
                print(f"cond_logits shape: {cond_logits.shape}")
            cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
            """
            if debug:
                print(f"cond_mask_logits shape (after vq_mask): {tuple(cond_mask_logits.shape)}")
                # print few values
                tmp = cond_mask_logits.detach().cpu().numpy()
                flat_tmp = tmp.reshape(-1, tmp.shape[-1])
                if flat_tmp.shape[0] > 0:
                    print("cond_mask_logits[first_row, first_10]:", flat_tmp[0, :min(10, flat_tmp.shape[1])].tolist())
"""
            # unconditional logits
            uncond_out = model(uncond, infer=True, use_cache=use_cache)
            uncond_logits = uncond_out.logits[..., vocab_offset : vocab_offset + codebook_size]
            if debug:
                print(f"uncond_logits shape: {uncond_logits.shape}")
            uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            """
            if debug:
                print(f"uncond_mask_logits shape: {tuple(uncond_mask_logits.shape)}")
                tmpu = uncond_mask_logits.detach().cpu().numpy()
                if tmpu.size:
                    print("uncond_mask_logits[first_row, first_10]:", tmpu.reshape(-1, tmpu.shape[-1])[0, :min(10, tmpu.shape[-1])].tolist())
"""
            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
            if debug:
                print(f"combined logits shape: {logits.shape}")

        else:
            out = model(x, infer=True)
            # logits for masked positions: (B, num_masked, codebook_size)
            # here we index directly by boolean mask along sequence dim
            logits = out.logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            if debug:
                print(f"logits shape (no-cfg): {logits.shape}")
                ltmp = logits.detach().cpu().numpy()
                if ltmp.size:
                    print("logits[first_pos, first_10]:", ltmp[0, :min(10, ltmp.shape[1])].tolist() if ltmp.ndim == 2 else ltmp.reshape(-1, ltmp.shape[-1])[0, :min(10, ltmp.shape[-1])].tolist())

        # sample
        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset  # bring to full token space
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        if debug:
            print(f"sampled.shape={sampled.shape}, sampled_full.shape={sampled_full.shape}, conf.shape={conf.shape}")
            # print some sampled tokens
            sf_np = sampled_full.detach().cpu().numpy().reshape(-1).tolist()
            print(f"sampled_full(first {min(len(sf_np), max_print_tokens)}): {sf_np[:min(len(sf_np), max_print_tokens)]}")

        # write sampled tokens into x at masked positions
        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        if debug:
            print(f"flat_idx (masked positions indices) length={flat_idx.shape[0]}")
            if flat_idx.numel() > 0:
                print(f"flat_idx first 30: {flat_idx[:min(30, flat_idx.shape[0])].detach().cpu().numpy().tolist()}")

        x.view(-1)[flat_idx] = sampled_full.view(-1)

        # confidence map (for display / selection)
        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        if debug:
            # show some stats of conf_map in code region
            try:
                conf_np = conf.detach().cpu().numpy().reshape(-1)
                print(f"conf stats (min/mean/max): {float(conf_np.min()):.6f}/{float(conf_np.mean()):.6f}/{float(conf_np.max()):.6f}")
            except Exception:
                pass

        # mask selection -> re-mask some tokens for next step
        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        if debug:
            print(f"mask_sel.shape={mask_sel.shape}, mask_sel.sum()={int(mask_sel.sum().item())}")
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        if debug:
            print(f"after masking, vq_mask.sum()={int(vq_mask.sum().item())}, unknown_cnt={int(unknown_cnt.item())}")

        # Save debug artifacts if requested
        if debug and debug_log_dir:
            step_base = os.path.join(debug_log_dir, f"step_{step}")
            try:
                np.save(step_base + "_x.npy", x.detach().cpu().numpy())
                np.save(step_base + "_vq_mask.npy", vq_mask.detach().cpu().numpy())
                # logits may be large; save as float32
                np.save(step_base + "_logits.npy", logits.detach().cpu().numpy().astype(np.float32))
                np.save(step_base + "_sampled_full.npy", sampled_full.detach().cpu().numpy())
            except Exception as e:
                print(f"[debug] failed to save debug npy at step {step}: {e}")

        # Update cond/uncond compute masks for caching only if cfg_scale>0
        if use_cache and step < timesteps - 1 and not refresh_steps[step+1] and cfg_scale > 0:
            cond_conf = cond_logits.max(dim=-1)[0]
            cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            cond_to_compute_mask = cond_conf <= cond_conf_threshold

            uncond_conf = uncond_logits.max(dim=-1)[0]
            uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold

            if debug:
                print(f"cond_conf shape: {cond_conf.shape}, threshold: {cond_conf_threshold.detach().cpu().numpy().tolist()}")
                print(f"uncond_conf shape: {uncond_conf.shape}, threshold: {uncond_conf_threshold.detach().cpu().numpy().tolist()}")

    # Remove newline tokens and shape properly
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)

    if debug:
        print("=== generate_image debug end ===")
        print(f"final vq_ids.shape={vq_ids.shape}")
        try:
            print("final vq_ids first 100:", vq_ids.detach().cpu().numpy().reshape(-1)[:min(max_print_tokens, vq_ids.numel())].tolist())
        except Exception:
            pass

    return vq_ids
