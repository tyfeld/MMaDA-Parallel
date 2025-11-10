import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import math
from PIL import Image
import torch
from transformers import AutoTokenizer
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (
    preprocess_image, decode_vq_to_image, calculate_vq_params,
    generate_crop_size_list, var_center_crop, add_break_line, encode_img_with_breaks,
    encode_img_with_paint
)
from generators.parallel_generator import generate_ti2ti
from utils.prompt_utils import generate_text_image_to_text_image_prompt

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
SYSTEM_PROMPT = (
    "Generate an image applying the following editing instruction based on the original image."
)


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


def main():
    parser = argparse.ArgumentParser(description="Text+Image to Text+Image inference (TI2TI)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for editing")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of diffusion timesteps")
    parser.add_argument("--text_steps", type=int, default=256, help="Number of text generation steps")
    parser.add_argument("--text_gen_length", type=int, default=256, help="Maximum text generation length")
    parser.add_argument("--text_block_length", type=int, default=32, help="Text generation block length")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="CFG scale for text")
    parser.add_argument("--cfg_img", type=float, default=4.0, help="CFG scale for image")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--text_temperature", type=float, default=0.7, help="Text generation temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_ti2ti", help="Output directory")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Remasking strategy")
    parser.add_argument("--painting_mode", type=str, default=None, help="If set, use painting-mode encoding")
    parser.add_argument("--mask_h_ratio", type=float, default=0.5, help="mask height ratio for painting mode")
    parser.add_argument("--mask_w_ratio", type=float, default=0.5, help="mask width ratio for painting mode")
    parser.add_argument("--debug_tokens", action="store_true", help="Print token debug info to verify sequence layout")
    args = parser.parse_args()

    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    if args.seed != 0:
        setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )

    config = model.config
    text_vocab_size = getattr(config, 'text_vocab_size', 126356)
    codebook_size = getattr(config, 'codebook_size', 8192)

    print(f"Vocabulary config: text_vocab_size={text_vocab_size}, codebook_size={codebook_size}")

    print(f"Loading VQ-VAE from {args.vae_ckpt}...")
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    prompt_text = args.prompt
    input_image_path = args.image_path

    print(f"\n{'='*80}")
    print(f"TI2TI Generation")
    print(f"{'='*80}")
    print(f"Input image: {input_image_path}")
    print(f"Prompt: {prompt_text}")
    print(f"Output size: {args.height}x{args.width}")
    print(f"{'='*80}\n")

    input_prompt, uncon_text = generate_text_image_to_text_image_prompt(
        prompt_text, SYSTEM_PROMPT
    )

    print("Conditioning prompt:\n", input_prompt)
    if args.debug_tokens:
        print("Unconditional text prompt (first 200 chars):", uncon_text[:200])

    prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_text_ids = tokenizer(uncon_text)["input_ids"]

    img = Image.open(input_image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
    img = var_center_crop(img, crop_size_list=crop_size_list)

    input_image_width, input_image_height = img.size

    print("Encoding input image for conditioning...")
    input_img_token = encode_img_with_breaks(img, vqvae)

    con_input_list = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
    uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
    uncon_input_image = prompt_ids

    output_image_height = args.height
    output_image_width = args.width
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
        output_image_height, output_image_width, vae_scale
    )

    text_mask_tokens = [MASK] * args.text_gen_length

    if args.painting_mode:
        img_mask_token, img_vis = encode_img_with_paint(
            img, vqvae=vqvae, mask_h_ratio=args.mask_h_ratio, mask_w_ratio=args.mask_w_ratio, mask_mode=args.painting_mode
        )
    else:
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)

    end_token_ids = tokenizer("</answer>", add_special_tokens=False).input_ids

    pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + text_mask_tokens + end_token_ids

    code_start = len(con_input_list)
    image_start = len(con_input_list) + 2
    image_end = image_start + len(img_mask_token)
    text_start = image_end + 1
    text_end = text_start + args.text_gen_length

    full_input_ids = con_input_list + pred_token
    con_input = torch.tensor(full_input_ids, device=device).unsqueeze(0)
    uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
    uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)
    start_time = time.time()

    if args.seed != 0:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    output_tokens, generated_text = generate_ti2ti(
        model=model,
        input_ids=con_input,
        text_start=text_start,
        text_end=text_end,
        image_start=image_start,
        seq_len=seq_len,
        newline_every=newline_every,
        text_steps=args.text_steps,
        text_gen_length=args.text_gen_length,
        text_block_length=args.text_block_length,
        timesteps=args.timesteps,
        temperature=args.temperature,
        text_temperature=args.text_temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_input_text,
        uncon_image=uncon_input_image,
        tokenizer=tokenizer,
        remasking=args.remasking,
        noise_schedule=cosine_schedule,
        generator=generator,
        text_vocab_size=text_vocab_size,
        codebook_size=codebook_size,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'='*80}")
    print(f"Generated thinking/text output:")
    print(f"{'='*80}")
    print(generated_text)
    print(f"{'='*80}\n")

    print(f"Converting {len(output_tokens)} VQ tokens to tensor...")
    output_tokens_tensor = torch.tensor(output_tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"VQ tokens range: [{min(output_tokens)}, {max(output_tokens)}]")

    words = (prompt_text or "").split()
    filename_words = words[:10] if len(words) > 10 else words
    filename = "_".join(filename_words)
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    filename = f"{filename}_{output_image_height}x{output_image_width}_t{args.timesteps}_cfg{args.cfg_scale}_ti2ti.png"

    save_path = os.path.join(args.output_dir, filename)

    print("Decoding image...")
    out_img = decode_vq_to_image(
        output_tokens_tensor,
        save_path,
        vae_ckpt=args.vae_ckpt,
        image_height=output_image_height,
        image_width=output_image_width,
        vqvae=vqvae
    )

    w1, h1 = img.size
    w2, h2 = out_img.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(out_img, (w1, 0))
    concat_path = save_path.replace(".png", "_concat.png")
    canvas.save(concat_path)

    text_path = save_path.replace(".png", "_thinking.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{generated_text}\n")

    print(f"\n[✓] Image saved to: {concat_path}")
    print(f"[✓] Text saved to: {text_path}")
    print(f"[✓] Total time: {elapsed_time:.2f}s")


if __name__ == '__main__':
    main()
