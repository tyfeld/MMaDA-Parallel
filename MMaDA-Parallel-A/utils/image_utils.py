# -*- coding: utf-8 -*-
"""
Image processing utilities
"""
import torch
import PIL
import random
from PIL import Image, ImageDraw
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F

def decode_vq_to_image(
    vq_codes: torch.LongTensor, 
    save_path: str = None, 
    vae_ckpt: str = None, 
    image_height: int = 512, 
    image_width: int = 512,
    vqvae: VQModel = None
) -> Image.Image:
    """
    Decode VQ codes to image
    
    Args:
        vq_codes: VQ codes in range [0, codebook_size), shape [batch_size, seq_len]
        save_path: Save path (optional, if None will not save to file)
        vae_ckpt: VAE checkpoint path (optional if vqvae is provided)
        image_height: Image height
        image_width: Image width
        vqvae: VQ-VAE model, if None will load from vae_ckpt
    
    Returns:
        PIL image
    """
    device = vq_codes.device
    if vqvae is None:
        vqvae = VQModel.from_pretrained(vae_ckpt, subfolder="vqvae").to(device)
    
    scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
    img_proc = VaeImageProcessor(vae_scale_factor=scale, do_normalize=False)

    # Calculate latent space grid size
    latent_height = image_height // scale
    latent_width = image_width // scale

    # Ensure VQ codes length matches
    expected_len = latent_height * latent_width
    if vq_codes.shape[1] != expected_len:
        raise ValueError(
            f"VQ codes length mismatch: {vq_codes.shape[1]} != {expected_len} "
            f"for image size ({image_height},{image_width}) with scale {scale}"
        )

    # Reshape to 2D grid: [batch_size, seq_len] -> [batch_size, latent_height, latent_width]
    # vq_codes should already be in range [0, codebook_size), no offset needed

    
    latents = vq_codes.view(vq_codes.shape[0], latent_height, latent_width).long()
    # latents = (vq_codes.view(1, latent_height, latent_width) - 126356).long()

    # Decode
    recon = vqvae.decode(
        latents,
        force_not_quantize=True,
        shape=(vq_codes.shape[0], latent_height, latent_width, vqvae.config.latent_channels),
    ).sample.clip(0, 1)

    # Post-process
    img = img_proc.postprocess(recon.detach(), output_type="pil")[0]
    
    # Save image (only if save_path is provided)
    if save_path is not None:
        img.save(save_path)
    
    return img


def preprocess_image(image_path: str, target_size: tuple = (512, 512)):
    """
    Preprocess image: load, crop, resize
    
    Args:
        image_path: Image path
        target_size: Target size (width, height)
    
    Returns:
        Processed PIL image
    """
    img = Image.open(image_path).convert("RGB")
    crop_size_list = generate_crop_size_list((target_size[0] // 32) ** 2, 32)
    processed_img = var_center_crop(img, crop_size_list=crop_size_list)
    return processed_img


def calculate_vq_params(image_height: int, image_width: int, vae_scale: int = 16):
    """
    Calculate VQ related parameters
    
    Args:
        image_height: Image height
        image_width: Image width
        vae_scale: VAE scale factor
    
    Returns:
        seq_len, newline_every, token_grid_height, token_grid_width
    """
    token_grid_height = image_height // vae_scale
    token_grid_width = image_width // vae_scale
    seq_len = token_grid_height * token_grid_width
    newline_every = token_grid_width
    return seq_len, newline_every, token_grid_height, token_grid_width

def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def add_break_line(sequence: list, H: int, W: int, new_number: int = 0) -> list:
    """Add newline characters to sequence"""
    result = []
    for i in range(H):
        start = i * W
        end = start + W
        row = sequence[start:end]
        result.extend(row + [new_number])
    return result

def encode_img_with_breaks(img, vqvae, vae_scale_factor: int = 16):
    """Encode image and add newline characters"""
    from diffusers.image_processor import VaeImageProcessor
    
    orig = img.convert("RGB")
    orig_resized = orig
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(orig_resized).to(vqvae.device)
    latents = vqvae.encode(x).latents
    latents_bsz, channels, lat_h, lat_w = latents.shape
    quantized = vqvae.quantize(latents)[2][2] + 126356
    quantized = quantized.reshape(latents_bsz, lat_h, lat_w).flatten().tolist()
    img_token = add_break_line(quantized, lat_h, lat_w, new_number=126084)
    img_token = [126349] + img_token + [126350]
    return img_token

@torch.no_grad()
def encode_img_with_paint(
    img: Image.Image,
    vqvae: VQModel,
    *,
    mask_h_ratio: float = 1,   # Height ratio
    mask_w_ratio: float = 0.2,    # Width ratio
    gray_value: int = 127,        # Visualization gray value
    downsample_mode: str = "area",# Pixel mask alignment to latent grid
    dilate_latent_k: int = 0,     # Optional dilation on latent grid (grid count)
    mask_mode: str = "inpainting",   # "inpainting" | "outpainting"
):
    """
    Encode image with mask for inpainting/outpainting tasks
    
    Args:
        img: Input PIL image
        vqvae: VQ-VAE model for encoding
        mask_h_ratio: Height ratio for mask region (default: 1.0)
        mask_w_ratio: Width ratio for mask region (default: 0.2)
        gray_value: Gray value for mask visualization (default: 127)
        downsample_mode: Downsampling mode for mask alignment ("area", "nearest", "bilinear")
        dilate_latent_k: Dilation kernel size for latent grid (default: 0)
        mask_mode: Mask mode - "inpainting" (mask inside) or "outpainting" (mask outside)
    
    Returns:
        img_token: List[int] - Token sequence with newlines (126084) inserted at row ends;
                              masked positions = 126336, others = index + 126356
        vis_img: PIL.Image - Gray mask visualization image (consistent with mask_mode)
    
    Note:
        * Encoding uses original image strictly; mask only maps to latent grid to determine
          which tokens are set to MASK_TOKEN_ID.
        * mask_mode="inpainting": mask inside rectangle; "outpainting": mask outside rectangle (inverse).
    """
    MASK_TOKEN_ID = 126336      # mask token
    NEWLINE_TOKEN_ID = 126084   # newline token
    VQ_OFFSET = 126356          # quantization index offset

    assert mask_mode in ("inpainting", "outpainting"), "mask_mode must be 'inpainting' or 'outpainting'"

    # --- 1) Calculate center rectangle and generate visualization ---
    img = img.convert("RGB")
    W, H = img.size
    mh = int(round(H * mask_h_ratio))
    mw = int(round(W * mask_w_ratio))
    top = (H - mh) // 2
    left = (W - mw) // 2
    bottom = top + mh
    right = left + mw

    if mask_mode == "inpainting":
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        draw.rectangle([left, top, right, bottom], fill=(gray_value, gray_value, gray_value))
    elif mask_mode == "outpainting":  # outpainting
        bg = Image.new("RGB", (W, H), (gray_value, gray_value, gray_value))
        crop = img.crop((left, top, right, bottom))
        bg.paste(crop, (left, top))
        vis_img = bg

    # --- 2) VQ encoding using original image ---
    vae_scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(img).to(vqvae.device)  # 1 x 3 x H' x W'
    latents = vqvae.encode(x).latents                     # 1 x C x h x w
    _, _, lat_h, lat_w = latents.shape

    # Quantization indices
    quant_pack = vqvae.quantize(latents)
    indices = quant_pack[2][2].view(1, lat_h, lat_w)      # 1 x h x w, long

    # --- 3) Pixel mask -> latent grid mask (aligned with encoding input size) ---
    Hp, Wp = x.shape[-2:]
    mask_px = torch.zeros((1, 1, Hp, Wp), dtype=torch.float32, device=vqvae.device)
    # First generate mask where "rectangle inside=1, outside=0"
    top_p  = int(round(top  * Hp / H))
    left_p = int(round(left * Wp / W))
    bh_p   = int(round(mh   * Hp / H))
    bw_p   = int(round(mw   * Wp / W))
    mask_px[:, :, top_p:top_p+bh_p, left_p:left_p+bw_p] = 1.0

    # If outpainting, need to invert (outside=1, inside=0 is the masked region)
    if mask_mode == "outpainting":
        mask_px = 1.0 - mask_px

    if downsample_mode not in ("nearest", "area", "bilinear"):
        downsample_mode = "area"
    mask_lat = F.interpolate(mask_px, size=(lat_h, lat_w), mode=downsample_mode)
    mask_lat = (mask_lat > 0.5) if downsample_mode == "area" else (mask_lat >= 0.5)
    mask_lat = mask_lat[0, 0]        # h x w (bool)

    # Optional: latent grid dilation (after inversion is applied)
    if dilate_latent_k > 0:
        m = mask_lat.float().unsqueeze(0).unsqueeze(0)
        ker = 2 * dilate_latent_k + 1
        m = F.max_pool2d(m, kernel_size=ker, stride=1, padding=dilate_latent_k)
        mask_lat = (m[0, 0] > 0.5)

    # --- 4) Generate tokens: masked positions=MASK_TOKEN_ID, others=indices+VQ_OFFSET ---
    idx_flat = indices.view(-1)
    mask_flat = mask_lat.view(-1)
    tokens = torch.empty_like(idx_flat)
    tokens[mask_flat] = MASK_TOKEN_ID
    tokens[~mask_flat] = idx_flat[~mask_flat] + VQ_OFFSET
    tokens_list = tokens.tolist()

    # --- 5) Insert newlines (no longer wrapped in <boi>/<eoi>, consistent with current return) ---

    img_token = add_break_line(tokens_list, lat_h, lat_w, NEWLINE_TOKEN_ID)
    return img_token, vis_img