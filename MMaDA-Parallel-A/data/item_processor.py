import torch
import logging
from transformers import AutoTokenizer, AutoModel
from diffusers import VQModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from PIL import Image
import PIL
import random
import json
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms

logger = logging.getLogger(__name__)

def center_crop(pil_image, crop_size):
    """Randomized center-like crop with resize-to-fit behavior.

    The function progressively downsamples large images for efficiency, then
    resizes to ensure the requested crop size fits and selects a random window
    of the exact target dimensions.
    """
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
    """Select a crop size from `crop_size_list` favoring aspect-ratio match.

    It ranks candidate crop sizes by how well they match the original image's
    aspect ratio, samples from the top-k, and applies `center_crop`.
    """
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    """Enumerate candidate crop sizes (multiples of `patch_size`).

    The generated sizes are constrained by a maximum aspect ratio. This list is
    later used to choose size candidates for variable center cropping.
    """
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

class DimooItemProcessor:
    def __init__(self, 
                 tokenizer_path, 
                 vq_ckpt_path,
                 target_size,
                 device="cuda", 
                 codebook_size=8192, 
                 codebook_embed_dim=8):
        """Initialize VQ-VAE and preprocessing utilities.

        Args:
            tokenizer_path: Path or repo-id for the text tokenizer (not used
                directly here but expected by subclasses).
            vq_ckpt_path: Path or repo-id with a `vqvae` subfolder for diffusers
                `VQModel.from_pretrained`.
            target_size: Target image size used to construct candidate crop sizes.
            device: Preferred device; will fall back to CPU if CUDA unavailable.
            codebook_size: Expected VQ codebook size (informational).
            codebook_embed_dim: Expected embedding dim (informational).
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vq_model = VQModel.from_pretrained(vq_ckpt_path, subfolder="vqvae", )
        self.vq_model.to(self.device)
        self.vq_model.eval()
        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        logger.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))
    

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        """Replace transparent regions with white background, return RGB image."""
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def encode_image(self, image: PIL.Image) -> list[int]:
        """Encode an image into VQ indices using the loaded VQ-VAE model."""
        image = self._whiten_transparency(image)
        vae_scale_factor = 2 ** (len(self.vq_model.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
        x = image_processor.preprocess(image).to(self.device)
        latents = self.vq_model.encode(x).latents
        latents_bsz, channels, lat_h, lat_w = latents.shape
        quantized = self.vq_model.quantize(latents)[2][2].reshape(latents_bsz, lat_h, lat_w)
        self.vq_model.to(self.device)
        return quantized.flatten().tolist()

    def process_image(self, image):
        """Convert one or multiple images to VQ token ids and collect size meta.

        Returns a tuple with tokens and dimensions. For multiple images, returns
        three lists: tokens per image, heights, and widths. For a single image,
        returns tokens, height, width.
        """
        satrt_img_token_id = 126356
        if isinstance(image, Image.Image):
            pass
        elif isinstance(image, list):
            image_collection, height_collection, width_colletion = [], [], []
            for img in image:
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                height, width = img.size[0], img.size[1]
                print(img.size)
                img_tokens = self.encode_image(img)
                image_toks = [x + satrt_img_token_id for x in img_tokens]
                image_collection.append(image_toks)
                height_collection.append(height)
                width_colletion.append(width)
            return image_collection, height_collection, width_colletion
        
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = var_center_crop(image, crop_size_list=self.crop_size_list)
        height, width = image.size[0], image.size[1]
        print(image.size)
        img_tokens = self.encode_image(image)
        image_toks = [x + satrt_img_token_id for x in img_tokens]
        return image_toks, height, width
        
    def process_text(self, text):
        """Tokenize text with BOS/EOS markers using the configured tokenizer."""
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids
        text_toks = input_ids[0].tolist()
        result_toks = (
            [self.tokenizer._convert_token_to_id_add("<|beginoftext|>")] +
            text_toks +
            [self.tokenizer._convert_token_to_id_add("<|endoftext|>")]
        )

        return result_toks
    
    def process_item(self, text, image_path):
        """Create standardized record(s) from text prompt and image path(s).

        For multi-image inputs, returns a list of dicts; otherwise returns a
        single dict containing `input_ids` and image dimensions.
        """
        image_tokens, height, width = self.process_image(image_path)
        if len(image_tokens)<20:
            result_toks=[]
            for (tokens, h, w) in zip(image_tokens, height, width):
                result_tok = {
                    "input_ids": tokens,
                    "length": len(tokens),
                    "height": h,
                    "width": w
                }
                result_toks.append(result_tok)
        else:
            result_toks = {
                "input_ids": image_tokens,
                "length": len(image_tokens),
                "height": height,
                "width": width
            }
        
        return result_toks

    
