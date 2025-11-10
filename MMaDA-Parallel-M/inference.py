reserved_token_mapping = {
    '<|soi|>': 126084,  
    '<|eoi|>': 126085,
    '<|sov|>': 126086,
    '<|eov|>': 126087,
    '<|t2i|>': 126088,
    '<|mmu|>': 126089,
    '<|t2v|>': 126090,
    '<|v2v|>': 126091,
    '<|lvg|>': 126092,
    '[iPAD]': 126093,
    '<|r2i|>': 126094,
    '<|interleave|>': 126095,
}
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import numpy as np
import torch
import wandb
from models import MAGVITv2, get_mask_schedule, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf, image_transform_squash
from transformers import AutoTokenizer

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="mmada-interleave-infer",
        name='interleave',
        config=wandb_config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("tyfeld/MMaDA-Parallel-M", padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len= 256, special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),ignore_id=-100, cond_dropout_prob=0.1, use_reserved_token=True)

    vq_model = get_vq_model_class("magvitv2")
    vq_model = vq_model.from_pretrained("showlab/magvitv2").to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = MMadaModelLM.from_pretrained("tyfeld/MMaDA-Parallel-M", trust_remote_code=True, torch_dtype=torch.bfloat16)

    model.to(device)

    result_table = wandb.Table(columns=["image", "edit_instruction", "output_text", "output_image"])
    image_list = []
    edit_instruction_list = []
    for file in os.listdir(config.interleave_root):
        if file.endswith('.jpg'):
            image_list.append(Image.open(os.path.join(config.interleave_root, file)))
            with open(os.path.join(config.interleave_root, file.replace('.jpg', '.txt')), 'r') as f:
                edit_instruction_list.append(f.read())

    for image, edit_instruction in zip(image_list, edit_instruction_list):
        print(f"image: {image}, edit_instruction: {edit_instruction}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image_transform_squash(image, resolution=512).to(device).unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        uncond_image_tokens = torch.zeros_like(image_tokens)

        input_text_ids = uni_prompting.text_tokenizer(edit_instruction)["input_ids"]

        uncond_input_text_ids = uni_prompting.text_tokenizer("")["input_ids"]

        if input_text_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
            input_text_ids = [uni_prompting.text_tokenizer.bos_token_id] + input_text_ids
        input_text_ids = input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] 

        if len(uncond_input_text_ids) == 0 or uncond_input_text_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
            uncond_input_text_ids = [uni_prompting.text_tokenizer.bos_token_id] + uncond_input_text_ids
        uncond_input_text_ids = uncond_input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] 

        if len(uncond_input_text_ids) < len(input_text_ids):
            uncond_input_text_ids = uncond_input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] * (len(input_text_ids) - len(uncond_input_text_ids))

        input_interleave_ids = torch.cat([
            torch.tensor([reserved_token_mapping['<|interleave|>']]).to(device),
            torch.tensor([reserved_token_mapping['<|soi|>']]).to(device),
            image_tokens[0],
            torch.tensor([reserved_token_mapping['<|eoi|>']]).to(device),
            torch.tensor(input_text_ids).to(device)
        ])

        uncond_input_interleave_ids = torch.cat([
            torch.tensor([reserved_token_mapping['<|interleave|>']]).to(device),
            torch.tensor([reserved_token_mapping['<|soi|>']]).to(device),
            uncond_image_tokens[0],
            torch.tensor([reserved_token_mapping['<|eoi|>']]).to(device),
            torch.tensor(uncond_input_text_ids).to(device)
        ])

        output_image_ids, output_text_ids = model.interleave_generate(
            input_interleave_ids,
            uncond_input_interleave_ids,
            text_cfg = 2.5,
            image_cfg = 4.0,
            noise_schedule= get_mask_schedule("cosine"),
            text_steps = 128,
            image_steps = 30,
            reserved_token_mapping = reserved_token_mapping,
            uni_prompting = uni_prompting,
        )
        
        output_text = uni_prompting.text_tokenizer.batch_decode(output_text_ids, skip_special_tokens=True)
        output_image = vq_model.decode_code(output_image_ids)
        print(f"output_text: {output_text}")
        output_image = torch.clamp((output_image + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
        output_image = output_image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(output_image[0])

        result_table.add_data(wandb.Image(image), edit_instruction, output_text[0], wandb.Image(pil_image))

    wandb.log({
        "result_table": result_table
    })

