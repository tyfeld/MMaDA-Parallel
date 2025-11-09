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
    '<|t2it|>': 126096,
}

from bdb import effective
import os
import glob
from pydoc import text
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_DATASETS_CACHE"] = "/data_storage/ty/.cache/huggingface/datasets"
os.environ["HF_HOME"] = "/data_storage/ty/.cache/huggingface"
import json
import pandas
import logging
import math
import shutil
import time
import html
import random
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from training.utils import get_config, flatten_omega_conf, image_transform, image_transform_squash
from training.imagenet_dataset import ImageNetDataset
from torchvision import transforms
from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from models.configuration_llada import ActivationCheckpointingStrategy

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch.nn.functional as F

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def image_transform_squash_sample(sample, resolution=512):
    transform_pipeline = transforms.Compose([
        transforms.Resize((resolution,resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])

    # sample是一个batch，包含多个样本
    batch_size = len(sample["input_text"])
    is_text_only_list = sample["is_text_only"]
    
    processed_input_images = []
    processed_output_images = []
    
    for i in range(batch_size):
        is_text_only = bool(is_text_only_list[i])
        
        if not is_text_only:
            # 处理有input_image的样本
            input_image = sample["input_image"][i]
            output_image = sample["output_image"][i]
            
            # 确保是单个PIL图像
            if isinstance(input_image, list):
                input_image = input_image[0]
            if isinstance(output_image, list):
                output_image = output_image[0]
                
            input_image = transform_pipeline(input_image)
            output_image = transform_pipeline(output_image)
        else:
            # 处理text_only样本
            output_image = sample["output_image"][i]
            if isinstance(output_image, list):
                output_image = output_image[0]
            output_image = transform_pipeline(output_image)
            
            # 创建占位图像
            input_image = transform_pipeline(Image.new("RGB", (resolution, resolution), (0, 0, 0)))
        
        processed_input_images.append(input_image)
        processed_output_images.append(output_image)
    
    sample["input_image"] = processed_input_images
    sample["output_image"] = processed_output_images
    return sample

def main():

    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
    )

    total_batch_size_per_gpu = config.training.batch_size
    total_batch_size = (
            config.training.batch_size
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.tokenizer_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob, use_reserved_token=True)


    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize mmada in pretraining s
    # base_config = AutoConfig.from_pretrained(config.model.mmada.pretrained_model_path).to_dict()
    # mmada_config_dict = {k: v for k, v in config.model.mmada.items()}
    # merged_config = {**base_config, **mmada_config_dict}
    # mmada_config = MMadaConfig(**merged_config)
    # model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16, config=mmada_config)
    # model.resize_token_embeddings(mmada_config.new_vocab_size)
    # model.config.embedding_size = model.config.vocab_size
    model = MMadaModelLM.from_pretrained(config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16).to(accelerator.device)
    strategy_to_use = ActivationCheckpointingStrategy.fine_grained
    model.model.set_activation_checkpointing(strategy_to_use)

    mask_id = model.config.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    
    
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params
    

    # yjf: run scaled datasets
    base_shard_pattern = "/data_storage/shared/datasets/processed/mmada_scaled_ti2i_shards/shard_*.tar"
    shard_list = sorted(glob.glob(base_shard_pattern))
    logger.info(f"Found shards: {shard_list}")

    if len(shard_list) == 0:
        raise FileNotFoundError(f"No shards found with pattern {base_shard_pattern}. Please check the path.")

    try:
        raw_dataset = load_dataset("webdataset", data_files=shard_list, split="train")
    except Exception as e:
        logger.error("Failed to load shards via webdataset. Exception: %s", str(e))
        try:
            raw_dataset = load_from_disk("/data_storage/shared/datasets/processed/mmada_scaled_ti2i_shards")
        except Exception as e2:
            logger.error("Also failed to load via load_from_disk. Exception: %s", str(e2))
            raise

    logger.info(f"Loaded dataset columns: {raw_dataset.column_names}")

    # 打印第一个样本的结构以便调试
    if len(raw_dataset) > 0:
        logger.info(f"First sample keys: {list(raw_dataset[0].keys())}")
        for key in list(raw_dataset[0].keys())[:5]:  # 只打印前5个key
            value = raw_dataset[0][key]
            logger.info(f"  {key}: type={type(value).__name__}, preview={str(value)[:50] if isinstance(value, (str, bytes)) else type(value).__name__}")


    def normalize_example(ex):
        """标准化样本，处理webdataset格式的字段名"""
        
        # webdataset加载后的字段名是 "input_text.txt" 而不是 "input_text"
        # 需要同时检查两种可能的字段名
        
        # 1. 获取input_text
        input_text = None
        for key in ["input_text.txt", "input_text", "instruction", "prompt", "text"]:
            value = ex.get(key)
            if value:
                # 如果是bytes，解码
                if isinstance(value, bytes):
                    try:
                        input_text = value.decode('utf-8').strip()
                    except:
                        continue
                else:
                    input_text = str(value).strip()
                
                if input_text:  # 找到非空值就退出
                    break
        
        if not input_text:
            input_text = ""
            logger.warning(f"Empty input_text found! Available keys: {list(ex.keys())}")
        
        # 2. 获取output_text
        output_text = None
        for key in ["output_text.txt", "output_text", "reasoning_text", "caption", "answer"]:
            value = ex.get(key)
            if value:
                if isinstance(value, bytes):
                    try:
                        output_text = value.decode('utf-8').strip()
                    except:
                        continue
                else:
                    output_text = str(value).strip()
                
                if output_text:
                    break
        
        if not output_text:
            output_text = ""
            logger.warning(f"Empty output_text found! Available keys: {list(ex.keys())}")
        
        # 3. 获取图像
        def get_image_from_field(d, cands):
            """从字典中获取图像"""
            for k in cands:
                if k not in d or d[k] is None:
                    continue
                
                v = d[k]
                
                # 如果已经是PIL Image
                if isinstance(v, Image.Image):
                    return v
                
                # 如果是bytes
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return Image.open(io.BytesIO(v)).convert("RGB")
                    except Exception as e:
                        logger.debug(f"Failed to decode image from bytes for key {k}: {e}")
                        continue
                
                # 如果是dict（某些webdataset格式）
                if isinstance(v, dict):
                    if 'bytes' in v:
                        try:
                            return Image.open(io.BytesIO(v['bytes'])).convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to decode image from dict bytes for key {k}: {e}")
                            continue
                    if 'path' in v and os.path.exists(v['path']):
                        try:
                            return Image.open(v['path']).convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to load image from path for key {k}: {e}")
                            continue
                
                # 如果是list
                if isinstance(v, list) and len(v) > 0:
                    first = v[0]
                    if isinstance(first, Image.Image):
                        return first
                    if isinstance(first, (bytes, bytearray)):
                        try:
                            return Image.open(io.BytesIO(first)).convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to decode image from list bytes for key {k}: {e}")
                            continue
                    if isinstance(first, str) and os.path.exists(first):
                        try:
                            return Image.open(first).convert("RGB")
                        except Exception as e:
                            logger.debug(f"Failed to load image from list path for key {k}: {e}")
                            continue
                
                # 如果是路径字符串
                if isinstance(v, str) and os.path.exists(v):
                    try:
                        return Image.open(v).convert("RGB")
                    except Exception as e:
                        logger.debug(f"Failed to load image from path string for key {k}: {e}")
                        continue
            
            return None
        
        # webdataset的字段名是 "input.jpg" 而不是 "input_image"
        img_candidates = ["input.jpg", "input_image", "image.jpg", "image", "img"]
        out_candidates = ["output.jpg", "output_image", "output_image.jpg", "out_img"]
        
        input_image = get_image_from_field(ex, img_candidates)
        output_image = get_image_from_field(ex, out_candidates)
        
        #判断是否为text_only任务
        is_text_only = (input_image is None)
        
        # 如果output_image也是None，创建一个占位图像
        if output_image is None:
            logger.warning(f"Missing output_image, creating placeholder. Available keys: {list(ex.keys())}")
            output_image = Image.new("RGB", (512, 512), (0, 0, 0))
        
        # 如果是text_only但input_image是None，创建占位图像
        if is_text_only and input_image is None:
            input_image = Image.new("RGB", (512, 512), (0, 0, 0))
        
        return {
            "input_text": input_text,
            "output_text": output_text,
            "input_image": input_image,
            "output_image": output_image,
            "is_text_only": is_text_only,
        }

    logger.info(f"Raw dataset columns before mapping: {raw_dataset.column_names}")

    # 使用num_proc=1避免多进程问题，并添加详细日志
    ds_norm = raw_dataset.map(
        normalize_example, 
        remove_columns=raw_dataset.column_names,
        num_proc=1,
        desc="Normalizing dataset"
    )

    logger.info(f"Columns after normalize: {ds_norm.column_names}")

    # 检查前几个样本
    if len(ds_norm) > 0:
        logger.info("Checking first normalized sample:")
        first_sample = ds_norm[0]
        logger.info(f"  Keys: {list(first_sample.keys())}")
        logger.info(f"  input_text length: {len(first_sample.get('input_text', ''))}")
        logger.info(f"  output_text length: {len(first_sample.get('output_text', ''))}")
        logger.info(f"  input_text preview: {first_sample.get('input_text', '')[:100]}")
        logger.info(f"  output_text preview: {first_sample.get('output_text', '')[:100]}")
        logger.info(f"  input_image type: {type(first_sample.get('input_image'))}")
        logger.info(f"  output_image type: {type(first_sample.get('output_image'))}")
        logger.info(f"  is_text_only: {first_sample.get('is_text_only')}")

    # 设置transform
    ds_norm.set_transform(lambda sample: image_transform_squash_sample(sample, resolution=config.dataset.preprocessing.resolution))

    # 分割训练和验证集
    train_val_split = ds_norm.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    logger.info(f"成功加载数据集，训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=total_batch_size_per_gpu, 
        shuffle=True, 
        num_workers=config.dataset.params.num_workers, 
        pin_memory=config.dataset.params.pin_memory, 
        persistent_workers=config.dataset.params.persistent_workers, 
        drop_last=True
    )

    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=total_batch_size_per_gpu, 
        shuffle=False,  # No shuffling for validation
        num_workers=config.dataset.params.num_workers, 
        pin_memory=config.dataset.params.pin_memory, 
        persistent_workers=config.dataset.params.persistent_workers, 
        drop_last=False  # Keep all validation samples
    )    

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    logger.info(f"num_update_steps_per_epoch: {num_update_steps_per_epoch}")
    logger.info(f"num_train_epochs: {num_train_epochs}")

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0
    start_step = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        logger.info(f"dirs: {dirs}")
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        logger.info(f"path: {path}")
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = start_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            if os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            # 如果是分片模型文件
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin.index.json'):
                # 从索引文件加载模型
                from safetensors.torch import load_file
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            # if safetensors sharded checkpoint exists
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(
                    model, 
                    f'{path}/unwrapped_model/',
                    # weight_map=None, 
                    # load_state_dict_fn="safetensors"
                )
            else:
                raise FileNotFoundError(f"Checkpoint {path}/unwrapped_model/pytorch_model.bin not found")
    else:
        logger.info("Not resuming from checkpoint")

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader)

    vq_model.to(device=accelerator.device)

    # if hasattr(model, 'module'):
    #     mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    # else:
    #     mask_dtype = model.showo.model.embed_tokens.weight.dtype  
    mask_dtype = model.get_input_embeddings().weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels_for_interleave_data(
            input_pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            input_text: Union[str, list[str]],
            output_pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            output_text: Union[str, list[str]],
            eps: float = 1e-3, 
            text_tokenizer: AutoTokenizer = None,
            mask_id: int = None,
            is_text_only_mask: torch.BoolTensor | None = None,
            seed: int = None,
            cond_dropout_prob: float = 0.0,
            max_text_len: int = None,   # Include bos and eos. Not Include task sign.
    ):
        if not text_tokenizer:
            raise ValueError("You should give a text tokenizer.")
        if not mask_id:
            raise ValueError("You should give a mask_id.")

        device = input_pixel_values.device
        
        # create image tokens using vq_model
        input_image_tokens = vq_model.get_code(input_pixel_values)
        input_image_tokens = input_image_tokens + len(text_tokenizer)
        batch_size = input_image_tokens.shape[0]
        if is_text_only_mask is None:
            is_text_only_mask = torch.zeros(batch_size, dtype=torch.bool, device=input_pixel_values.device)
        for i in range(batch_size):
            if is_text_only_mask[i]:
                input_image_tokens[i] = torch.zeros_like(input_image_tokens[i])

        input_text_ids = text_tokenizer(input_text)['input_ids']
        output_text_ids = text_tokenizer(output_text)['input_ids']

        output_image_tokens = vq_model.get_code(output_pixel_values)
        output_image_tokens = output_image_tokens + len(text_tokenizer)

        # adding noise scheduler
        
        t = torch.rand(batch_size, device=input_image_tokens.device)
        t = t * (1 - eps) + eps  # t is the mask probability

        # mask image tokens
        batch_size, output_image_seq_len = output_image_tokens.shape
        mask_prob = mask_schedule(t).clip(eps)   # mask_prob's shape is (batch_size)
        mask_prob = torch.cos(mask_prob * math.pi * 0.5)
        num_token_masked = (output_image_seq_len * mask_prob).round().clamp(min=1) # num_token_masked's shape is (batch_size, 1)
        batch_randperm = torch.rand(batch_size, output_image_seq_len, device=input_image_tokens.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        masked_output_image_ids = torch.where(mask, mask_id, output_image_tokens)
        output_image_labels = torch.where(mask, output_image_tokens, -100)
        loss_weight = None
        
        # mask text tokens 
        text_masked_indices = torch.rand(batch_size, max_text_len, device=input_image_tokens.device) < mask_prob.unsqueeze(1)
        # print(f"text_masked_indices: {text_masked_indices}")
        # don't mask the first token
        text_masked_indices[:, 0] = False
        # noisy_output_text_ids = torch.where(text_masked_indices, mask_id, output_text_ids)
        # output_text_labels = torch.where(text_masked_indices, output_text_ids, -100)


        dropout_text_probs = torch.rand(batch_size) # for randomly dropout text condition
        dropout_image_probs = torch.rand(batch_size) # for randomly dropout image condition

        output_sequences_ids = []
        output_labels_ids = []
        output_attention_masks = []

        for i in range(batch_size):

            task_token_id = reserved_token_mapping['<t2it>'] if is_text_only_mask[i] else reserved_token_mapping['<|interleave|>']


            # create the task sign and text part
            # ensure the first token is bos_token_id

            # first let's design the input part 

            if len(input_text_ids[i]) == 0 or input_text_ids[i][0] != text_tokenizer.bos_token_id:
                input_text_ids[i] = [text_tokenizer.bos_token_id] + input_text_ids[i]
            # ensure the last token is eos_token
            if input_text_ids[i][-1] != text_tokenizer.eos_token_id:
                input_text_ids[i] = input_text_ids[i] + [text_tokenizer.eos_token_id]

            # randomly dropout input text 
            if dropout_text_probs[i] < cond_dropout_prob:
                input_text_ids[i] = [text_tokenizer.bos_token_id, text_tokenizer.eos_token_id]

            if dropout_image_probs[i] < cond_dropout_prob:
                input_image_tokens[i] = torch.zeros_like(input_image_tokens[i])

            if max_text_len >= len(input_text_ids[i]):
                # mask padding ones
                input_text_padding_masks = [1] * (len(input_text_ids[i]) + 3 + 
                input_image_tokens.shape[-1]) + [0] * (max_text_len - len(input_text_ids[i]))
                input_text_padding_masks = torch.tensor(input_text_padding_masks, device=device)

                input_text_ids[i] = input_text_ids[i] + [text_tokenizer.eos_token_id] * (max_text_len - len(input_text_ids[i]))
            else:
                # mask padding ones
                input_text_padding_masks = torch.tensor([1] * (max_text_len + 3 + input_image_tokens.shape[-1]), device=device)

                input_text_ids[i] = input_text_ids[i][:max_text_len-1] + [text_tokenizer.eos_token_id]

            # combine input image and input text

            input_interleave_ids = torch.cat([
                torch.tensor([task_token_id], device=device),
                torch.tensor([reserved_token_mapping['<|soi|>']], device=device),
                input_image_tokens[i],
                torch.tensor([reserved_token_mapping['<|eoi|>']], device=device),
                torch.tensor(input_text_ids[i], device=device),
            ])
            

            # input don't have labels, do not calculate CE loss
            input_interleave_labels = [torch.tensor([-100], device=device)] * len(input_interleave_ids)
            input_interleave_labels = torch.tensor(input_interleave_labels, device=device)


            # now for output part 
            if len(output_text_ids[i]) == 0 or output_text_ids[i][0] != text_tokenizer.bos_token_id:
                output_text_ids[i] = [text_tokenizer.bos_token_id] + output_text_ids[i]
            # ensure the last token is eos_token
            if output_text_ids[i][-1] != text_tokenizer.eos_token_id:
                output_text_ids[i] = output_text_ids[i] + [text_tokenizer.eos_token_id]

            if max_text_len >= len(output_text_ids[i]):
                output_text_padding_masks = torch.tensor([1] * (len(output_text_ids[i]) + 2 + output_image_tokens.shape[-1]) + [0] * (max_text_len - len(output_text_ids[i])), device=device)
                output_text_ids[i] = output_text_ids[i] + [text_tokenizer.eos_token_id] * (max_text_len - len(output_text_ids[i]))
            else:
                output_text_padding_masks = torch.tensor([1] * (max_text_len + 2 + output_image_tokens.shape[-1]), device=device)
                output_text_ids[i] = output_text_ids[i][:max_text_len-1] + [text_tokenizer.eos_token_id]

            output_text_ids[i] = torch.tensor(output_text_ids[i], device=device)

            # add noise to output_text_ids 
            output_noisy_text_ids = torch.where(text_masked_indices[i], mask_id, output_text_ids[i])
            output_text_labels = torch.where(text_masked_indices[i], output_text_ids[i], -100)



            output_interleave_ids = torch.cat([
                torch.tensor([reserved_token_mapping['<|soi|>']], device=device),
                masked_output_image_ids[i],
                torch.tensor([reserved_token_mapping['<|eoi|>']], device=device),
                torch.tensor(output_noisy_text_ids, device=device),
            ])

            output_interleave_labels = torch.cat([
                torch.tensor([-100], device=device),
                output_image_labels[i],
                torch.tensor([-100], device=device),
                output_text_labels,
            ])

            # now let's combine the input and output
            sequence_ids = torch.cat([
                input_interleave_ids,
                output_interleave_ids,
            ], dim=0)

            label_ids = torch.cat([
                input_interleave_labels,
                output_interleave_labels,
            ], dim=0)
            
            # print(f"input_text_padding_masks: {input_text_padding_masks.shape}")
            # print(f"output_text_padding_masks: {output_text_padding_masks.shape}")
            all_mask = torch.cat([input_text_padding_masks, output_text_padding_masks], dim=0)
            # print(f"all_mask: {all_mask.shape}")

            # print("all shapes: ")
            # print(f"sequence_ids.shape: {sequence_ids.shape}")
            # print(f"label_ids.shape: {label_ids.shape}")
            # print(f"all_mask.shape: {all_mask.shape}")
            
            
            


            output_sequences_ids.append(sequence_ids.unsqueeze(0))
            output_labels_ids.append(label_ids.unsqueeze(0))
            output_attention_masks.append(all_mask.unsqueeze(0))

        return torch.cat(output_sequences_ids, dim=0), torch.cat(output_labels_ids, dim=0), torch.cat(output_attention_masks, dim=0), t



    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    micro_step = 0

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in train_dataloader:
            # 实时处理图像
            input_pixel_values = batch["input_image"]
            output_pixel_values = batch["output_image"]
            # 移动到设备
            input_pixel_values = input_pixel_values.to(accelerator.device, non_blocking=True)
            output_pixel_values = output_pixel_values.to(accelerator.device, non_blocking=True)
            
            # 获取其他数据
            input_text = batch["input_text"]
            output_text = batch["output_text"]
            batch_size = len(input_text)
            
            # print(f"batch_size: {batch_size}, batch: {batch}")
            
            # 继续后续处理...
            # for loss calculation
            # batch_size = len(batch["input_text"])
            # print(f"batch_size: {batch_size}, batch: {batch}")
            # input_pixel_values, input_text, output_pixel_values, output_text = batch["input_image"], batch["input_text"], batch["output_image"], batch["output_text"]
            # input_pixel_values = input_pixel_values.to(accelerator.device, non_blocking=True)
            # output_pixel_values = output_pixel_values.to(accelerator.device, non_blocking=True)
            # data_time_m.update(time.time() - end)
            
            semi_ar_mask = False

            is_text_only_mask = torch.tensor([bool(x) for x in batch["is_text_only"]], dtype=torch.bool, device=accelerator.device)
            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids_interleave,
                labels_interleave,
                interleave_masks,
                t
            ) = prepare_inputs_and_labels_for_interleave_data(
                input_pixel_values=input_pixel_values, 
                input_text=input_text, 
                output_pixel_values=output_pixel_values,
                output_text=output_text,
                eps=config.training.min_masking_rate,
                text_tokenizer=uni_prompting.text_tokenizer,
                mask_id=mask_id,
                cond_dropout_prob=config.training.cond_dropout_prob,
                max_text_len=config.dataset.preprocessing.max_seq_length
                )
            logits = model.forward(
                input_ids = input_ids_interleave,
                attention_mask = interleave_masks,
            ).logits

            unscaled_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels_interleave.view(-1), ignore_index=-100, reduction='none').view(batch_size, -1)
            # loss = unscaled_loss / t.view(batch_size, 1)
            # print(f"labels_interleave: {labels_interleave.shape}")
            text_unscaled_loss = unscaled_loss[:, - config.dataset.preprocessing.max_seq_length:] / t.view(batch_size, 1)
            # # print(f"text_unscaled_loss: {text_unscaled_loss.shape}, {text_unscaled_loss}")  
            # num_text_valid_tokens = (labels_interleave[:, - config.dataset.preprocessing.max_seq_length:] != -100).sum()
            # # print(f"num_text_valid_tokens: {num_text_valid_tokens}")
            text_loss = text_unscaled_loss.sum() / config.dataset.preprocessing.max_seq_length


            image_loss = unscaled_loss[:, - config.dataset.preprocessing.max_seq_length - config.model.mmada.num_vq_tokens - 2 : - config.dataset.preprocessing.max_seq_length].mean()
            # image_unscaled_loss = unscaled_loss[:, - config.dataset.preprocessing.max_seq_length - config.model.mmada.num_vq_tokens - 2 : - config.dataset.preprocessing.max_seq_length] / t.view(batch_size, 1)
            # # print(f"image_unscaled_loss: {image_unscaled_loss.shape}, {image_unscaled_loss}")
            # num_image_valid_tokens = (labels_interleave[:, - config.dataset.preprocessing.max_seq_length - config.model.mmada.num_vq_tokens - 2 : - config.dataset.preprocessing.max_seq_length] != -100).sum()
            # # print(f"num_image_valid_tokens: {num_image_valid_tokens}")
            # image_loss = image_unscaled_loss.sum() / num_image_valid_tokens
            # image_loss = image_unscaled_loss.sum()
            # print(f"unscaled_loss: {unscaled_loss.shape}, t.shape: {t.shape}")

            loss = text_loss * config.training.text_coeff + image_loss * config.training.image_coeff

            # print(f"loss: {loss}, text_loss: {text_loss}, image_loss: {image_loss}")

            # num_valid_tokens = (labels_interleave != -100).sum()

            # loss = loss.sum() / num_valid_tokens


            
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss_interleave = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
            avg_text_loss = accelerator.gather(text_loss.repeat(config.training.batch_size)).mean()
            avg_image_loss = accelerator.gather(image_loss.repeat(config.training.batch_size)).mean()

            # avg_masking_rate = accelerator.gather(mask_prob_t2i.repeat(config.training.batch_size_t2i)).mean()
            accelerator.backward(loss)
            if config.training.max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)
            optimizer.step()
            lr_scheduler.step()            
            optimizer.zero_grad(set_to_none=True)

            batch_time_m.update(time.time() - end)
            end = time.time()

            # Log metrics
            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                )
                logs = {
                    "step_loss_interleave": avg_loss_interleave.item(),
                    "step_text_loss": avg_text_loss.item(),
                    "step_image_loss": avg_image_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                }
                accelerator.log(logs, step=global_step + 1)
                logger.info(
                    f"Step: {global_step + 1} "
                    f"Loss_interleave: {avg_loss_interleave.item():0.4f} "
                    f"Text Loss: {avg_text_loss.item():0.4f} "
                    f"Image Loss: {avg_image_loss.item():0.4f} "
                    f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_m.val:0.4f} "
                    f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                )

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()

            # if (global_step + 1) % config.experiment.val_every == 0:
            #     val_loss_t2i = 0
            #     for i in range(10):
            #         val_loss_t2i += validate_t2i(seed=(i+config.training.validation_seed))
            #     val_loss_t2i = val_loss_t2i / 10
            #     accelerator.log({"val_loss_t2i": val_loss_t2i.item()}, step=global_step + 1)
            #     logger.info(f"Validation loss at step {global_step + 1}: {val_loss_t2i.item():0.4f}")

            # Save model checkpoint
            if (global_step + 1) % config.experiment.save_every == 0:
                save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

            if ((global_step + 1) % config.experiment.generate_every == 0) and accelerator.is_main_process:

                generate_interleave(
                    model,
                    vq_model,
                    uni_prompting,
                    accelerator,
                    config,
                    global_step + 1,
                    mask_schedule,
                    val_dataloader,
                )
            global_step += 1
            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for
    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)

    accelerator.end_training()





@torch.no_grad()
def generate_interleave(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
        val_dataloader,
):
    logger.info("Generating interleave...")
    model.eval()
    image_list = []
    edit_instruction_list = []
    ground_truth_text_list = []
    ground_truth_image_list = []
    is_text_only_list = []
    result_table = wandb.Table(columns=["task", "image", "edit_instruction", "output_text", "output_image", "ground_truth_text", "ground_truth_image"])

    if val_dataloader is not None:
        batch_count = 0
        for batch in val_dataloader:
            if batch_count >= 2:  # 只处理前两个batch
                break
            # 将batch中的每个样本分别添加到列表中
            for i in range(len(batch["input_image"])):
                image_list.append(batch["input_image"][i])
                edit_instruction_list.append(batch["input_text"][i])
                ground_truth_text_list.append(batch["output_text"][i])
                ground_truth_image_list.append(batch["output_image"][i])
                is_text_only_list.append(batch["is_text_only"][i])
            batch_count += 1
    else:
        for file in os.listdir(config.validation.interleave_root):
            if file.endswith('.jpg'):
                image_list.append(Image.open(os.path.join(config.validation.interleave_root, file)))
                with open(os.path.join(config.validation.interleave_root, file.replace('.jpg', '.txt')), 'r') as f:
                    edit_instruction_list.append(f.read())




    # mask_dtype = model.get_input_embeddings().weight.dtype
    # mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    # image_tokens = torch.ones((len(validation_prompts), config.model.mmada.num_vq_tokens), dtype=torch.long,
    #                           device=accelerator.device) * mask_token_id
    # input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    # if not force_no_cfg and config.training.guidance_scale > 0:
    #     uncond_input_ids, uncond_attention_mask = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
    #     cfg_scale = config.training.guidance_scale
    # else:
    #     uncond_input_ids = None
    #     uncond_attention_mask = None
    #     cfg_scale = 0
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    device = accelerator.device

    for i, (image, edit_instruction) in enumerate(zip(image_list, edit_instruction_list)):
        is_text_only = bool(is_text_only_list[i]) if len(is_text_only_list) > i else False
        print(f"is_text_only: {is_text_only}")
        device = accelerator.device
        # image = image_transform_squash(image, resolution=config.dataset.params.resolution).to(device).unsqueeze(0)
        image = image.to(device).unsqueeze(0)
        # image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        if is_text_only:
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            image_tokens = torch.zeros_like(image_tokens)
        else:
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        uncond_image_tokens = torch.zeros_like(image_tokens)
        # print(f" image_tokens shape: {image_tokens.shape}")


        task_token_id = reserved_token_mapping['<|t2i|>'] if is_text_only \
                        else reserved_token_mapping['<|interleave|>']
        input_text_ids = uni_prompting.text_tokenizer(edit_instruction)["input_ids"]
        # print(f"input_text_ids: {input_text_ids}")

        uncond_input_text_ids = uni_prompting.text_tokenizer("")["input_ids"]

        if input_text_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
            input_text_ids = [uni_prompting.text_tokenizer.bos_token_id] + input_text_ids
        input_text_ids = input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] 

        if len(uncond_input_text_ids) == 0 or uncond_input_text_ids[0] != uni_prompting.text_tokenizer.bos_token_id:
            uncond_input_text_ids = [uni_prompting.text_tokenizer.bos_token_id] + uncond_input_text_ids
        uncond_input_text_ids = uncond_input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] 

        if len(uncond_input_text_ids) < len(input_text_ids):
            uncond_input_text_ids = uncond_input_text_ids + [uni_prompting.text_tokenizer.eos_token_id] * (len(input_text_ids) - len(uncond_input_text_ids))

        # print(f"input_text_ids: {input_text_ids}, shape: {input_text_ids.shape}")

        input_interleave_ids = torch.cat([
            torch.tensor([task_token_id]).to(device),
            torch.tensor([reserved_token_mapping['<|soi|>']]).to(device),
            image_tokens[0],
            torch.tensor([reserved_token_mapping['<|eoi|>']]).to(device),
            torch.tensor(input_text_ids).to(device)
        ])

        uncond_input_interleave_ids = torch.cat([
            torch.tensor([task_token_id]).to(device),
            torch.tensor([reserved_token_mapping['<|soi|>']]).to(device),
            uncond_image_tokens[0],
            torch.tensor([reserved_token_mapping['<|eoi|>']]).to(device),
            torch.tensor(uncond_input_text_ids).to(device)
        ])

        # print(f"input_interleave_ids shape: {input_interleave_ids.shape}, uncond_input_interleave_ids shape: {uncond_input_interleave_ids.shape}")

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            output_image_ids, output_text_ids = model.interleave_generate(
                input_interleave_ids,
                uncond_input_interleave_ids,
                text_cfg = config.training.get("text_cfg", 0.0),
                image_cfg = config.training.get("image_cfg", 3.5),
                noise_schedule= mask_schedule,
                text_steps = config.training.get("text_steps", 128),
                image_steps = config.training.get("image_steps", 30),
                reserved_token_mapping = reserved_token_mapping,
                config = config,
                uni_prompting = uni_prompting,
            )
        
        output_text = uni_prompting.text_tokenizer.batch_decode(output_text_ids, skip_special_tokens=True)
        output_image = vq_model.decode_code(output_image_ids)

        output_image = torch.clamp((output_image + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
        output_image = output_image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(output_image[0])
        # pil_image.save(os.path.join(config.interleave_root, f"{image.split('/')[-1].split('.')[0]}_output.jpg"))

        result_table.add_data("t2i" if is_text_only else "interleave", wandb.Image(image[0]), edit_instruction, output_text[0], wandb.Image(pil_image), ground_truth_text_list[i], wandb.Image(ground_truth_image_list[i]))

    wandb.log({
        "result_table": result_table
    }, step=global_step)


    
    # with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
    #     # Generate images
    #     gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
    #         input_ids=input_ids,
    #         uncond_input_ids=uncond_input_ids,
    #         attention_mask=attention_mask,
    #         uncond_attention_mask=uncond_attention_mask,
    #         guidance_scale=cfg_scale,
    #         temperature=config.training.get("generation_temperature", 1.0),
    #         timesteps=config.training.generation_timesteps,
    #         noise_schedule=mask_schedule,
    #         noise_type=config.training.get("noise_type", "mask"),
    #         predict_all_tokens=config.training.get("predict_all_tokens", False),
    #         seq_len=config.model.mmada.num_vq_tokens,
    #         uni_prompting=uni_prompting,
    #         config=config,
    #     )
    # # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # # so we clamp them to the correct range.
    # gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    # images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # # Convert to PIL images
    # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    # images *= 255.0
    # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    # pil_images = [Image.fromarray(image) for image in images]

    # # Log images
    # wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    # wandb.log({f"Generated images with cfg {cfg_scale}": wandb_images}, step=global_step)






@torch.no_grad()
def understanding_images(
        model,
        vq_model,
        uni_prompting, # 包含了 text_tokenizer
        accelerator,
        config,
        global_step,
):
    """
    Processes images and multi-turn conversation prompts for image understanding,
    generates responses, and logs results to Weights & Biases.
    Uses tokenizer.apply_chat_template for handling conversation history.
    """
    logger.info("Understanding images (multi-turn)...")
    model.eval()
    prompts_file_path = config.dataset.params.mmu_validation_prompts_file
    image_root = config.dataset.params.mmu_image_root
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            validation_data = json.load(f) # 加载整个 JSON 列表
        logger.info(f"Successfully loaded {len(validation_data)} validation items from {prompts_file_path}")
    except Exception as e:
        logger.error(f"Error loading prompts from {prompts_file_path}: {e}. Skipping image understanding.")
        model.train() # 确保模型状态恢复
        return
    wandb_logs = []
    device = accelerator.device
    # 确定混合精度类型
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    for item in validation_data:
        file_name = item.get('file_name')
        messages = item.get('messages') # 获取消息列表
        if not file_name or not messages:
            logger.warning(f"Skipping item due to missing 'file_name' or 'messages': {item}")
            continue
        image_path = os.path.join(image_root, file_name)
        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}. Skipping.")
            continue
        try:
            # --- 图像处理 ---
            image_ori = Image.open(image_path).convert("RGB")
            # 根据文件名判断使用何种图像变换（保持与原代码逻辑一致）
            if any(tag in file_name for tag in ['ai2d', 'clevr', 'docvqa', 'geo', 'llava']):
                 image = image_transform_squash(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            else:
                 image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
            image = image.unsqueeze(0) # 增加 batch 维度 (1, C, H, W)
            # --- VQ 编码 ---
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer) # (1, num_img_tokens)
            # print(len(uni_prompting.text_tokenizer))
            batch_size = image_tokens.shape[0] # 确保 batch size 为 1
            # --- 使用 apply_chat_template 处理文本对话 ---
            # add_generation_prompt=True 会在末尾添加 assistant 角色的起始提示，以便模型续写
            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True, # 重要：为生成下一轮回复做准备
                return_tensors="pt"
            ).to(device) # (1, num_text_tokens)
            # --- 构建最终输入 ---
            # 格式: <|mmu|> <|soi|> image_tokens <|eoi|> templated_chat_tokens
            # templated_chat_tokens 已经包含了模板添加的 BOS (如 <|startoftext|>) 和角色标记
            input_ids = torch.cat([
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                text_token_ids # 直接拼接，因为 apply_chat_template 已处理好文本部分
            ], dim=1).long()
            # --- 模型生成 ---
            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                output_ids = accelerator.unwrap_model(model).mmu_generate(
                    input_ids,
                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                    # 使用与原代码一致的生成参数，可按需调整
                    steps=config.dataset.preprocessing.max_seq_length // 2,
                    block_length=config.dataset.preprocessing.max_seq_length // 4,
                )
            # --- 解码生成的文本 ---
            # output_ids 包含了输入 input_ids，需要切片获取新生成的部分
            generated_ids = output_ids[:, input_ids.shape[1]:]
            response_text = uni_prompting.text_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # --- 准备 W&B 日志 ---
            # 将完整的多轮对话（包括生成的回复）格式化为字符串，用于 W&B caption
            conversation_str = f"Image: {file_name}\n" + "="*20 + "\n"
            for msg in messages:
                role_prefix = "User: " if msg['role'] == 'user' else "Assistant: "
                conversation_str += f"{role_prefix}{msg['content']}\n"
            # 添加模型生成的回复
            conversation_str += f"Assistant (Generated): {response_text}\n"
            # 将 PyTorch 张量图像转换回 PIL Image 以便 W&B 显示
            log_image_tensor = torch.clamp((image.squeeze(0) + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
            log_image_np = log_image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pil_image = Image.fromarray(log_image_np)
            # 添加到日志列表
            wandb_logs.append(wandb.Image(pil_image, caption=conversation_str.strip()))
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}", exc_info=True) # 添加堆栈跟踪信息
    # --- 统一记录到 W&B ---
    if wandb_logs:
        try:
            wandb.log({"Understanding images (multi-turn)": wandb_logs}, step=global_step)
            logger.info(f"Logged {len(wandb_logs)} understanding image results to W&B for step {global_step}.")
        except Exception as e:
            logger.error(f"Failed to log understanding images to W&B: {e}")
    else:
        logger.warning("No images were successfully processed for understanding in this step.")
    model.train() # 确保在函数结束时将模型设置回训练模式

@torch.no_grad()
def generate_chat_text(
        model,
        uni_prompting,
        accelerator,
        config,
        global_step,
):
    logger.info("Generating chat text...")
    model.eval()

    # 读取数据，获取 prompt 列表
    df = pandas.read_json(config.dataset.params.lm_chat_validation_jsonl, lines=True)
    prompts = df['question'].tolist()
    responses = [''] * len(prompts)

    device = accelerator.device

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # 累积所有 prompt/response 对的 HTML 内容，作为一个整体 log 到 wandb
    html_content = "<div style='font-family:Arial, sans-serif;'>"
    html_content += f"<h2 style='color:navy;'>Step {global_step}</h2>"


    for i, prompt in enumerate(prompts):
        # 原始 prompt 用于展示
        original_prompt = prompt

        # 构造生成输入
        prompt_with_tags = "<|start_header_id|>user<|end_header_id|>\n" + f"{prompt}" + "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
        token_ids = uni_prompting.text_tokenizer([prompt_with_tags])['input_ids'][0]
        token_ids = [uni_prompting.text_tokenizer.bos_token_id] + token_ids
        input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            output_ids = accelerator.unwrap_model(model).mmu_generate(
                input_ids, 
                max_new_tokens=config.dataset.preprocessing.max_seq_length, 
                steps=config.dataset.preprocessing.max_lm_text_length // 2, 
                block_length=config.dataset.preprocessing.max_seq_length // 4
            )
        text = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        responses[i] += text[0]

        # 将每一组 prompt 和 response 的展示信息添加到 HTML 中
        escaped_prompt = html.escape(original_prompt)
        escaped_response = html.escape(responses[i])
        html_content += f"""
        <div style='border: 1px solid #ddd; margin:10px 0; padding:10px;'>
          <h4 style='margin: 0;'>Prompt</h4>
          <p style='margin: 0;'>{escaped_prompt}</p>
          <h4 style='margin: 0; margin-top:5px;'>Response</h4>
          <p style='margin: 0;'>{escaped_response}</p>
        </div>
        """

    html_content += "</div>"  # 结束整体容器

    model.train()

    # 在一个 step 内统一 log 生成的单个 HTML 对象（这样就不会多次 log 同一个 step）
    wandb.log({"chat_text_generation": wandb.Html(html_content)}, step=global_step)


    # # 打印所有问答对
    # logger.info("\n===== chat generated =====")
    # for i, (prompt, response) in enumerate(zip(prompts, responses)):
    #     logger.info(f"\nQuestion {i+1}：{prompt}")
    #     logger.info(f"\nAnswer {i+1}：{response}")
    #     logger.info("-" * 50)

def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

        # save tokenizer
        uni_prompting.text_tokenizer.save_pretrained(save_path/ "unwrapped_model")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def process_batch_images(batch, resolution=512):
    """在训练过程中实时处理batch中的图像"""
    transform_pipeline = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    
    batch_size = len(batch["input_text"])
    is_text_only_list = batch["is_text_only"]
    
    processed_input_images = []
    processed_output_images = []
    
    for i in range(batch_size):
        is_text_only = bool(is_text_only_list[i])
        
        if not is_text_only:
            # 处理有input_image的样本
            input_image = batch["input_image"][i]
            output_image = batch["output_image"][i]
            
            # 确保是单个PIL图像
            if isinstance(input_image, list):
                input_image = input_image[0]
            if isinstance(output_image, list):
                output_image = output_image[0]
                
            input_image = transform_pipeline(input_image)
            output_image = transform_pipeline(output_image)
        else:
            # 处理text_only样本
            output_image = batch["output_image"][i]
            if isinstance(output_image, list):
                output_image = output_image[0]
            output_image = transform_pipeline(output_image)
            
            # 创建占位图像
            input_image = transform_pipeline(Image.new("RGB", (resolution, resolution), (0, 0, 0)))
        
        processed_input_images.append(input_image)
        processed_output_images.append(output_image)
    
    # 转换为tensor并堆叠
    input_pixel_values = torch.stack(processed_input_images)
    output_pixel_values = torch.stack(processed_output_images)
    
    return input_pixel_values, output_pixel_values


if __name__ == "__main__":
    main()
