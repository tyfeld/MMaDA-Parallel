import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from argparse import ArgumentParser
import json
import math
import pickle
from PIL import Image
import io
from datasets import load_dataset, concatenate_datasets

from data.item_processor import DimooItemProcessor
from data.item_processor import var_center_crop, generate_crop_size_list


def _load_image_from_field(field_value):
    """
    Accept either:
      - bytes -> return PIL.Image
      - PIL.Image -> return as-is (converted to RGB)
      - str path -> open from disk (if exists)
    Otherwise raise ValueError.
    """
    if field_value is None:
        raise ValueError("None image field")
    if isinstance(field_value, (bytes, bytearray)):
        return Image.open(io.BytesIO(field_value)).convert("RGB")
    try:
        from PIL import Image as PILImage
        if isinstance(field_value, PILImage.Image):
            return field_value.convert("RGB")
    except Exception:
        pass
    if isinstance(field_value, str) and os.path.exists(field_value):
        return Image.open(field_value).convert("RGB")

    raise ValueError(f"Unsupported image field type: {type(field_value)}")


class ParquetItemProcessor(DimooItemProcessor):
    """Enhanced pre-tokenization processor that directly reads images from arrow/parquet datasets."""
    
    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-DiMOO",
        vq_ckpt_path="Alpha-VLLM/Lumina-DiMOO",
        target_size=512,
    ):
        super().__init__(tokenizer, vq_ckpt_path, target_size)
        self.target_size = target_size
        try:
            _ = self.crop_size_list
        except Exception:
            self.crop_size_list = generate_crop_size_list((self.target_size // 32) ** 2, 32)

    def process_item(self, raw_item, task_type=None):
        def _get_field_image(keys):
            for k in keys:
                if k in raw_item and raw_item[k] is not None:
                    try:
                        return _load_image_from_field(raw_item[k])
                    except Exception as e:
                        print(f"Warning: failed to load image from key '{k}': {e}")
                        continue
            return None

        if task_type in ("edit", "ti2ti"):
            img_ori = _get_field_image(["input_image", "input_image_bytes", "image"])
            img_edit = _get_field_image(["output_image", "output_image_bytes", "edited_image"])

            if img_ori is None or img_edit is None:
                raise ValueError(
                    f"Missing input or output image for ti2ti task. "
                    f"Available keys: {list(raw_item.keys())}"
                )

            crop_size_list = generate_crop_size_list((self.target_size // 32) ** 2, 32)
            img_ori = var_center_crop(img_ori, crop_size_list=crop_size_list)
            img_edit = var_center_crop(img_edit, crop_size_list=crop_size_list)
            img_path = [img_ori, img_edit]

        elif task_type == "t2i":
            image = _get_field_image(["output_image", "image", "output_image_bytes", "image_bytes"])
            if image is None:
                raise ValueError(f"No image found in t2i item: keys={list(raw_item.keys())}")
            img_path = image

        elif task_type in ("mmu_single_image", "mmu"):
            image = _get_field_image(["input_image", "image", "input_image_bytes", "image_bytes"])
            if image is None:
                raise ValueError(f"No image found in mmu_single_image item: keys={list(raw_item.keys())}")

            area = image.size[0] * image.size[1]
            if area < self.target_size * self.target_size:
                crop_size_list = generate_crop_size_list((self.target_size // 32) ** 2, 32)
            elif area > 1024 * 1024:
                crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
            else:
                crop_size_list = [(image.size[0] // 16 * 16, image.size[1] // 16 * 16)]

            image = var_center_crop(image, crop_size_list=crop_size_list)
            img_path = image

        elif task_type in ("mmu_multi_image",):
            iterable = None
            for key in ["image_list", "images", "input_images"]:
                if key in raw_item and raw_item[key]:
                    iterable = raw_item[key]
                    break

            if iterable is None:
                raise ValueError(f"No image list found for mmu_multi_image. keys={list(raw_item.keys())}")

            img_list = []
            for image_entry in iterable:
                im = _load_image_from_field(
                    image_entry["image"] if isinstance(image_entry, dict) else image_entry
                )
                crop_size_list = generate_crop_size_list((self.target_size // 32) ** 2, 32)
                im = var_center_crop(im, crop_size_list=crop_size_list)
                img_list.append(im)
            img_path = img_list

        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        caption = None
        for key in ("input_text", "prompt", "caption", "instruction", "text"):
            if key in raw_item and raw_item.get(key) is not None:
                caption = raw_item.get(key)
                break

        if caption is None:
            raise ValueError(f"No prompt/caption/instruction found in item. keys={list(raw_item.keys())}")

        return super().process_item(caption, img_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--type", type=str, default="ti2ti")
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--dataset_ids",
        type=str,
        required=True,
        help="Comma-separated HuggingFace dataset ids"
    )
    args = parser.parse_args()

    print(f"Initializing ParquetItemProcessor with target_size={args.target_size}...")
    item_processor = ParquetItemProcessor(target_size=args.target_size)

    dataset_ids = [d.strip() for d in args.dataset_ids.split(",") if d.strip()]
    print(f"Loading datasets: {dataset_ids}")

    all_datasets = []
    for did in dataset_ids:
        try:
            print(f"  Loading {did}...")
            ds = load_dataset(did, split="train")
            print(f"    Loaded {len(ds)} samples")
            all_datasets.append(ds)
        except Exception as e:
            print(f"  Warning: Failed to load {did}: {e}")

    if not all_datasets:
        raise RuntimeError("Failed to load any datasets!")

    ori_contents = concatenate_datasets(all_datasets)
    print(f"Total number of items: {len(ori_contents)}")

    num = len(ori_contents)
    if args.max_samples is not None:
        num = min(num, args.max_samples)
        print(f"Limited to max_samples: {num}")

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)
    start_idx = num_per_rank * rank
    end_idx = min(num_per_rank * (rank + 1), num)

    # Check for existing progress
    progress_file = os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt")
    try:
        with open(progress_file, "r") as f:
            content = f.read().strip()
            if content == "finished":
                print(f"Rank {rank} already finished. Exiting.")
                sys.exit(0)
            start_idx = int(content) + 1
        print(f"Resuming from index {start_idx}")
    except Exception:
        print(f"Starting from index {start_idx}")

    print(f"Rank {rank}/{splits}: Processing items {start_idx} to {end_idx-1}")

    # Statistics
    processed_count = 0
    skipped_count = 0

    for i in range(start_idx, end_idx):
        if i % 100 == 0:
            print(f"Rank {rank}: Progress {i}/{end_idx} (processed: {processed_count}, skipped: {skipped_count})")
        
        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        
        try:
            raw_item = ori_contents[i]
            new_item = item_processor.process_item(raw_item, task_type=args.type)

            # Helper to extract answer text
            def _extract_answer_text(d):
                for k in ("output_text", "reasoning_text", "answer_text", "explanation", "reasoning"):
                    if k in d and d.get(k):
                        val = d.get(k)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                return ""

            if args.type == "t2i":
                with open(pkl_path, "wb") as f:
                    pickle.dump(new_item, f)
                
                prompt = raw_item.get("input_text", raw_item.get("prompt", raw_item.get("caption", "")))
                record = {
                    "system_prompt": "Generate an image according to the text prompt.",
                    "user_prompt": prompt,
                    "user_image": "",
                    "answer_text": "",
                    "answer_image": pkl_path,
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item.get("input_ids", []))
                }

            elif args.type in ("edit", "ti2ti"):
                # new_item is [img_ori_tokens, img_edit_tokens]
                pkl_path0 = os.path.join(save_dir, f"{i}_0.pkl")
                pkl_path1 = os.path.join(save_dir, f"{i}_1.pkl")
                
                with open(pkl_path0, "wb") as f:
                    pickle.dump(new_item[0], f)
                with open(pkl_path1, "wb") as f:
                    pickle.dump(new_item[1], f)

                # Extract answer_text from output_text field
                answer_text = _extract_answer_text(raw_item)
                if not answer_text:
                    # Fallback to instruction if no output_text
                    answer_text = raw_item.get("instruction", "")

                user_prompt = raw_item.get("input_text", raw_item.get("prompt", raw_item.get("instruction", "")))

                record = {
                    "system_prompt": "Generate an image applying the following editing instruction based on the original image.",
                    "user_prompt": user_prompt,
                    "user_image": pkl_path0,
                    "answer_text": answer_text,
                    "answer_image": pkl_path1,
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item[0].get("input_ids", []))
                }

            elif args.type == "mmu":
                if isinstance(new_item, list):
                    pkl_paths = []
                    for idx, item in enumerate(new_item):
                        path = os.path.join(save_dir, f"{i}_{idx}.pkl")
                        with open(path, "wb") as f:
                            pickle.dump(item, f)
                        pkl_paths.append(path)
                    pkl_path_used = pkl_paths
                else:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(new_item, f)
                    pkl_path_used = pkl_path

                user_prompt = ""
                answer_text = ""
                for k in raw_item.get("conversations", []):
                    if k.get("from") == "human":
                        user_prompt = k.get("value", "").replace("<image>\n", "")
                    elif k.get("from") == "gpt":
                        answer_text = k.get("value", "")

                record = {
                    "system_prompt": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images.",
                    "user_prompt": user_prompt,
                    "user_image": pkl_path_used,
                    "answer_text": answer_text,
                    "answer_image": "",
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item.get("input_ids", [])) if isinstance(new_item, dict) else (len(new_item[0].get("input_ids", [])) if isinstance(new_item, list) and len(new_item) else 0)
                }

            else:
                raise ValueError(f"Unsupported task_type: {args.type}")

            processed_count += 1

        except Exception as e:
            from traceback import format_exc
            print(f"Error processing item {i}:")
            print(f"  Raw item keys: {list(ori_contents[i].keys()) if i < len(ori_contents) else 'N/A'}")
            print(f"  Error: {e}")
            print(format_exc())
            skipped_count += 1

        # Write record if successfully processed
        if record is not None:
            record_file = os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl")
            with open(record_file, "a", encoding="utf-8") as f:
                record_str = json.dumps(record, ensure_ascii=False) + "\n"
                f.write(record_str)

        # Update progress
        with open(progress_file, "w", encoding="utf-8") as f:
            if i == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{i}")

    print(f"Rank {rank} finished!")
    print(f"  Total processed: {processed_count}")
    print(f"  Total skipped: {skipped_count}")