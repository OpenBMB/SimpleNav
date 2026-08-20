from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from decord import VideoReader
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from starVLA.dataloader.qwenvl_llavajson.qwen_data_config import data_list
from starVLA.model.modules.vlm.MiniCPMV import IGNORE_INDEX, configure_minicpm_processor


def _read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank0_print(*args: Any) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args)


def _normalise_role(role: str) -> str:
    roles = {"human": "user", "gpt": "assistant"}
    return roles.get(str(role), str(role))


def _conversation_text(message: dict[str, Any]) -> str:
    return str(message.get("content", message.get("value", "")))


def _strip_visual_tags(text: str) -> str:
    return text.replace("<image>", "").replace("<video>", "").strip()


def _split_conversation(conversations: list[dict[str, Any]]) -> tuple[str, str]:
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in conversations:
        role = _normalise_role(message.get("role", message.get("from", "")))
        text = _strip_visual_tags(_conversation_text(message))
        if not text:
            continue
        if role == "assistant":
            assistant_parts.append(text)
        elif role == "user":
            user_parts.append(text)

    instruction = "\n".join(user_parts).strip()
    solution = "\n".join(assistant_parts).strip()
    if not instruction:
        raise ValueError("MiniCPM VLM sample is missing user instruction")
    if not solution:
        raise ValueError("MiniCPM VLM sample is missing assistant answer")
    return instruction, solution


def _resolve_media_paths(sample: dict[str, Any], key: str) -> list[Path]:
    values = sample.get(key, [])
    singular_key = key[:-1] if key.endswith("s") else key
    if not values and singular_key != key:
        values = sample.get(singular_key, [])
    if values is None:
        return []
    if isinstance(values, dict):
        values = list(values.values())
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    data_path = Path(sample.get("data_path", ""))
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            path = data_path / path
        paths.append(path)
    return paths


def _path_has_prefix(path: Path, prefixes: list[str]) -> bool:
    path_text = str(path)
    for prefix in prefixes:
        normalized_prefix = prefix.rstrip("/")
        if path_text == normalized_prefix or path_text.startswith(f"{normalized_prefix}/"):
            return True
    return False


def _filter_samples_by_media_path_prefix(samples: list[dict[str, Any]], prefixes: list[str]) -> list[dict[str, Any]]:
    if not prefixes:
        return samples
    kept_samples: list[dict[str, Any]] = []
    skipped_count = 0
    for sample in samples:
        media_paths = _resolve_media_paths(sample, "images") + _resolve_media_paths(sample, "videos")
        if media_paths and any(_path_has_prefix(path, prefixes) for path in media_paths):
            skipped_count += 1
            continue
        kept_samples.append(sample)
    _rank0_print(f"Skipped {skipped_count} MiniCPM VLM samples by media path prefixes: {prefixes}")
    return kept_samples


def _filter_samples_by_image_count(samples: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    kept_samples: list[dict[str, Any]] = []
    skipped_count = 0
    for sample in samples:
        if len(_resolve_media_paths(sample, "images")) > max_images:
            skipped_count += 1
            continue
        kept_samples.append(sample)
    _rank0_print(f"Skipped {skipped_count} MiniCPM VLM samples with more than {max_images} images")
    return kept_samples


def _rebase_dataset_specs(dataset_specs: list[dict[str, Any]], data_root_dir: str | None) -> list[dict[str, Any]]:
    if not data_root_dir:
        return dataset_specs
    root = Path(data_root_dir)
    rebased_specs: list[dict[str, Any]] = []
    for spec in dataset_specs:
        spec = dict(spec)
        annotation_name = Path(spec["annotation_path"]).name
        spec["annotation_path"] = str(root / "llava_jsons" / annotation_name)
        spec["data_path"] = str(root / "images")
        rebased_specs.append(spec)
    return rebased_specs


class MiniCPMVLMDataset(Dataset):
    def __init__(self, data_args: Any) -> None:
        super().__init__()
        dataset_names = str(data_args.dataset_use).split(",")
        dataset_specs = data_list(dataset_names)
        dataset_specs = _rebase_dataset_specs(dataset_specs, getattr(data_args, "data_root_dir", None))
        _rank0_print(f"Loading MiniCPM VLM datasets: {dataset_specs}")

        samples: list[dict[str, Any]] = []
        for spec in dataset_specs:
            annotation_path = spec["annotation_path"]
            file_format = str(annotation_path).rsplit(".", 1)[-1]
            if file_format == "jsonl":
                annotations = _read_jsonl(annotation_path)
            else:
                with open(annotation_path, "r", encoding="utf-8") as handle:
                    annotations = json.load(handle)

            sampling_rate = float(spec.get("sampling_rate", 1.0))
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                _rank0_print(f"Sampling {len(annotations)} examples from {annotation_path}")

            for annotation in annotations:
                annotation = dict(annotation)
                if spec.get("data_path", ""):
                    annotation["data_path"] = spec["data_path"]
                elif "raw_data" in annotation:
                    annotation["data_path"] = annotation["raw_data"].get("data_root", "")
                samples.append(annotation)

        skip_media_path_prefixes = list(getattr(data_args, "skip_media_path_prefixes", []))
        samples = _filter_samples_by_media_path_prefix(samples, skip_media_path_prefixes)
        samples = _filter_samples_by_image_count(samples, max_images=4)
        self.samples = self._pre_filter_long_case(samples, max_words=int(getattr(data_args, "model_max_length", 2048)))
        random.shuffle(self.samples)
        self.data_args = data_args
        _rank0_print(f"Total MiniCPM VLM training samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def _pre_filter_long_case(self, samples: list[dict[str, Any]], max_words: int) -> list[dict[str, Any]]:
        def count_words(sample: dict[str, Any]) -> int:
            return sum(len(_conversation_text(message).strip().split()) for message in sample.get("conversations", []))

        return [sample for sample in samples if count_words(sample) <= max_words]

    def _resize_image(self, image: Image.Image) -> Image.Image:
        resampling = getattr(Image, "Resampling", Image)
        return image.convert("RGB").resize((384, 384), resample=resampling.BICUBIC)

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Missing image file: {path}")
        with Image.open(path) as image:
            return self._resize_image(image)

    def _load_video_frames(self, path: Path) -> list[Image.Image]:
        if not path.exists():
            raise FileNotFoundError(f"Missing video file: {path}")
        vr = VideoReader(str(path), num_threads=int(getattr(self.data_args, "video_num_threads", 4)))
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / max(avg_fps, 1e-6)
        interval = float(getattr(self.data_args, "base_interval", 4))
        num_frames_to_sample = round(video_length / interval)
        min_frames = int(getattr(self.data_args, "video_min_frames", 4))
        max_frames = int(getattr(self.data_args, "video_max_frames", 8))
        target_frames = min(max(num_frames_to_sample, min_frames), max_frames)
        frame_idx = torch.linspace(0, total_frames - 1, target_frames).long().unique().tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        return [self._resize_image(Image.fromarray(frame)) for frame in frames]

    def _get_item(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        conversations = copy.deepcopy(sample.get("conversations", []))
        instruction, solution = _split_conversation(conversations)

        images: list[Image.Image] = []
        for image_path in _resolve_media_paths(sample, "images"):
            images.append(self._load_image(image_path))
        for video_path in _resolve_media_paths(sample, "videos"):
            images.extend(self._load_video_frames(video_path))

        if not images:
            raise ValueError("MiniCPM VLM sample has no images or videos")

        return {"images": images, "instruction": instruction, "solution": solution}

    def __getitem__(self, index: int) -> dict[str, Any]:
        num_base_retries = int(getattr(self.data_args, "num_base_retries", 3))
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(index)
            except Exception as exc:
                print(f"[MiniCPM VLM try #{attempt_idx}] Failed to fetch sample {index}. Exception: {exc}")
                time.sleep(1)

        for attempt_idx in range(num_base_retries):
            next_index = random.randrange(len(self.samples))
            try:
                return self._get_item(next_index)
            except Exception as exc:
                print(
                    f"[MiniCPM VLM fallback #{attempt_idx}] Failed to fetch sample {next_index}. Exception: {exc}"
                )

        return self._get_item(index)


@dataclass
class MiniCPMVLMDataCollator:
    processor: Any
    downsample_mode: str = "4x"
    max_slice_nums: int = 36
    use_image_id: bool = False
    enable_thinking: bool = False
    max_text_tokens: int = 2048

    def _truncate_text(self, text: str) -> str:
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)
        if len(token_ids) <= self.max_text_tokens:
            return str(text)
        return tokenizer.decode(token_ids[: self.max_text_tokens], skip_special_tokens=False)

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages: list[list[dict[str, Any]]] = []
        for instance in instances:
            content = [{"type": "image", "image": image} for image in instance["images"]]
            content.append({"type": "text", "text": self._truncate_text(instance["instruction"])})
            messages.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": str(instance["solution"])}]},
                ]
            )

        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            downsample_mode=self.downsample_mode,
            max_slice_nums=self.max_slice_nums,
            use_image_id=self.use_image_id,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch["labels"] = labels
        return dict(batch)


def make_cpm_vlm_dataloader(cfg: Any) -> dict[str, DataLoader]:
    data_args = cfg.datasets.vlm_data
    processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm, trust_remote_code=True)
    configure_minicpm_processor(processor, cfg)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.model_max_length = int(data_args.get("model_max_length", 2048))

    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    train_dataset = MiniCPMVLMDataset(data_args=data_args_ns)
    data_collator = MiniCPMVLMDataCollator(
        processor=processor,
        downsample_mode=str(cfg.framework.qwenvl.get("downsample_mode", "4x")),
        max_slice_nums=int(cfg.framework.qwenvl.get("max_slice_nums", 36)),
        use_image_id=bool(cfg.framework.qwenvl.get("use_image_id", False)),
        enable_thinking=bool(cfg.framework.qwenvl.get("enable_thinking", False)),
        max_text_tokens=int(cfg.framework.qwenvl.get("max_text_tokens", data_args.get("model_max_length", 2048))),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(data_args.per_device_batch_size),
        collate_fn=data_collator,
        num_workers=int(data_args.get("num_workers", 4)),
        shuffle=bool(data_args.get("shuffle", True)),
    )
    return {"train_dataloader": train_dataloader}
