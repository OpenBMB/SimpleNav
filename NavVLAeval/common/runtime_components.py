from __future__ import annotations

from typing import Any

from NavVLAeval.common.model.action_codec import ActionCodec
from NavVLAeval.common.config import EvalConfig, load_class
from NavVLAeval.common.data.runtime_dataset import get_runtime_dataset_adapter


def build_benchmark_runtime(cfg: EvalConfig):
    spec_cls = load_class(cfg.benchmark.class_path)
    spec = spec_cls(**cfg.benchmark.kwargs)
    return spec.create_runtime(cfg)


def build_model(cfg: EvalConfig):
    class_path = cfg.model.kwargs.get("model_class_path")
    if not class_path:
        raise ValueError("model.model_class_path is required for worker execution")
    model_cls = load_class(str(class_path))
    kwargs = {
        key: value
        for key, value in cfg.model.kwargs.items()
        if key
        not in {
            "model_class_path",
            "action_codec_class_path",
            "model_outputs_raw",
            "framework_name",
        }
    }
    kwargs.setdefault("checkpoint", cfg.model.checkpoint)
    return model_cls(**kwargs)


def build_action_codec(cfg: EvalConfig):
    class_path = cfg.model.kwargs.get("action_codec_class_path")
    if class_path:
        codec_cls = load_class(str(class_path))
        kwargs = {
            key: value
            for key, value in cfg.model.kwargs.items()
            if key.startswith("action_codec_") and key != "action_codec_class_path"
        }
        return codec_cls(**_strip_prefix(kwargs, "action_codec_"))
    return ActionCodec.from_checkpoint(
        cfg.model.checkpoint,
        unnorm_key=cfg.model.unnorm_key,
        framework_name=_optional_str(cfg.model.kwargs.get("framework_name")),
        model_outputs_raw=bool(cfg.model.kwargs.get("model_outputs_raw", False)),
    )


def build_runtime_dataset(cfg: EvalConfig):
    class_path = cfg.dataset.raw.get("runtime_dataset_class_path")
    if class_path:
        dataset_cls = load_class(str(class_path))
        kwargs = dict(cfg.dataset.raw.get("runtime_dataset_kwargs") or {})
        return dataset_cls(**kwargs)
    dataset = get_runtime_dataset_adapter(cfg.dataset.raw)
    action_codec = build_action_codec(cfg)
    dataset.set_action_stats(action_codec.action_stats)
    return dataset


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_prefix(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key[len(prefix) :]: value for key, value in payload.items()}
