from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class LeRobotGTActionEvalModel:
    """Diagnostic model that replays GT action chunks from a NavVLA-LeRobot root."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        dataset_root: str | Path,
        episode_id: str | int,
        gt_unnorm_key: str | None = None,
        action_horizon: int = 8,
        **_: Any,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.episode_id = str(episode_id)
        self.action_horizon = int(action_horizon)
        self._step = 0
        self._actions = self._load_actions()
        self._action_stats = self._load_action_stats(str(gt_unnorm_key or "vln_val_seen_train"))

    def predict_action(self, example: dict[str, Any]) -> dict[str, Any]:
        del example
        index = min(self._step, self._actions.shape[0] - 1)
        raw_action = self._actions[index]
        self._step += 1
        from starVLA.dataloader.airsim_utils import normalize_array

        normalized = normalize_array(raw_action, self._action_stats)
        return {
            "normalized_actions": normalized.astype(np.float32),
            "metadata": {
                "source": "lerobot_gt_action",
                "episode_id": self.episode_id,
                "gt_frame_index": int(index),
            },
        }

    def _load_actions(self) -> np.ndarray:
        import pandas as pd

        episode_paths = sorted((self.dataset_root / "meta" / "episodes").glob("chunk-*/part-*.parquet"))
        if not episode_paths:
            raise FileNotFoundError(f"No episode metadata parquet files under {self.dataset_root / 'meta' / 'episodes'}")
        episodes = pd.concat((pd.read_parquet(path) for path in episode_paths), ignore_index=True)
        rows = episodes[episodes["episode_id"].astype(str) == self.episode_id]
        if rows.empty:
            raise KeyError(f"episode_id={self.episode_id!r} not found in {self.dataset_root}")
        row = rows.iloc[0]
        data_path = (
            self.dataset_root
            / "data"
            / f"chunk-{int(row['data/chunk_index']):03d}"
            / f"part-{int(row['data/file_index']):03d}.parquet"
        )
        data = pd.read_parquet(data_path)
        episode_rows = data[data["episode_index"].astype(int) == int(row["episode_index"])].sort_values("frame_index")
        if episode_rows.empty:
            raise KeyError(f"episode_index={int(row['episode_index'])} not found in {data_path}")
        actions = np.stack([np.vstack(value).astype(np.float32) for value in episode_rows["action"].to_numpy()]).astype(np.float32)
        if actions.ndim != 3 or actions.shape[1:] != (self.action_horizon, 4):
            raise ValueError(f"Expected GT action shape [T, {self.action_horizon}, 4], got {actions.shape}")
        return actions

    def _load_action_stats(self, unnorm_key: str) -> dict[str, Any]:
        candidates = []
        if self.checkpoint.is_file():
            candidates.extend([self.checkpoint.parent / "dataset_statistics.json", self.checkpoint.parent.parent / "dataset_statistics.json"])
        else:
            candidates.extend([self.checkpoint / "dataset_statistics.json", self.checkpoint.parent / "dataset_statistics.json"])
        candidates.append(self.dataset_root / "dataset_statistics.json")
        for path in candidates:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if unnorm_key not in data:
                    continue
                return dict(data[unnorm_key]["action"])
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"Action stats for {unnorm_key!r} not found; searched: {searched}")


class StarVLAEvalModel:
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        repo_root: str | Path | None = None,
        config_overrides: dict[str, Any] | None = None,
        inference_tvi_mask_probability: float | None = None,
        inference_seed: int | None = None,
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self.config_overrides = dict(config_overrides or {})
        self.inference_tvi_mask_probability = (
            None if inference_tvi_mask_probability is None else float(inference_tvi_mask_probability)
        )
        if self.inference_tvi_mask_probability is not None and not 0.0 <= self.inference_tvi_mask_probability <= 1.0:
            raise ValueError("inference_tvi_mask_probability must be between 0 and 1")
        self.inference_seed = None if inference_seed is None else int(inference_seed)
        self.predict_kwargs: dict[str, Any] = {}
        self.model = self._load_model()

    @classmethod
    def from_loaded_model(
        cls,
        model: Any,
        *,
        inference_seed: int | None = None,
        **predict_kwargs: Any,
    ) -> "StarVLAEvalModel":
        instance = cls.__new__(cls)
        instance.checkpoint = None
        instance.repo_root = None
        instance.config_overrides = {}
        instance.inference_tvi_mask_probability = None
        instance.inference_seed = None if inference_seed is None else int(inference_seed)
        instance.predict_kwargs = dict(predict_kwargs)
        instance.model = model
        return instance

    def predict_action(self, example: dict[str, Any]) -> dict[str, Any]:
        predict_kwargs = dict(self.predict_kwargs)
        if self.inference_tvi_mask_probability is not None:
            predict_kwargs["tvi_mask_probability"] = self.inference_tvi_mask_probability
        with self._prediction_rng(example) as derived_seed:
            output = self.model.predict_action(examples=[example], **predict_kwargs)
        if not isinstance(output, dict) or "normalized_actions" not in output:
            raise ValueError("StarVLA model predict_action must return a dict with normalized_actions")
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized.ndim == 3 and normalized.shape[0] == 1:
            normalized = normalized[0]
        result = dict(output)
        result["normalized_actions"] = normalized
        if derived_seed is not None:
            metadata = dict(result.get("metadata") or {})
            metadata["inference_seed"] = int(derived_seed)
            result["metadata"] = metadata
        return result

    @contextmanager
    def _prediction_rng(self, example: dict[str, Any]):
        if self.inference_seed is None:
            yield None
            return

        import torch

        derived_seed = _derive_prediction_seed(self.inference_seed, example)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(derived_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(derived_seed)
            yield derived_seed

    def encode_history_images(self, images: list[Any]) -> list[np.ndarray] | None:
        encoder = getattr(self.model, "encode_history_images", None)
        if encoder is None:
            return None
        return encoder(images)

    def _load_model(self):
        checkpoint = _resolve_checkpoint_file(self.checkpoint)
        if checkpoint is None or not checkpoint.exists():
            raise FileNotFoundError(f"missing StarVLA checkpoint: {self.checkpoint}")
        if self.repo_root is None or not self.repo_root.exists():
            raise FileNotFoundError(f"missing StarVLA repo root: {self.repo_root}")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from starVLA.model.framework.base_framework import baseframework

        model = baseframework.from_pretrained(
            str(checkpoint),
            **({"config_overrides": self.config_overrides} if self.config_overrides else {}),
        )
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        except Exception as exc:
            raise RuntimeError(f"failed to move StarVLA model to device: {exc}") from exc
        model.eval()
        return model


def _resolve_checkpoint_file(checkpoint: Path | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.is_dir():
        return checkpoint / "pytorch_model.pt"
    return checkpoint


def _derive_prediction_seed(inference_seed: int, example: dict[str, Any]) -> int:
    metadata = dict(example.get("metadata") or {})
    episode_uid = str(metadata.get("episode_uid") or metadata.get("episode_id") or "online")
    frame_index = int(metadata.get("frame_index", 0))
    digest = hashlib.sha256(f"{int(inference_seed)}:{episode_uid}:{frame_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63 - 1)
