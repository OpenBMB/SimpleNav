from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.context_index import ContextIndexConfig
from tool.navvla.schema import NavVLADatasetSpec, NavVLAEpisode


class NavVLASourceAdapter(ABC):
    name: str

    def configure(self, **kwargs: Any) -> "NavVLASourceAdapter":
        unknown = sorted(key for key, value in kwargs.items() if value is not None)
        if unknown:
            raise ValueError(f"adapter {self.name} does not support options: {unknown}")
        return self

    @abstractmethod
    def load_episodes(
        self,
        source_root: str | Path,
        *,
        split: str = "train",
        max_episodes: int | None = None,
    ) -> list[NavVLAEpisode]:
        raise NotImplementedError

    def convert(
        self,
        *,
        source_root: str | Path,
        output_root: str | Path,
        dataset_name: str,
        max_episodes: int | None,
        fps: float,
        action_horizon: int,
        overwrite: bool,
        control_frequency_hz: float | None = None,
        repair_existing: bool = False,
        split: str = "train",
        context_policy_version: str = "bats-v1",
        cache_policy_version: str = "smoke-coarse-v1",
        cache_workers: int | None = None,
        write_visual_token_cache: bool = True,
        visual_token_profile: Any | None = None,
        visual_token_encoder: Any | None = None,
        visual_token_encoder_factory: Any | None = None,
        episodes_per_file: int = 20,
        files_per_chunk: int = 50,
        context_index_config: ContextIndexConfig | None = None,
    ) -> dict[str, Any]:
        episodes = self.load_episodes(source_root, split=split, max_episodes=max_episodes)
        spec = NavVLADatasetSpec(
            dataset_name=dataset_name,
            fps=fps,
            action_horizon=action_horizon,
            action_dim=4,
            state_dim=4,
            control_frequency_hz=control_frequency_hz if control_frequency_hz is not None else fps,
            context_policy_version=context_policy_version,
            cache_policy_version=cache_policy_version,
            split=split,
            episodes_per_file=episodes_per_file,
            files_per_chunk=files_per_chunk,
        )
        return write_navvla_lerobot_dataset(
            episodes,
            output_root=Path(output_root),
            spec=spec,
            overwrite=overwrite,
            repair_existing=repair_existing,
            cache_workers=cache_workers,
            write_visual_token_cache=write_visual_token_cache,
            visual_token_profile=visual_token_profile,
            visual_token_encoder=visual_token_encoder,
            visual_token_encoder_factory=visual_token_encoder_factory,
            context_index_config=context_index_config,
        )


_ADAPTERS: dict[str, NavVLASourceAdapter] = {}


def register_adapter(adapter: NavVLASourceAdapter) -> NavVLASourceAdapter:
    if not adapter.name:
        raise ValueError("adapter name must be non-empty")
    if adapter.name in _ADAPTERS:
        raise ValueError(f"adapter already registered: {adapter.name}")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def get_adapter(name: str) -> NavVLASourceAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown NavVLA source adapter: {name}") from exc
