from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from starVLA.model.modules.qwen35_vision import BFLOAT16_BITS_STORAGE_ENCODING

from NavVLAeval.common.types import EpisodeHistory
from starVLA.model.modules.bats import (
    BATSSelectionResult,
    online_bats_history_budget,
    select_bats_history,
    select_long_memory_candidate,
)
from tool.navvla.compute_bats_k import MINICPM_IMAGE_WRAPPER_TOKENS


@dataclass(frozen=True)
class OnlineHistoryConfig:
    history_policy: str = "recent"
    history_image_frames: int | None = None
    history_candidate_source_stride: int | None = None
    dataset_name: str = "online_eval"
    required_cameras: tuple[str, ...] = ("front",)
    bats_seed: int = 42
    bats_epsilon: float = 0.1
    bats_k: float = 4.0
    bats_use_dynamic_k: bool = True
    bats_sampling_mode: str = "priority_capped"
    bats_token_budget: int = 1024
    budget_num_cameras: int | None = None
    current_visual_tokens: int = 64
    history_visual_tokens: int = 4
    tvi_tokens: int = 1
    current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS
    history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS

    def __post_init__(self) -> None:
        if self.history_image_frames is not None and int(self.history_image_frames) < 0:
            raise ValueError(f"history_image_frames must be non-negative, got {self.history_image_frames}")
        if self.history_candidate_source_stride is not None and int(self.history_candidate_source_stride) <= 0:
            raise ValueError(
                "history_candidate_source_stride must be a positive integer, "
                f"got {self.history_candidate_source_stride}"
            )
        policy = str(self.history_policy)
        if policy not in {"recent", "uniform", "bats"}:
            raise ValueError(f"history_policy must be one of ['bats', 'recent', 'uniform'], got {policy!r}")
        if str(self.bats_sampling_mode) not in {"priority_capped", "independent"}:
            raise ValueError(
                "bats_sampling_mode must be one of ['independent', 'priority_capped'], "
                f"got {self.bats_sampling_mode!r}"
            )
        if not self.required_cameras:
            raise ValueError("required_cameras must not be empty")
        if min(int(self.current_wrapper_tokens), int(self.history_wrapper_tokens)) < 0:
            raise ValueError("visual wrapper token costs must be non-negative")


def build_online_navvla_history_sample(
    *,
    observation: dict[str, Any],
    history: EpisodeHistory,
    instruction: str,
    state: np.ndarray | None,
    config: OnlineHistoryConfig,
    platform_text: str = "",
) -> dict[str, Any]:
    current_frame_index = _observation_frame_index(observation, fallback=len(history.observations))
    current_timestamp = _observation_timestamp(observation, fallback=float(current_frame_index))
    selection = _select_history_observations(
        history=history,
        current_observation=observation,
        current_frame_index=current_frame_index,
        config=config,
    )

    primary_camera = str(config.required_cameras[0])
    images = {
        camera_name: np.asarray(_camera_image(observation, camera_name, primary_camera=primary_camera), dtype=np.uint8)
        for camera_name in config.required_cameras
        if _camera_image(observation, camera_name, primary_camera=primary_camera) is not None
    }
    history_images = {camera_name: [] for camera_name in config.required_cameras}
    history_tvi: list[list[float]] = []
    history_mask: list[bool] = []
    history_blocks: list[dict[str, Any]] = []
    history_steps: list[dict[str, Any]] = []
    history_records: list[tuple[str, np.ndarray | None, dict[str, Any] | None]] = []

    for selected_index, item in enumerate(selection.selected):
        frame_index = _observation_frame_index(item, fallback=selected_index)
        timestamp = _observation_timestamp(item, fallback=float(frame_index))
        history_steps.append({"index": frame_index, "frame_index": frame_index, "timestamp": timestamp})
        for camera_name in config.required_cameras:
            image = _camera_image(item, camera_name, primary_camera=primary_camera)
            cached_record = _online_visual_record(item, camera_name)
            if image is None and cached_record is None:
                continue
            history_tvi.append([timestamp, _camera_phi(camera_name)])
            history_mask.append(True)
            history_blocks.append(
                {
                    "step_index": selected_index,
                    "frame_index": frame_index,
                    "camera_name": camera_name,
                }
            )
            history_records.append(
                (
                    camera_name,
                    None if image is None else np.asarray(image, dtype=np.uint8),
                    cached_record,
                )
            )

    history_cached_embeds = None
    history_cached_grid_thw = None
    cache_stage = ""
    cache_profile = ""
    cache_encoder = ""
    cache_storage_encoding = ""
    record_modes = {
        "cache" if cached_record is not None else "image"
        for _camera_name, _image, cached_record in history_records
    }
    if len(record_modes) > 1:
        raise ValueError("online NavVLA history must be entirely cached or entirely raw images")
    if record_modes == {"cache"}:
        records = [cached_record for _camera_name, _image, cached_record in history_records if cached_record is not None]
        history_cached_embeds = np.stack([record["tokens"] for record in records], axis=0)
        stages = {str(record.get("cache_stage", "")) for record in records}
        profiles = {str(record.get("visual_token_profile", "")) for record in records if record.get("visual_token_profile")}
        encoders = {str(record.get("encoder_ckpt", "")) for record in records if record.get("encoder_ckpt")}
        storage_encodings = {str(record.get("storage_encoding", "")) for record in records}
        if len(stages) != 1 or len(profiles) > 1 or len(encoders) > 1 or len(storage_encodings) > 1:
            raise ValueError(
                f"online visual cache mixes contracts: stages={sorted(stages)} "
                f"profiles={sorted(profiles)} encoders={sorted(encoders)} "
                f"storage_encodings={sorted(storage_encodings)}"
            )
        cache_stage = next(iter(stages))
        cache_profile = next(iter(profiles), "")
        cache_encoder = next(iter(encoders), "")
        cache_storage_encoding = next(iter(storage_encodings), "")
        if cache_stage:
            grids = [record.get("grid_thw") for record in records]
            if any(grid is None for grid in grids):
                raise ValueError("pre-merge online visual cache requires grid_thw for every camera")
            history_cached_grid_thw = np.asarray(grids, dtype=np.int64).reshape(-1, 3)
    elif record_modes == {"image"}:
        for camera_name, image, _cached_tokens in history_records:
            if image is not None:
                history_images[camera_name].append(image)

    current_tvi = [[current_timestamp, _camera_phi(camera_name)] for camera_name in config.required_cameras]
    sample: dict[str, Any] = {
        "images": images,
        "current_tvi": np.asarray(current_tvi, dtype=np.float32).reshape(-1, 2),
        "history_images": history_images,
        "history_tvi": np.asarray(history_tvi, dtype=np.float32).reshape(-1, 2),
        "history_mask": np.asarray(history_mask, dtype=bool),
        "history_tokens": np.zeros((0, 1, 3), dtype=np.float32),
        "lang": instruction,
        "language": instruction,
        "platform_text": platform_text,
        "metadata": {
            "history_policy": str(config.history_policy),
            "history_candidate_source_stride": config.history_candidate_source_stride,
            "history_steps": history_steps,
            "history_blocks": history_blocks,
            "history_token_refs": [],
            "visual_token_mode": "cached_history_online_current" if history_cached_embeds is not None else "online_bats",
            "timestamp": current_timestamp,
            "frame_index": current_frame_index,
            "bats_k": float(config.bats_k),
            "bats_sampling_mode": str(config.bats_sampling_mode),
            "required_cameras": list(config.required_cameras),
            **({"visual_token_profile": cache_profile} if history_cached_embeds is not None and cache_profile else {}),
        },
    }
    if state is not None:
        sample["state"] = np.asarray(state, dtype=np.float32).reshape(-1)
    if history_cached_embeds is not None:
        sample["history_cached_embeds"] = np.asarray(history_cached_embeds, dtype=history_cached_embeds.dtype)
        sample["history_cached_mask"] = np.ones((history_cached_embeds.shape[0],), dtype=bool)
        if cache_stage:
            assert history_cached_grid_thw is not None
            sample["history_cached_grid_thw"] = history_cached_grid_thw
            sample["history_cached_cache_stage"] = cache_stage
            sample["history_cached_encoder_ckpt"] = cache_encoder
            sample["history_cached_storage_encoding"] = cache_storage_encoding
    if history.long_memory_tokens is not None:
        sample["long_memory_tokens"] = np.asarray(history.long_memory_tokens)
        sample["long_memory_tvi"] = np.asarray(history.long_memory_tvi, dtype=np.float32).reshape(-1, 2)
        sample["metadata"]["long_memory_blocks"] = list(history.long_memory_blocks)
    _attach_online_long_memory_update(
        sample=sample,
        selection=selection,
        history=history,
        required_cameras=config.required_cameras,
    )
    return sample


def _select_history_observations(
    *,
    history: EpisodeHistory,
    current_observation: dict[str, Any],
    current_frame_index: int,
    config: OnlineHistoryConfig,
) -> BATSSelectionResult:
    budget_max_frames = online_bats_history_budget(
        token_budget=int(config.bats_token_budget),
        budget_num_cameras=_budget_num_cameras(config),
        current_visual_tokens=int(config.current_visual_tokens),
        history_visual_tokens=int(config.history_visual_tokens),
        tvi_tokens=int(config.tvi_tokens),
        current_wrapper_tokens=int(config.current_wrapper_tokens),
        history_wrapper_tokens=int(config.history_wrapper_tokens),
    )
    configured_max_frames = None if config.history_image_frames is None else int(config.history_image_frames)
    max_frames = budget_max_frames if configured_max_frames is None else min(budget_max_frames, configured_max_frames)
    if max_frames <= 0:
        return BATSSelectionResult([], [], float(config.bats_k), 0)
    candidates = [
        (_observation_frame_index(item, fallback=index), item)
        for index, item in enumerate(history.observations)
        if ("image" in item or "images" in item or bool(item.get("navvla_online_visual_tokens")))
        and _observation_frame_index(item, fallback=index) < int(current_frame_index)
        and (
            config.history_candidate_source_stride is None
            or _observation_source_frame_index(item, fallback=index)
            % int(config.history_candidate_source_stride)
            == 0
        )
    ]
    if str(config.history_policy) == "bats":
        selection = select_bats_history(
            candidates=candidates,
            anchor_frame_index=int(current_frame_index),
            episode_id=str(_metadata_value(current_observation, "episode_id", "online")),
            dataset_name=str(config.dataset_name),
            seed=int(config.bats_seed),
            epsilon=float(config.bats_epsilon),
            k=float(config.bats_k),
            use_dynamic_bats_k=bool(config.bats_use_dynamic_k),
            sampling_mode=str(config.bats_sampling_mode),
            token_budget=int(config.bats_token_budget),
            budget_num_cameras=_budget_num_cameras(config),
            current_visual_tokens=int(config.current_visual_tokens),
            history_visual_tokens=int(config.history_visual_tokens),
            tvi_tokens=int(config.tvi_tokens),
            current_wrapper_tokens=int(config.current_wrapper_tokens),
            history_wrapper_tokens=int(config.history_wrapper_tokens),
            max_history_frames=configured_max_frames,
        )
        return selection
    if str(config.history_policy) == "uniform" and len(candidates) > int(max_frames):
        selected_indices = np.linspace(
            0,
            len(candidates) - 1,
            num=int(max_frames),
            dtype=np.int64,
        )
        selected = [candidates[index][1] for index in np.unique(selected_indices).tolist()]
    else:
        selected = [item for _frame_index, item in candidates[-int(max_frames):]]
    return BATSSelectionResult(selected, list(selected), float(config.bats_k), max_frames)


def _attach_online_long_memory_update(
    *,
    sample: dict[str, Any],
    selection: BATSSelectionResult,
    history: EpisodeHistory,
    required_cameras: tuple[str, ...],
) -> None:
    candidate = select_long_memory_candidate(
        selection.ranked_selected,
        memory_frame_indices=history.long_memory_frame_indices,
    )
    if candidate is None:
        return
    frame_index = _observation_frame_index(candidate, fallback=-1)
    timestamp = _observation_timestamp(candidate, fallback=float(frame_index))
    records: list[dict[str, Any]] = []
    tvi: list[list[float]] = []
    blocks: list[dict[str, Any]] = []
    for camera_name in required_cameras:
        cached = _online_visual_record(candidate, camera_name)
        if cached is None:
            return
        records.append(cached)
        tvi.append([timestamp, _camera_phi(camera_name)])
        blocks.append({"step_index": 0, "frame_index": frame_index, "camera_name": camera_name})
    if not records:
        return
    stages = {str(record.get("cache_stage", "")) for record in records}
    storage_encodings = {str(record.get("storage_encoding", "")) for record in records}
    if len(stages) != 1:
        raise ValueError(f"online long-memory source mixes cache stages: {sorted(stages)}")
    if len(storage_encodings) != 1:
        raise ValueError(
            f"online long-memory source mixes cache storage encodings: {sorted(storage_encodings)}"
        )
    sample["online_long_memory_update_tokens"] = np.stack([record["tokens"] for record in records], axis=0)
    stage = next(iter(stages))
    if stage:
        grids = [record.get("grid_thw") for record in records]
        if any(grid is None for grid in grids):
            raise ValueError("pre-merge online long-memory source requires grid_thw")
        sample["online_long_memory_update_grid_thw"] = np.asarray(grids, dtype=np.int64).reshape(-1, 3)
        sample["online_long_memory_update_cache_stage"] = stage
        sample["online_long_memory_update_storage_encoding"] = next(iter(storage_encodings))
        encoders = {str(record.get("encoder_ckpt", "")) for record in records if record.get("encoder_ckpt")}
        if len(encoders) > 1:
            raise ValueError(f"online long-memory source mixes cache encoders: {sorted(encoders)}")
        sample["online_long_memory_update_encoder_ckpt"] = next(iter(encoders), "")
    sample["online_long_memory_update_tvi"] = np.asarray(tvi, dtype=np.float32)
    sample["online_long_memory_update_mask"] = np.ones((len(records),), dtype=bool)
    sample["metadata"]["online_long_memory_update_blocks"] = blocks
    sample["metadata"]["online_long_memory_update_frame_index"] = int(frame_index)


def _budget_num_cameras(config: OnlineHistoryConfig) -> int:
    if config.budget_num_cameras is not None:
        return int(config.budget_num_cameras)
    return len(config.required_cameras)


def _camera_image(observation: dict[str, Any], camera_name: str, *, primary_camera: str) -> Any | None:
    images = observation.get("images")
    if isinstance(images, dict) and camera_name in images:
        return images[camera_name]
    if camera_name == primary_camera and "image" in observation:
        return observation["image"]
    return None


def _online_visual_tokens(observation: dict[str, Any], camera_name: str) -> np.ndarray | None:
    record = _online_visual_record(observation, camera_name)
    return None if record is None else record["tokens"]


def _online_visual_record(observation: dict[str, Any], camera_name: str) -> dict[str, Any] | None:
    cache = observation.get("navvla_online_visual_tokens")
    if not isinstance(cache, dict) or camera_name not in cache:
        return None
    value = cache[camera_name]
    payload = dict(value) if isinstance(value, dict) else {"tokens": value}
    tokens = np.asarray(payload.get("tokens"))
    if tokens.ndim != 2 or tokens.shape[0] <= 0:
        return None
    if str(payload.get("storage_encoding", "")) == BFLOAT16_BITS_STORAGE_ENCODING and tokens.dtype != np.uint16:
        raise TypeError(
            "online bfloat16_bits visual cache must use numpy uint16 tokens, "
            f"got {tokens.dtype}"
        )
    output = {**payload, "tokens": tokens}
    if output.get("grid_thw") is not None:
        grid = np.asarray(output["grid_thw"], dtype=np.int64).reshape(3)
        stage = str(output.get("cache_stage", ""))
        if stage == "vit_postmerge_pool4":
            valid = bool((grid > 0).all()) and int(grid[1]) % 2 == 0 and int(grid[2]) % 2 == 0
            valid = valid and int(tokens.shape[0]) == 4
        elif stage == "vit_postmerge":
            valid = int(grid.prod()) // 4 == int(tokens.shape[0])
        else:
            valid = int(grid.prod()) == int(tokens.shape[0])
        if not valid:
            raise ValueError(
                f"online visual cache grid {grid.tolist()} does not match cache token count {tokens.shape[0]}"
            )
        output["grid_thw"] = grid
    return output


def _observation_metadata(observation: dict[str, Any]) -> dict[str, Any]:
    metadata = observation.get("navvla_eval")
    if isinstance(metadata, dict):
        return metadata
    for container_key in ("traveluav_episode",):
        container = observation.get(container_key)
        if isinstance(container, dict) and isinstance(container.get("navvla_eval"), dict):
            return container["navvla_eval"]
    return {}


def _metadata_value(observation: dict[str, Any], key: str, default: Any) -> Any:
    metadata = _observation_metadata(observation)
    if key in metadata:
        return metadata[key]
    return observation.get(key, default)


def _observation_frame_index(observation: dict[str, Any], *, fallback: int) -> int:
    return int(_metadata_value(observation, "frame_index", fallback))


def _observation_source_frame_index(observation: dict[str, Any], *, fallback: int) -> int:
    return int(
        _metadata_value(
            observation,
            "source_frame_index",
            _observation_frame_index(observation, fallback=fallback),
        )
    )


def _observation_timestamp(observation: dict[str, Any], *, fallback: float) -> float:
    return float(_metadata_value(observation, "timestamp", fallback))


def _camera_phi(camera_name: str) -> float:
    if camera_name == "left":
        return float(np.pi / 2)
    if camera_name == "right":
        return float(-np.pi / 2)
    if camera_name == "rear":
        return float(np.pi)
    return 0.0
