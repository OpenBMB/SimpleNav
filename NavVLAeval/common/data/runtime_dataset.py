from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from NavVLAeval.common.data.runtime_history import OnlineHistoryConfig, build_online_navvla_history_sample
from NavVLAeval.common.runtime_defaults import BaseRuntimeDatasetAdapter
from NavVLAeval.common.types import ActionPrediction, EnvironmentStepResult, EpisodeHistory
from NavVLAeval.traveluav.sample_builder import (
    TRAVELUAV_CAMERA_INDEX_BY_NAVVLA_NAME,
    TRAVELUAV_NAVVLA_PLATFORM_TEXT,
    build_navvla_history_state,
    build_traveluav_stage_instruction,
)
from tool.navvla.compute_bats_k import MINICPM_IMAGE_WRAPPER_TOKENS
from tool.navvla.statistics import build_repeated_state_statistics, normalize_values


AERIALVLN_PROMPT_PREFIX = "this is aerialvln. "
AERIALVLN_PLATFORM_TEXT = AERIALVLN_PROMPT_PREFIX + (
    "The platform is UAV for urban uav navigation. The control frequency is 1 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)
OPENFLY_PLATFORM_TEXT = (
    "The platform is UAV for urban uav navigation. The control frequency is 1 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)
UAVFLOW_PLATFORM_TEXT = (
    "The platform is UAV for uav vla. The control frequency is 5 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)
VLNCE_PLATFORM_TEXT = (
    "The platform is Indoor Robot for indoor robot navigation. The control frequency is 1 Hz. "
    "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
)


class RuntimeDatasetAdapter(Protocol):
    def prepare_observation_for_step(self, *, observation: dict[str, Any], step: int) -> dict[str, Any]:
        ...

    def build_example(self, *, observation: dict[str, Any], history: EpisodeHistory, instruction: str) -> dict[str, Any]:
        ...

    def update_history(
        self,
        *,
        history: EpisodeHistory,
        observation: dict[str, Any],
        prediction: ActionPrediction,
        instruction: str,
    ) -> EpisodeHistory:
        ...

    def set_action_stats(self, action_stats: dict[str, Any]) -> None:
        ...


class OnlineNavVLARuntimeDatasetAdapter(BaseRuntimeDatasetAdapter):
    def __init__(
        self,
        *,
        required_cameras: tuple[str, ...] = ("front",),
        history_image_frames: int | None = None,
        history_candidate_source_stride: int | None = None,
        state_dim: int = 0,
        action_horizon: int = 8,
        history_selection: str = "bats",
        history_update_mode: str = "action_observations",
        dataset_name: str = "online_eval",
        bats_seed: int = 42,
        bats_epsilon: float = 0.1,
        bats_k: float = 4.0,
        use_dynamic_bats_k: bool = True,
        bats_sampling_mode: str = "priority_capped",
        bats_token_budget: int = 1024,
        budget_num_cameras: int | None = None,
        current_visual_tokens: int = 64,
        history_visual_tokens: int = 4,
        tvi_tokens: int = 1,
        current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
        history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
        eval_image_fps: float = 1.0,
        include_state: bool = False,
        platform_text: str = "",
        source: str = "online_eval",
    ):
        self.required_cameras = tuple(str(camera) for camera in required_cameras)
        self.history_image_frames = None if history_image_frames is None else int(history_image_frames)
        self.state_dim = int(state_dim)
        self.action_horizon = int(action_horizon)
        self.history_selection = str(history_selection).strip().lower()
        if self.history_selection not in {"bats", "recent", "uniform"}:
            raise ValueError(
                "history_selection must be one of ['bats', 'recent', 'uniform'], "
                f"got {history_selection!r}"
            )
        self.history_update_mode = str(history_update_mode)
        self.eval_image_fps = float(eval_image_fps)
        self.include_state = bool(include_state)
        self.platform_text = str(platform_text)
        self.source = str(source)
        self.history_config = OnlineHistoryConfig(
            history_policy=self.history_selection,
            history_image_frames=self.history_image_frames,
            history_candidate_source_stride=history_candidate_source_stride,
            dataset_name=str(dataset_name),
            required_cameras=self.required_cameras,
            bats_seed=int(bats_seed),
            bats_epsilon=float(bats_epsilon),
            bats_k=float(bats_k),
            bats_use_dynamic_k=bool(use_dynamic_bats_k),
            bats_sampling_mode=str(bats_sampling_mode),
            bats_token_budget=int(bats_token_budget),
            budget_num_cameras=None if budget_num_cameras is None else int(budget_num_cameras),
            current_visual_tokens=int(current_visual_tokens),
            history_visual_tokens=int(history_visual_tokens),
            tvi_tokens=int(tvi_tokens),
            current_wrapper_tokens=int(current_wrapper_tokens),
            history_wrapper_tokens=int(history_wrapper_tokens),
        )
        self.action_stats: dict[str, Any] | None = None

    def set_action_stats(self, action_stats: dict[str, Any]) -> None:
        self.action_stats = action_stats

    def prepare_observation_for_step(self, *, observation: dict[str, Any], step: int) -> dict[str, Any]:
        prepared = dict(observation)
        metadata = dict(prepared.get("navvla_eval") or {})
        timestamp = float(step) / self.eval_image_fps if self.eval_image_fps > 0 else float(step)
        metadata.update({"frame_index": int(step), "timestamp": timestamp})
        prepared["navvla_eval"] = metadata
        return prepared

    def build_example(self, *, observation: dict[str, Any], history: EpisodeHistory, instruction: str) -> dict[str, Any]:
        online_observation = self._online_observation(observation, instruction=instruction)
        online_history = self._online_history(history)
        state = self._build_state(history=history, observation=online_observation) if self.include_state else None
        sample = build_online_navvla_history_sample(
            observation=online_observation,
            history=online_history,
            instruction=self._instruction_for_sample(observation=observation, history=history, instruction=instruction),
            state=state,
            config=self.history_config,
            platform_text=self.platform_text,
        )
        sample["action"] = np.zeros((self.action_horizon, 4), dtype=np.float32)
        sample["action_padding_mask"] = np.zeros((self.action_horizon,), dtype=bool)
        sample["distance_to_goal"] = float(online_observation.get("distance_to_goal", 0.0))
        metadata = dict(sample.get("metadata") or {})
        metadata.update({"source": self.source})
        sample["metadata"] = metadata
        return sample

    def history_visual_images(self, *, observation: dict[str, Any], instruction: str) -> dict[str, Any]:
        online_observation = self._online_observation(observation, instruction=instruction)
        images = online_observation.get("images")
        if not isinstance(images, dict):
            return {}
        return {
            camera_name: images[camera_name]
            for camera_name in self.required_cameras
            if images.get(camera_name) is not None
        }

    def update_history(
        self,
        *,
        history: EpisodeHistory,
        observation: dict[str, Any],
        prediction: ActionPrediction,
        instruction: str,
    ) -> EpisodeHistory:
        _commit_online_long_memory_update(history, prediction)
        online_observation = self._online_observation(observation, instruction=instruction)
        stored_observation = _compact_cached_observation(online_observation)
        existing_index = _matching_cached_history_frame_index(history, stored_observation)
        if existing_index is not None:
            history.observations[existing_index] = stored_observation
            return history
        if "image" in stored_observation:
            history.images.append(stored_observation["image"])
        history.observations.append(stored_observation)
        history.raw_actions.append(np.asarray(prediction.raw_actions, dtype=np.float32).reshape(-1, 4)[-1])
        history.instructions.append(instruction)
        return history

    def history_observations_for_update(
        self,
        *,
        pre_observation: dict[str, Any],
        post_observation: dict[str, Any],
        step_result: EnvironmentStepResult,
        action_observations: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del step_result
        if self.history_update_mode in {"action_observations", "waypoint", "waypoints"}:
            observations = list(action_observations or [])
            if _has_online_visual_token_cache(pre_observation):
                return [pre_observation, *observations]
            if observations:
                return observations
            return [post_observation]
        if self.history_update_mode == "post_action":
            return [post_observation]
        raise ValueError(f"Unsupported history_update_mode: {self.history_update_mode}")

    def _online_history(self, history: EpisodeHistory) -> EpisodeHistory:
        return history

    def _online_observation(self, observation: dict[str, Any], *, instruction: str) -> dict[str, Any]:
        del instruction
        prepared = dict(observation)
        if _has_online_visual_token_cache(prepared) and "image" not in prepared and "images" not in prepared:
            return prepared
        if "images" not in prepared:
            prepared["images"] = {"front": prepared["image"]}
        if "image" not in prepared:
            primary_camera = self.required_cameras[0]
            prepared["image"] = prepared["images"][primary_camera]
        return prepared

    def _instruction_for_sample(self, *, observation: dict[str, Any], history: EpisodeHistory, instruction: str) -> str:
        del observation, history
        return instruction

    def _build_state(self, *, history: EpisodeHistory, observation: dict[str, Any]) -> np.ndarray:
        del observation
        state_dim = self.state_dim if self.state_dim > 0 else max(1, int(self.history_image_frames or 1)) * 4
        if state_dim % 4 != 0:
            raise ValueError(f"state_dim must be a multiple of 4, got {state_dim}")
        history_steps = state_dim // 4
        raw_chunks = np.zeros((history_steps, 4), dtype=np.float32)
        padding_mask = np.ones((history_steps, 4), dtype=bool)
        actions = [
            np.asarray(action, dtype=np.float32).reshape(-1, 4)[-1]
            for action in history.raw_actions[-history_steps:]
        ]
        if actions:
            chunks = np.stack(actions, axis=0).astype(np.float32)
            raw_chunks[-chunks.shape[0] :] = chunks
            padding_mask[-chunks.shape[0] :] = False
        if self.action_stats is None:
            return raw_chunks.reshape(-1).astype(np.float32)
        state_stats = build_repeated_state_statistics(self.action_stats, history_steps)
        normalized = normalize_values(raw_chunks.reshape(-1), state_stats).astype(np.float32)
        normalized[padding_mask.reshape(-1)] = 0.0
        return normalized


class OpenFlyRuntimeDatasetAdapter(OnlineNavVLARuntimeDatasetAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            required_cameras=("front",),
            platform_text=OPENFLY_PLATFORM_TEXT,
            source="openfly_online_eval",
            **kwargs,
        )


class UAVFlowRuntimeDatasetAdapter(OnlineNavVLARuntimeDatasetAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            required_cameras=("front",),
            platform_text=UAVFLOW_PLATFORM_TEXT,
            source="uavflow_online_eval",
            **kwargs,
        )

    def prepare_observation_for_step(self, *, observation: dict[str, Any], step: int) -> dict[str, Any]:
        prepared = super().prepare_observation_for_step(observation=observation, step=step)
        uavflow_pose = dict(prepared.get("uavflow_pose") or {})
        metadata = dict(uavflow_pose.get("navvla_eval") or {})
        metadata.update(dict(prepared.get("navvla_eval") or {}))
        uavflow_pose["navvla_eval"] = metadata
        prepared["uavflow_pose"] = uavflow_pose
        return prepared


class TravelUAVRuntimeDatasetAdapter(OnlineNavVLARuntimeDatasetAdapter):
    def __init__(self, **kwargs: Any):
        super().__init__(
            required_cameras=("front", "left", "right", "rear"),
            platform_text=TRAVELUAV_NAVVLA_PLATFORM_TEXT,
            source="traveluav_online_eval",
            **kwargs,
        )

    def prepare_observation_for_step(self, *, observation: dict[str, Any], step: int) -> dict[str, Any]:
        if "traveluav_episode" not in observation:
            return super().prepare_observation_for_step(observation=observation, step=step)
        prepared = dict(observation)
        episode = dict(prepared["traveluav_episode"])
        metadata = dict(episode.get("navvla_eval") or {})
        timestamp = float(step) / self.eval_image_fps if self.eval_image_fps > 0 else float(step)
        metadata.update({"frame_index": int(step), "timestamp": timestamp})
        episode["navvla_eval"] = metadata
        prepared["traveluav_episode"] = episode
        return prepared

    def _online_history(self, history: EpisodeHistory) -> EpisodeHistory:
        observations = []
        for index, item in enumerate(history.observations):
            episode = item.get("traveluav_episode")
            cache_only = _has_online_visual_token_cache(item) and isinstance(episode, dict) and "rgb" not in episode
            if "images" in item or cache_only:
                observations.append(item)
                continue
            instruction = history.instructions[index] if index < len(history.instructions) else ""
            observations.append(self._online_observation(item, instruction=instruction))
        return EpisodeHistory(
            images=list(history.images),
            observations=observations,
            poses=list(history.poses),
            raw_actions=list(history.raw_actions),
            instructions=list(history.instructions),
            long_memory_tokens=history.long_memory_tokens,
            long_memory_tvi=history.long_memory_tvi,
            long_memory_blocks=list(history.long_memory_blocks),
            long_memory_frame_indices=set(history.long_memory_frame_indices),
        )

    def _online_observation(self, observation: dict[str, Any], *, instruction: str) -> dict[str, Any]:
        if "traveluav_episode" not in observation:
            return super()._online_observation(observation, instruction=instruction)
        episode = dict(observation["traveluav_episode"])
        if _has_online_visual_token_cache(observation) and "rgb" not in episode:
            return dict(observation)
        episode["instruction"] = instruction
        images = {
            camera_name: np.asarray(_traveluav_episode_camera_image(episode, camera_name), dtype=np.uint8)
            for camera_name in self.required_cameras
        }
        metadata = dict(episode.get("navvla_eval") or {})
        online = {
            "image": images[self.required_cameras[0]],
            "images": images,
            "navvla_eval": metadata,
            "traveluav_episode": episode,
            "distance_to_goal": _traveluav_distance_to_goal(observation),
        }
        if _has_online_visual_token_cache(observation):
            online["navvla_online_visual_tokens"] = dict(observation["navvla_online_visual_tokens"])
        return online

    def _instruction_for_sample(self, *, observation: dict[str, Any], history: EpisodeHistory, instruction: str) -> str:
        del history
        episode = observation.get("traveluav_episode")
        if not isinstance(episode, dict):
            return instruction
        current_episode = dict(episode)
        current_episode["instruction"] = instruction
        if not _traveluav_has_object_description(current_episode):
            stage = str(observation.get("stage") or "cruise").strip()
            return f"Stage: {stage}\n\nInstruction: {str(instruction).strip()}"
        return build_traveluav_stage_instruction(
            episodes=[current_episode],
            assist_notice=observation.get("stage"),
        )

    def _build_state(self, *, history: EpisodeHistory, observation: dict[str, Any]) -> np.ndarray:
        episode = observation.get("traveluav_episode")
        if not isinstance(episode, dict):
            return super()._build_state(history=history, observation=observation)
        episodes = [
            item["traveluav_episode"]
            for item in history.observations
            if isinstance(item, dict) and isinstance(item.get("traveluav_episode"), dict)
        ]
        episodes.append(episode)
        history_steps = self.state_dim // 4 if self.state_dim > 0 else max(1, int(self.history_image_frames or 1))
        return build_navvla_history_state(
            episodes=episodes,
            history_steps=history_steps,
            action_stats=self.action_stats,
        )


class AerialVLNRuntimeDatasetAdapter(OnlineNavVLARuntimeDatasetAdapter):
    def __init__(self, *, include_aerialvln_prompt_prefix: bool = True, **kwargs: Any):
        platform_text = AERIALVLN_PLATFORM_TEXT
        if not include_aerialvln_prompt_prefix:
            platform_text = platform_text.removeprefix(AERIALVLN_PROMPT_PREFIX)
        super().__init__(
            required_cameras=("front",),
            platform_text=platform_text,
            source="aerialvln_online_eval",
            **kwargs,
        )

    def build_example(self, *, observation: dict[str, Any], history: EpisodeHistory, instruction: str) -> dict[str, Any]:
        sample = super().build_example(observation=observation, history=history, instruction=instruction)
        metadata = _navvla_eval_metadata(observation)
        sample_metadata = dict(sample.get("metadata") or {})
        sample_metadata.update(
            {
                "episode_id": metadata.get("episode_id"),
                "episode_uid": metadata.get("episode_uid"),
                "scene_id": metadata.get("scene_id"),
            }
        )
        sample["metadata"] = sample_metadata
        return sample


def _traveluav_episode_camera_image(episode: dict[str, Any], camera_name: str) -> np.ndarray:
    rgb = episode.get("rgb")
    if rgb is None:
        raise ValueError("TravelUAV observation is missing rgb cameras")
    camera_index = TRAVELUAV_CAMERA_INDEX_BY_NAVVLA_NAME[camera_name]
    if camera_index >= len(rgb):
        raise ValueError(f"TravelUAV observation is missing {camera_name} rgb camera")
    return np.asarray(rgb[camera_index], dtype=np.uint8)


def _has_online_visual_token_cache(observation: dict[str, Any]) -> bool:
    cache = observation.get("navvla_online_visual_tokens")
    return isinstance(cache, dict) and bool(cache)


def _matching_cached_history_frame_index(
    history: EpisodeHistory,
    observation: dict[str, Any],
) -> int | None:
    if not _has_online_visual_token_cache(observation):
        return None
    frame_index = _navvla_eval_metadata(observation).get("frame_index")
    if frame_index is None:
        return None
    for history_index in range(len(history.observations) - 1, -1, -1):
        existing_frame_index = _navvla_eval_metadata(history.observations[history_index]).get("frame_index")
        if existing_frame_index is not None and int(existing_frame_index) == int(frame_index):
            return history_index
    return None


def _compact_cached_observation(observation: dict[str, Any]) -> dict[str, Any]:
    if not _has_online_visual_token_cache(observation):
        return observation

    compact = {
        key: value
        for key, value in observation.items()
        if key not in {"image", "images"}
    }
    episode = compact.get("traveluav_episode")
    if isinstance(episode, dict):
        compact_episode = dict(episode)
        compact_episode.pop("rgb", None)
        compact["traveluav_episode"] = compact_episode
    return compact


def _commit_online_long_memory_update(history: EpisodeHistory, prediction: ActionPrediction) -> None:
    updates = prediction.metadata.get("online_long_memory_updates")
    if not isinstance(updates, list) or not updates:
        return
    update = updates[-1]
    if not isinstance(update, dict):
        raise ValueError("online_long_memory_updates entries must be dictionaries")
    tokens = np.asarray(update.get("tokens"))
    tvi = np.asarray(update.get("tvi"), dtype=np.float32)
    blocks = update.get("blocks")
    if tokens.ndim != 3 or tvi.ndim != 2 or tvi.shape[-1] != 2 or not isinstance(blocks, list):
        raise ValueError("invalid online long-memory update payload")
    if tokens.shape[0] != tvi.shape[0] or tokens.shape[0] != len(blocks):
        raise ValueError("online long-memory update tokens, TVI, and blocks must have equal lengths")
    history.long_memory_tokens = tokens
    history.long_memory_tvi = tvi
    history.long_memory_blocks = [dict(block) for block in blocks]
    history.long_memory_frame_indices.add(int(update["frame_index"]))


def _traveluav_distance_to_goal(observation: dict[str, Any]) -> float:
    episode = observation.get("traveluav_episode")
    target_position = observation.get("target_position")
    if not isinstance(episode, dict) or target_position is None:
        return 0.0
    state = episode.get("sensors", {}).get("state", {}) if isinstance(episode.get("sensors"), dict) else {}
    position = state.get("position") if isinstance(state, dict) else None
    if position is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(position, dtype=np.float32).reshape(3) - np.asarray(target_position, dtype=np.float32).reshape(3)))


def _traveluav_has_object_description(episode: dict[str, Any]) -> bool:
    for key in ("object_description", "object_desc"):
        for container in (episode, episode.get("benchmark_metadata"), episode.get("source_metadata")):
            if isinstance(container, dict):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return True
                if isinstance(value, list) and any(str(item).strip() for item in value):
                    return True
    return False


def _navvla_eval_metadata(observation: dict[str, Any]) -> dict[str, Any]:
    metadata = observation.get("navvla_eval")
    return metadata if isinstance(metadata, dict) else {}


def _cfg_get(cfg: dict[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _online_adapter_kwargs(
    cfg: dict[str, Any] | Any,
    *,
    dataset_name: str,
    eval_image_fps: float = 1.0,
    history_update_mode: str = "action_observations",
    include_state: bool = False,
) -> dict[str, Any]:
    return {
        "history_image_frames": _cfg_get(cfg, "history_image_frames", None),
        "history_candidate_source_stride": _cfg_get(cfg, "history_candidate_source_stride", None),
        "state_dim": _cfg_get(cfg, "state_dim", 0),
        "action_horizon": _cfg_get(cfg, "action_horizon", 8),
        "eval_image_fps": _cfg_get(cfg, "eval_image_fps", eval_image_fps),
        "include_state": _cfg_get(cfg, "include_state", include_state),
        "history_selection": _cfg_get(cfg, "history_selection", _cfg_get(cfg, "history_policy", "bats")),
        "history_update_mode": _cfg_get(cfg, "history_update_mode", history_update_mode),
        "dataset_name": _cfg_get(cfg, "dataset_name", dataset_name),
        "bats_seed": _cfg_get(cfg, "bats_seed", 42),
        "bats_epsilon": _cfg_get(cfg, "bats_epsilon", 0.1),
        "bats_k": _cfg_get(cfg, "bats_k", 4.0),
        "use_dynamic_bats_k": _cfg_get(cfg, "use_dynamic_bats_k", _cfg_get(cfg, "bats_use_dynamic_k", True)),
        "bats_sampling_mode": _cfg_get(cfg, "bats_sampling_mode", "priority_capped"),
        "bats_token_budget": _cfg_get(cfg, "bats_token_budget", 1024),
        "budget_num_cameras": _cfg_get(cfg, "budget_num_cameras", None),
        "current_visual_tokens": _cfg_get(cfg, "current_visual_tokens", 64),
        "history_visual_tokens": _cfg_get(cfg, "history_visual_tokens", 4),
        "tvi_tokens": _cfg_get(cfg, "tvi_tokens", 1),
        "current_wrapper_tokens": _cfg_get(
            cfg, "current_wrapper_tokens", MINICPM_IMAGE_WRAPPER_TOKENS
        ),
        "history_wrapper_tokens": _cfg_get(
            cfg, "history_wrapper_tokens", MINICPM_IMAGE_WRAPPER_TOKENS
        ),
    }


def get_runtime_dataset_adapter(cfg: dict[str, Any] | Any) -> RuntimeDatasetAdapter:
    runtime_adapter = _cfg_get(cfg, "runtime_adapter", None)
    dataset_py = _cfg_get(cfg, "dataset_py", None)
    adapter_name = str(runtime_adapter or dataset_py or "").strip()
    if adapter_name in {"openfly", "airsim_openfly_datasets"}:
        return OpenFlyRuntimeDatasetAdapter(
            **_online_adapter_kwargs(cfg, dataset_name="openfly", eval_image_fps=1.0),
        )
    if adapter_name in {"uavflow", "navvla_uavflow_online"}:
        return UAVFlowRuntimeDatasetAdapter(
            **_online_adapter_kwargs(cfg, dataset_name="uavflow", eval_image_fps=1.0),
        )
    if adapter_name in {"traveluav", "airsim_datasets", "navvla_traveluav_online"}:
        return TravelUAVRuntimeDatasetAdapter(
            **_online_adapter_kwargs(
                cfg,
                dataset_name="vln_train",
                eval_image_fps=0.2,
                history_update_mode="action_observations",
                include_state=False,
            ),
        )
    if adapter_name in {"vlnce", "vlnce_r2r", "vlnce_rxr"}:
        state_dim = int(_cfg_get(cfg, "state_dim", 0))
        return OnlineNavVLARuntimeDatasetAdapter(
            required_cameras=tuple(_cfg_get(cfg, "required_cameras", ("front",))),
            platform_text=VLNCE_PLATFORM_TEXT,
            source=f"{_cfg_get(cfg, 'dataset_name', adapter_name)}_online_eval",
            **_online_adapter_kwargs(
                cfg,
                dataset_name=str(_cfg_get(cfg, "dataset_name", adapter_name)),
                eval_image_fps=float(_cfg_get(cfg, "fps", 1.0)),
                history_update_mode="action_observations",
                include_state=state_dim > 0,
            ),
        )
    if adapter_name in {"aerialvln", "navvla_aerialvln_online"}:
        return AerialVLNRuntimeDatasetAdapter(
            include_aerialvln_prompt_prefix=_cfg_get(cfg, "include_aerialvln_prompt_prefix", True),
            history_image_frames=_cfg_get(cfg, "history_image_frames", 0),
            history_candidate_source_stride=_cfg_get(cfg, "history_candidate_source_stride", None),
            state_dim=_cfg_get(cfg, "state_dim", 0),
            action_horizon=_cfg_get(cfg, "action_horizon", 8),
            eval_image_fps=_cfg_get(cfg, "eval_image_fps", 1.0),
            include_state=_cfg_get(cfg, "include_state", False),
            history_selection=_cfg_get(cfg, "history_selection", "recent"),
            history_update_mode=_cfg_get(cfg, "history_update_mode", "post_action"),
            dataset_name=_cfg_get(cfg, "dataset_name", "aerialvln"),
            bats_seed=_cfg_get(cfg, "bats_seed", 42),
            bats_epsilon=_cfg_get(cfg, "bats_epsilon", 0.1),
            bats_k=_cfg_get(cfg, "bats_k", 4.0),
            use_dynamic_bats_k=_cfg_get(cfg, "use_dynamic_bats_k", _cfg_get(cfg, "bats_use_dynamic_k", True)),
            bats_sampling_mode=_cfg_get(cfg, "bats_sampling_mode", "priority_capped"),
            bats_token_budget=_cfg_get(cfg, "bats_token_budget", 1024),
            budget_num_cameras=_cfg_get(cfg, "budget_num_cameras", None),
            current_visual_tokens=_cfg_get(cfg, "current_visual_tokens", 64),
            history_visual_tokens=_cfg_get(cfg, "history_visual_tokens", 4),
            tvi_tokens=_cfg_get(cfg, "tvi_tokens", 1),
            current_wrapper_tokens=_cfg_get(
                cfg, "current_wrapper_tokens", MINICPM_IMAGE_WRAPPER_TOKENS
            ),
            history_wrapper_tokens=_cfg_get(
                cfg, "history_wrapper_tokens", MINICPM_IMAGE_WRAPPER_TOKENS
            ),
        )
    raise ValueError(f"unknown runtime dataset adapter for NavVLA eval: {adapter_name}")
