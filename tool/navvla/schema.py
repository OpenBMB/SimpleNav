from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NavVLACameraSpec:
    name: str
    video_key: str
    viewpoint_type: str
    azimuth_rad: float
    intrinsics: list[list[float]] | None = None
    extrinsics_body: list[list[float]] | None = None
    calibration_status: str = "unknown"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera name must be non-empty")
        if not self.video_key:
            raise ValueError("camera video_key must be non-empty")


@dataclass(frozen=True)
class NavVLATaskSpec:
    task_index: int
    instruction: str
    task_type: str
    task_subtype: str
    platform_text: str
    dataset_source: str
    scene_id: str
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_index < 0:
            raise ValueError(f"task_index must be non-negative, got {self.task_index}")
        if not self.instruction:
            raise ValueError("instruction must be non-empty")
        if not self.task_type:
            raise ValueError("task_type must be non-empty")
        if not self.dataset_source:
            raise ValueError("dataset_source must be non-empty")
        if not self.scene_id:
            raise ValueError("scene_id must be non-empty")


@dataclass(frozen=True)
class NavVLAFrame:
    frame_index: int
    timestamp: float | None
    media_paths: Mapping[str, str | Path]
    state: Sequence[float]
    data_index: int | None = None
    action: Sequence[Sequence[float]] | None = None
    action_available: bool = True
    source_frame_index: int | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)
    task: NavVLATaskSpec | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        if self.data_index is not None and self.data_index < 0:
            raise ValueError(f"data_index must be non-negative, got {self.data_index}")
        if not self.media_paths:
            raise ValueError("media_paths must be non-empty")
        if not self.state:
            raise ValueError("state must be non-empty")
        if len(self.state) != 4:
            raise ValueError(f"NavVLAFrame state must have state_dim=4, got {len(self.state)}")


@dataclass(frozen=True)
class NavVLAEpisode:
    episode_id: str
    task: NavVLATaskSpec
    frames: list[NavVLAFrame]
    cameras: list[NavVLACameraSpec]
    split: str = "train"
    trajectory_id: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.frames:
            raise ValueError("frames must be non-empty")
        if not self.cameras:
            raise ValueError("cameras must be non-empty")
        camera_keys = {camera.video_key for camera in self.cameras}
        for frame in self.frames:
            unknown = set(frame.media_paths) - camera_keys
            if unknown:
                raise ValueError(f"frame {frame.frame_index} has media keys not declared by cameras: {sorted(unknown)}")


@dataclass(frozen=True)
class NavVLADatasetSpec:
    dataset_name: str
    fps: float
    context_policy_version: str
    cache_policy_version: str
    action_horizon: int = 8
    action_dim: int = 4
    state_dim: int = 4
    control_frequency_hz: float | None = None
    split: str = "train"
    episodes_per_file: int = 20
    files_per_chunk: int = 50
    state_mode: str = "source_world_absolute_pose_xyz_yaw"

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.control_frequency_hz is not None and self.control_frequency_hz <= 0:
            raise ValueError(f"control_frequency_hz must be positive, got {self.control_frequency_hz}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.state_dim != 4:
            raise ValueError(f"NavVLA state contract requires state_dim=4, got {self.state_dim}")
        if not self.state_mode:
            raise ValueError("state_mode must be non-empty")
        if self.episodes_per_file <= 0:
            raise ValueError(f"episodes_per_file must be positive, got {self.episodes_per_file}")
        if self.files_per_chunk <= 0:
            raise ValueError(f"files_per_chunk must be positive, got {self.files_per_chunk}")
