from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from navvla_conversion.context_index import ContextIndexConfig
from navvla_conversion.lerobot_v3_writer import write_navvla_lerobot_dataset
from navvla_conversion.schema import (
    NavVLACameraSpec,
    NavVLADatasetSpec,
    NavVLAEpisode,
    NavVLAFrame,
    NavVLATaskSpec,
)


def tiny_episodes(image_dir: Path) -> list[NavVLAEpisode]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for frame_index in range(3):
        path = image_dir / f"frame_{frame_index:06d}.png"
        Image.fromarray(np.full((16, 16, 3), frame_index * 40, dtype=np.uint8)).save(path)
        image_paths.append(path)
    camera = NavVLACameraSpec(
        name="front",
        video_key="front_image",
        viewpoint_type="front",
        azimuth_rad=0.0,
    )
    task = NavVLATaskSpec(
        task_index=0,
        instruction="go forward",
        task_type="navigation",
        task_subtype="vln",
        platform_text="uav",
        dataset_source="pytest",
        scene_id="scene-a",
    )
    frames = [
        NavVLAFrame(
            frame_index=index,
            timestamp=float(index),
            media_paths={"front_image": path},
            state=[float(index), 0.0, 0.0, 0.0],
            action=[[0.1, 0.0, 0.0, 0.0]],
            source_frame_index=index,
            source_metadata={"source_pose": [float(index), 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
        for index, path in enumerate(image_paths)
    ]
    return [
        NavVLAEpisode(
            episode_id="episode-a",
            task=task,
            frames=frames,
            cameras=[camera],
            split="vln_train",
            trajectory_id="traj-a",
        )
    ]


def tiny_spec(dataset_name: str = "vln_train") -> NavVLADatasetSpec:
    return NavVLADatasetSpec(
        dataset_name=dataset_name,
        fps=1.0,
        control_frequency_hz=1.0,
        context_policy_version="bats-v1",
        cache_policy_version="standalone-no-visual-cache",
        action_horizon=2,
        action_dim=4,
        state_dim=4,
        split="vln_train",
        episodes_per_file=10,
        files_per_chunk=10,
    )


@pytest.fixture
def tiny_dataset_root(tmp_path: Path) -> Path:
    summary = write_navvla_lerobot_dataset(
        tiny_episodes(tmp_path / "images"),
        output_root=tmp_path / "out",
        spec=tiny_spec(),
        write_workers=1,
        context_index_config=ContextIndexConfig(use_dynamic_bats_k=False, k=0.0),
    )
    return Path(summary["dataset_root"])
