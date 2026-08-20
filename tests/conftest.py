from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tool.navvla.context_index import ContextIndexConfig, iter_context_refs
from tool.navvla.lerobot_v3_writer import write_navvla_lerobot_dataset
from tool.navvla.schema import NavVLACameraSpec, NavVLADatasetSpec, NavVLAEpisode, NavVLAFrame, NavVLATaskSpec
from tool.navvla.visual_token_cache import (
    VisualTokenProfile,
    write_profile_index,
    write_profile_manifest,
    write_profile_token_record,
)


def tiny_navvla_episodes(image_dir: Path) -> list[NavVLAEpisode]:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for frame_index in range(3):
        path = image_dir / f"frame_{frame_index:06d}.png"
        Image.fromarray(np.full((8, 8, 3), frame_index * 40, dtype=np.uint8)).save(path)
        image_paths.append(path)

    camera = NavVLACameraSpec(name="front", video_key="front_image", viewpoint_type="front", azimuth_rad=0.0)
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
            frame_index=idx,
            timestamp=float(idx),
            media_paths={"front_image": path},
            state=[float(idx), 0.0, 0.0, 0.0],
            action=[[0.1, 0.0, 0.0, 0.0]],
            source_frame_index=idx,
        )
        for idx, path in enumerate(image_paths)
    ]
    return [NavVLAEpisode(episode_id="episode-a", task=task, frames=frames, cameras=[camera], split="vln_train", trajectory_id="traj-a")]


def tiny_navvla_spec(dataset_name: str = "traveluav") -> NavVLADatasetSpec:
    return NavVLADatasetSpec(
        dataset_name=dataset_name,
        fps=1.0,
        control_frequency_hz=1.0,
        context_policy_version="bats-v1",
        cache_policy_version="profile-cache-v1",
        action_horizon=2,
        action_dim=4,
        state_dim=4,
        split="vln_train",
        episodes_per_file=10,
        files_per_chunk=10,
    )


@pytest.fixture
def tiny_navvla_dataset_root(tmp_path: Path) -> Path:
    output_root = tmp_path / "out"
    summary = write_navvla_lerobot_dataset(
        tiny_navvla_episodes(tmp_path / "images"),
        output_root=output_root,
        spec=tiny_navvla_spec(),
        overwrite=True,
        cache_workers=1,
        context_index_config=ContextIndexConfig(
            use_dynamic_bats_k=False,
            k=0.0,
        ),
    )
    return Path(summary["dataset_root"])


@pytest.fixture
def profile_cache_dataset_root(tiny_navvla_dataset_root: Path) -> Path:
    profile = VisualTokenProfile(
        name="qwen3_vl_4b_pooled_history",
        visual_head="qwen3_vl_visual",
        encoder_name="Qwen3-VL-4B-Instruct",
        encoder_ckpt="/tmp/qwen3",
        token_level="pooled_history",
        token_count=4,
        hidden_dim=8,
        dtype="float16",
        has_deepstack=True,
        deepstack_layers=3,
    )
    write_profile_manifest(tiny_navvla_dataset_root, profile)
    refs = list(iter_context_refs(tiny_navvla_dataset_root, token_budget=1024))
    rows = []
    for ref in sorted(set(refs)):
        episode_id, frame_index, camera_name = ref.split("/", 2)
        record = write_profile_token_record(
            tiny_navvla_dataset_root,
            profile=profile,
            ref=ref,
            image_embeds=np.zeros((4, 8), dtype=np.float16),
            deepstack_embeds=np.zeros((3, 4, 8), dtype=np.float16),
        )
        rows.append(
            {
                "ref": record.ref,
                "path": record.path,
                "episode_id": episode_id,
                "trajectory_id": "traj-a",
                "frame_index": int(frame_index),
                "source_frame_index": int(frame_index),
                "data_index": int(frame_index),
                "camera_name": camera_name,
                "video_key": "front_image",
            }
        )
    write_profile_index(tiny_navvla_dataset_root, profile.name, rows)
    return tiny_navvla_dataset_root
