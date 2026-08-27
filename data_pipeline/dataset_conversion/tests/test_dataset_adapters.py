from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from navvla_conversion.adapters.aerialvln import AerialVLNAdapter
from navvla_conversion.adapters.cosfly import CosFlyAdapter
from navvla_conversion.adapters.traveluav import load_episode, state_4d
from navvla_conversion.adapters.vlnce_rendered import load_vlnce_rendered_episodes


def _write_rgb(path: Path, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5, 3), value, dtype=np.uint8)).save(path)


def _write_cosfly_trace(parent: Path, trace_type: str) -> None:
    trace_root = parent / trace_type
    poses = [(0.0, 0.0, 10.0, 0.0), (1.0, 0.0, 9.5, 0.0), (1.0, 1.0, 9.0, 90.0)]
    points = []
    for frame_index, (x, y, z, yaw) in enumerate(poses):
        frame_root = trace_root / "frames_playback" / f"frame_{frame_index:05d}"
        _write_rgb(frame_root / "rgb.png", frame_index)
        next_pose = poses[frame_index + 1] if frame_index + 1 < len(poses) else None
        points.append(
            {
                "index": frame_index,
                "timing": {"trace_timestamp": frame_index * 0.5, "sim_timestamp": 100.0 + frame_index},
                "drone_pose": {"x": x, "y": y, "z": z, "pitch": -45.0, "yaw": yaw, "roll": 0.0},
                "nav_waypoint": {
                    "t1_world": (
                        None
                        if next_pose is None
                        else {"x": next_pose[0], "y": next_pose[1], "z": next_pose[2]}
                    )
                },
                "target": {
                    "enabled": True,
                    "visible": True,
                    "in_view": True,
                    "depth": 25.0,
                    "world_location": {"x": 0.0, "y": 0.0, "z": 1.0},
                    "image_uv": [2.5, 2.0],
                    "bbox_2d": {"xmin": 1.0, "ymin": 1.0, "xmax": 4.0, "ymax": 3.0},
                },
            }
        )
    payload = {
        "schema": "drone_nav_traj_v7",
        "schema_version": "7.0.0",
        "dataset_format": "v7",
        "source": {
            "path_index": 7,
            "scenario_id": 7,
            "scenario_name": "drone_trace_oneshot_path_7",
            "algorithm": "oneshot",
        },
        "camera": {
            "width": 5,
            "height": 4,
            "focal_length_px": 4.0,
            "intrinsic": [[4.0, 0.0, 2.5], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
        },
        "trace_dir": trace_type,
        "frames_dir": "frames_playback",
        "augmentation": {
            "enabled": trace_type != "ORI",
            "aug_index": 0 if trace_type == "ORI" else 1,
            "is_original": trace_type == "ORI",
        },
        "points": points,
    }
    (trace_root / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")


def test_traveluav_identity_and_state_contract(tmp_path: Path) -> None:
    payload = {
        "episode_id": "/dataset/City/episode-001",
        "instruction": "fly to target",
        "frames": [
            {
                "frame_index": 0,
                "state": [1.0, 2.0, 3.0, 0.25],
                "media_paths": {"front_image": "dataset/City/episode-001/front.png"},
            }
        ],
    }
    path = tmp_path / "episode.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    episode = load_episode(path, source_root=tmp_path, split="train", task_index=0)
    assert episode.episode_id == "episode-001"
    assert episode.task.scene_id == "City"
    assert episode.frames[0].state == [1.0, 2.0, 3.0, 0.25]
    with pytest.raises(ValueError, match="at least"):
        state_4d({"state": [1.0, 2.0, 3.0]})


def test_aerialvln_maps_world_pose_and_anchor_actions(tmp_path: Path) -> None:
    annotation_root = tmp_path / "aerialvln"
    annotation_root.mkdir()
    poses = [
        [10.0, 20.0, 3.0, 0.0, 0.0, math.pi / 2],
        [10.0, 22.0, 1.0, 0.0, 0.0, math.pi / 2],
    ]
    (annotation_root / "train.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "episode-a",
                        "trajectory_id": "traj-a",
                        "scene_id": 5,
                        "instruction": {"instruction_text": "fly forward"},
                        "reference_path": poses,
                        "actions": [4, 0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for frame_index in range(2):
        _write_rgb(tmp_path / "media" / "vln_train" / "traj-a" / f"{frame_index:06d}.png")
    episode = AerialVLNAdapter(
        media_cache_root=tmp_path / "media",
        reuse_media_cache=True,
    ).load_episodes(tmp_path, split="train")[0]
    assert episode.task.scene_id == "5"
    assert episode.frames[0].state == [10.0, 20.0, 3.0, math.pi / 2]
    assert episode.frames[0].source_metadata["source_pose"] == poses[0]
    assert episode.frames[-1].action_available is False


def test_vlnce_manifest_preserves_identity_and_coordinates(tmp_path: Path) -> None:
    rows = []
    for frame_index, position in enumerate(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])):
        path = tmp_path / "rgb" / f"{frame_index:06d}.png"
        _write_rgb(path, frame_index)
        rows.append(
            {
                "dataset_family": "r2r",
                "split": "val_seen",
                "episode_id": "10",
                "trajectory_id": "20",
                "scene_id": "data/scene_datasets/mp3d/scan/scan.glb",
                "instruction_text": "Walk forward.",
                "frame_index": frame_index,
                "rgb_path": str(path),
                "position": position,
                "yaw": 0.0,
                "rotation_source": "test",
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    episode = load_vlnce_rendered_episodes(tmp_path)[0]
    assert episode.episode_id == "r2r_val_seen_10"
    assert episode.trajectory_id == "20"
    assert episode.task.scene_id == "data/scene_datasets/mp3d/scan/scan.glb"
    assert episode.frames[1].state == [1.0, 0.0, 0.0, 0.0]


def test_cosfly_preserves_pairs_and_rejects_odd_selection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    parent = source / "Town01" / "trajectory_123"
    for trace_name in ("ORI", "aug_001"):
        _write_cosfly_trace(parent, trace_name)
    episodes = CosFlyAdapter(
        media_cache_root=tmp_path / "cache",
        fps=2.0,
    ).load_episodes(source, split="train", max_episodes=2)
    assert [episode.episode_id for episode in episodes] == [
        "Town01__trajectory_123__ORI",
        "Town01__trajectory_123__aug_001",
    ]
    assert all(episode.task.metadata["parent_trajectory_id"] == "Town01/trajectory_123" for episode in episodes)
    with pytest.raises(ValueError, match="even"):
        CosFlyAdapter(media_cache_root=tmp_path / "cache").load_episodes(
            source,
            split="train",
            max_episodes=1,
        )
