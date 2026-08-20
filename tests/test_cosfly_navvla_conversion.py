from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_trace(
    parent_root: Path,
    trace_type: str,
    *,
    poses: list[tuple[float, float, float, float]],
    missing_rgb_index: int | None = None,
    scenario_id: int = 7,
) -> None:
    trace_root = parent_root / trace_type
    points = []
    for frame_index, (x, y, z_up, yaw_deg) in enumerate(poses):
        frame_root = trace_root / "frames_playback" / f"frame_{frame_index:05d}"
        frame_root.mkdir(parents=True, exist_ok=True)
        if frame_index != missing_rgb_index:
            Image.new("RGBA", (16, 9), color=(frame_index + 1, 20, 30, 255)).save(frame_root / "rgb.png")
        next_pose = poses[frame_index + 1] if frame_index + 1 < len(poses) else None
        points.append(
            {
                "index": frame_index,
                "is_perturbed": trace_type != "ORI",
                "perturbation": {
                    "drone_position": trace_type != "ORI",
                    "drone_rotation": trace_type != "ORI",
                },
                "timing": {
                    "trace_timestamp": frame_index * 0.5,
                    "sim_timestamp": 100.0 + frame_index,
                },
                "drone_pose": {
                    "x": x,
                    "y": y,
                    "z": z_up,
                    "pitch": -45.0,
                    "yaw": yaw_deg,
                    "roll": 0.0,
                },
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
                    "image_uv": [8.0, 4.5],
                    "bbox_2d": {"xmin": 4.0, "ymin": 2.0, "xmax": 12.0, "ymax": 7.0},
                },
            }
        )
    payload = {
        "schema": "drone_nav_traj_v7",
        "schema_version": "7.0.0",
        "dataset_format": "v7",
        "source": {
            "path_index": scenario_id,
            "scenario_id": scenario_id,
            "scenario_name": f"drone_trace_oneshot_path_{scenario_id}",
            "algorithm": "oneshot",
        },
        "camera": {
            "width": 16,
            "height": 9,
            "focal_length_px": 8.0,
            "intrinsic": [[8.0, 0.0, 8.0], [0.0, 8.0, 4.5], [0.0, 0.0, 1.0]],
        },
        "trace_dir": trace_type,
        "frames_dir": "frames_playback",
        "augmentation": {
            "enabled": True,
            "aug_index": 0 if trace_type == "ORI" else 1,
            "is_original": trace_type == "ORI",
        },
        "points": points,
    }
    (trace_root / "trajectory.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_parent(
    root: Path,
    *,
    town: str = "Town01",
    trajectory_name: str = "trajectory_123",
    missing_rgb_index: int | None = None,
    scenario_id: int = 7,
    frame_count: int = 3,
) -> Path:
    parent = root / town / trajectory_name
    base_ori_poses = [
        (0.0, 0.0, 10.0, 0.0),
        (1.0, 0.0, 9.5, 0.0),
        (1.0, 1.0, 9.0, 90.0),
    ]
    base_aug_poses = [
        (0.1, 0.0, 10.0, 5.0),
        (1.1, 0.0, 9.5, 5.0),
        (1.0, 1.1, 9.0, 95.0),
    ]
    if frame_count == 3:
        ori_poses = base_ori_poses
        aug_poses = base_aug_poses
    else:
        ori_poses = [(float(index), 0.0, 10.0, 0.0) for index in range(frame_count)]
        aug_poses = [(float(index) + 0.1, 0.0, 10.0, 5.0) for index in range(frame_count)]
    _write_trace(parent, "ORI", poses=ori_poses, scenario_id=scenario_id)
    _write_trace(
        parent,
        "aug_001",
        poses=aug_poses,
        missing_rgb_index=missing_rgb_index,
        scenario_id=scenario_id,
    )
    return parent


def test_cosfly_split_manifest_groups_pairs_and_builds_exact_paired_anchors(tmp_path: Path) -> None:
    from tool.navvla.adapters._cosfly_splits import build_cosfly_split_manifest

    source_root = tmp_path / "source"
    _write_parent(
        source_root,
        town="Town01",
        trajectory_name="trajectory_seen_base",
        scenario_id=1,
        frame_count=16,
    )
    _write_parent(
        source_root,
        town="Town01_Opt",
        trajectory_name="trajectory_seen_opt",
        scenario_id=1,
        frame_count=16,
    )
    _write_parent(
        source_root,
        town="Town01",
        trajectory_name="trajectory_train",
        scenario_id=2,
        frame_count=16,
    )
    _write_parent(
        source_root,
        town="Town10HD",
        trajectory_name="trajectory_unseen",
        scenario_id=3,
        frame_count=16,
    )
    _write_parent(
        source_root,
        town="Town02",
        trajectory_name="trajectory_incomplete",
        scenario_id=4,
        frame_count=16,
        missing_rgb_index=8,
    )

    manifest = build_cosfly_split_manifest(
        source_root,
        output_path=tmp_path / "manifest.json",
        seen_parent_quotas={"Town01": 2},
        seen_anchor_count_per_trace=2,
        unseen_anchor_count_per_trace=1,
    )

    parent_splits = {row["parent_id"]: row["split"] for row in manifest["parents"]}
    assert parent_splits == {
        "Town01/trajectory_seen_base": "seen",
        "Town01/trajectory_train": "train",
        "Town01_Opt/trajectory_seen_opt": "seen",
        "Town10HD/trajectory_unseen": "unseen",
    }
    assert manifest["excluded_parents"][0]["parent_id"] == "Town02/trajectory_incomplete"
    anchors = manifest["validation_anchors"]
    assert len(anchors) == 6
    assert manifest["summary"]["validation_anchor_counts"] == {
        "ORI": 3,
        "aug_001": 3,
        "seen": 4,
        "unseen": 2,
        "total": 6,
    }
    paired = {}
    for row in anchors:
        paired.setdefault((row["split"], row["parent_id"], row["frame_index"]), set()).add(row["trace_type"])
    assert paired
    assert all(trace_types == {"ORI", "aug_001"} for trace_types in paired.values())


def test_cosfly_adapter_loads_paired_ori_and_aug_with_trace_suffixes(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root)
    episodes = CosFlyAdapter(fps=2.0, action_horizon=8, media_cache_root=tmp_path / "cache").load_episodes(
        source_root,
        split="train",
        max_episodes=2,
    )

    assert [episode.episode_id for episode in episodes] == [
        "Town01__trajectory_123__ORI",
        "Town01__trajectory_123__aug_001",
    ]
    for episode in episodes:
        assert episode.trajectory_id == episode.episode_id
        assert episode.task.instruction == "Track the target pedestrian."
        assert episode.task.task_type == "tracking"
        assert episode.task.task_subtype == "urban_uav_tracking"
        assert episode.task.dataset_source == "cosfly"
        assert episode.split == "vln_train"
        assert episode.task.metadata["parent_trajectory_id"] == "Town01/trajectory_123"
        assert episode.frames[0].source_metadata["source_split"] == "train"
        assert episode.frames[0].source_metadata["target_split"] == "vln_train"
        assert "paired_episode_id" not in episode.task.metadata
        assert "trace_type" not in episode.frames[0].source_metadata
        assert "paired_episode_id" not in episode.frames[0].source_metadata


def test_cosfly_adapter_filters_whole_pairs_from_split_manifest(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root, town="Town01", trajectory_name="trajectory_train")
    _write_parent(source_root, town="Town02_Opt", trajectory_name="trajectory_seen")
    _write_parent(source_root, town="Town10HD", trajectory_name="trajectory_unseen")
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cosfly_navvla_split_manifest_v1",
                "parents": [
                    {"parent_id": "Town01/trajectory_train", "split": "train"},
                    {"parent_id": "Town02_Opt/trajectory_seen", "split": "seen"},
                    {"parent_id": "Town10HD/trajectory_unseen", "split": "unseen"},
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = CosFlyAdapter(
        fps=2.0,
        action_horizon=8,
        media_cache_root=tmp_path / "cache",
        split_manifest_path=manifest_path,
    )

    assert [episode.episode_id for episode in adapter.load_episodes(source_root, split="vln_val_seen")] == [
        "Town02_Opt__trajectory_seen__ORI",
        "Town02_Opt__trajectory_seen__aug_001",
    ]
    assert [episode.episode_id for episode in adapter.load_episodes(source_root, split="unseen")] == [
        "Town10HD__trajectory_unseen__ORI",
        "Town10HD__trajectory_unseen__aug_001",
    ]


def test_cosfly_adapter_rejects_odd_max_episodes_to_preserve_pairs(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root)
    with pytest.raises(ValueError, match="even"):
        CosFlyAdapter(media_cache_root=tmp_path / "cache").load_episodes(
            source_root,
            split="train",
            max_episodes=1,
        )


def test_cosfly_adapter_letterboxes_media_and_transforms_image_geometry(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root)
    cache_root = tmp_path / "cache"
    episode = CosFlyAdapter(
        fps=2.0,
        action_horizon=8,
        media_cache_root=cache_root,
    ).load_episodes(source_root, split="train", max_episodes=2)[1]

    frame = episode.frames[0]
    resized_path = Path(frame.media_paths["front_image"])
    assert resized_path.is_relative_to(cache_root)
    assert not resized_path.is_relative_to(source_root)
    with Image.open(resized_path) as image:
        assert image.mode == "RGB"
        assert image.size == (384, 256)
        assert image.getpixel((0, 0)) == (0, 0, 0)
        assert image.getpixel((192, 128)) == (1, 20, 30)

    assert np.allclose(
        episode.cameras[0].intrinsics,
        [[192.0, 0.0, 192.0], [0.0, 192.0, 128.0], [0.0, 0.0, 1.0]],
        atol=1e-7,
    )
    assert frame.source_metadata["image_resize"] == {
        "policy": "letterbox",
        "source_size": [16, 9],
        "resized_content_size": [384, 216],
        "target_size": [384, 256],
        "offset_xy": [0, 20],
        "scale_xy": [24.0, 24.0],
    }
    assert frame.source_metadata["source_target_image_uv"] == [8.0, 4.5]
    assert frame.source_metadata["target_image_uv"] == [192.0, 128.0]
    assert frame.source_metadata["source_target_bbox_2d"] == {
        "xmin": 4.0,
        "ymin": 2.0,
        "xmax": 12.0,
        "ymax": 7.0,
    }
    assert frame.source_metadata["target_bbox_2d"] == {
        "xmin": 96.0,
        "ymin": 68.0,
        "xmax": 288.0,
        "ymax": 188.0,
    }


def test_cosfly_adapter_keeps_aug_pose_and_actions_in_frd(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root)
    aug = CosFlyAdapter(
        fps=2.0,
        action_horizon=8,
        media_cache_root=tmp_path / "cache",
    ).load_episodes(
        source_root,
        split="vln_train",
        max_episodes=2,
    )[1]

    yaw5 = math.radians(5.0)
    yaw95 = math.radians(95.0)
    assert np.allclose(aug.frames[0].state, [0.1, 0.0, -10.0, yaw5], atol=1e-7)
    assert np.allclose(aug.frames[1].state, [1.1, 0.0, -9.5, yaw5], atol=1e-7)
    assert np.allclose(aug.frames[2].state, [1.0, 1.1, -9.0, yaw95], atol=1e-7)

    assert np.allclose(
        aug.frames[0].action[0],
        [math.cos(yaw5), -math.sin(yaw5), 0.5, 0.0],
        atol=1e-6,
    )
    assert np.isclose(aug.frames[0].action[1][3], math.pi / 2, atol=1e-6)
    assert np.isclose(aug.frames[1].action[0][3], math.pi / 2, atol=1e-6)
    assert aug.frames[0].action_available is True
    assert aug.frames[-1].action == []
    assert aug.frames[-1].action_available is False
    assert [frame.timestamp for frame in aug.frames] == [0.0, 0.5, 1.0]


def test_cosfly_adapter_rejects_a_selected_trace_with_missing_rgb(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import CosFlyAdapter

    source_root = tmp_path / "source"
    _write_parent(source_root, missing_rgb_index=1)

    with pytest.raises(FileNotFoundError, match="missing or empty CosFly RGB frame"):
        CosFlyAdapter(
            fps=2.0,
            action_horizon=8,
            media_cache_root=tmp_path / "cache",
        ).load_episodes(
            source_root,
            split="train",
            max_episodes=2,
        )


def test_cosfly_convert_places_media_cache_under_base_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool.navvla.adapters import cosfly

    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _write_parent(source_root)
    captured = {}

    def fake_writer(episodes, *, output_root, spec, **kwargs):
        captured["episodes"] = episodes
        captured["output_root"] = Path(output_root)
        captured["dataset_name"] = spec.dataset_name
        captured["state_mode"] = spec.state_mode
        return {"dataset_root": str(Path(output_root) / spec.dataset_name)}

    def fake_finalize(dataset_root, *, target_split):
        captured["finalize"] = (Path(dataset_root), target_split)
        return {"dataset_root": str(dataset_root)}

    monkeypatch.setattr(cosfly, "write_cosfly_lerobot_dataset", fake_writer)
    monkeypatch.setattr(cosfly, "finalize_cosfly_metadata", fake_finalize)
    cosfly.CosFlyAdapter(fps=2.0, action_horizon=8).convert(
        source_root=source_root,
        output_root=output_root,
        dataset_name="vln_train",
        max_episodes=2,
        fps=2.0,
        control_frequency_hz=2.0,
        action_horizon=8,
        overwrite=True,
        write_visual_token_cache=False,
    )

    resized_path = Path(captured["episodes"][0].frames[0].media_paths["front_image"])
    assert resized_path.is_relative_to(output_root / "_media_cache_384x256")
    assert captured["output_root"] == output_root
    assert captured["dataset_name"] == "vln_train"
    assert captured["state_mode"] == "source_world_absolute_pose_xyz_yaw"
    assert captured["finalize"] == (output_root / "vln_train", "vln_train")
    assert captured["episodes"][0].episode_id.endswith("__ORI")
    assert captured["episodes"][1].episode_id.endswith("__aug_001")


def test_cosfly_metadata_finalizer_standardizes_identity_state_and_source_split(tmp_path: Path) -> None:
    from tool.navvla.adapters.cosfly import finalize_cosfly_metadata

    root = tmp_path / "train"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "dataset_name": "train",
                "splits": {"vln_train": "0:2"},
                "navvla": {
                    "state_mode": "cosfly_carla_world_pose_x_forward_y_right_z_down_yaw_right_positive",
                    "action_mode": "anchor_relative_body_frame_xyz_yaw",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "dataset_statistics.json").write_text(
        json.dumps({"train_vln_train": {"action": {"q01": [0, 0, 0, 0], "q99": [1, 1, 1, 1]}}}),
        encoding="utf-8",
    )
    frame_metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    frame_metadata_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "index": index,
                    "source_frame_index": index,
                    "source_metadata": {
                        "source_dataset": "cosfly",
                        "source_split": "all",
                        "target_split": "vln_train",
                        "target_bbox_2d": {"xmin": 1.0},
                    },
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    report = finalize_cosfly_metadata(root, target_split="vln_train")

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    statistics = json.loads((root / "dataset_statistics.json").read_text(encoding="utf-8"))
    frame_rows = [json.loads(line) for line in frame_metadata_path.read_text(encoding="utf-8").splitlines()]
    assert info["dataset_name"] == "vln_train"
    assert info["navvla"]["state_mode"] == "source_world_absolute_pose_xyz_yaw"
    assert list(statistics) == ["vln_train_vln_train"]
    assert all(row["source_metadata"]["source_split"] == "train" for row in frame_rows)
    assert all(row["source_metadata"]["target_bbox_2d"] == {"xmin": 1.0} for row in frame_rows)
    assert report["frame_metadata_rows"] == 2

    first_bytes = {
        path: path.read_bytes()
        for path in (root / "meta" / "info.json", root / "dataset_statistics.json", frame_metadata_path)
    }
    finalize_cosfly_metadata(root, target_split="train")
    assert {path: path.read_bytes() for path in first_bytes} == first_bytes
