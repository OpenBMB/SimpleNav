from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_enhanced_fixture(root: Path, *, dataset_key: str, episode_count: int = 1) -> Path:
    source = root / "vln_train_enhanced"
    (source / "trajectories").mkdir(parents=True)
    (source / "meta").mkdir()
    (source / "videos").mkdir()

    if dataset_key == "OpenFly_lerobot":
        source_pose_kind = "original_openfly_annotation_world_pose"
        render_transform = "reflect-y-z-yaw"
        scene_id = "env_airsim_18"
    else:
        source_pose_kind = "original_aerialvln_reference_path_world_pose"
        render_transform = "identity"
        scene_id = "5"

    views = {
        "front": (0.25, 0.0, 0.0, 5.0, 2.0, -3.0, 100.0),
        "back": (-0.25, 0.0, 0.0, 185.0, -2.0, 3.0, 101.0),
        "left": (0.0, -0.25, 0.0, -85.0, 1.0, 4.0, 102.0),
        "right": (0.0, 0.25, 0.0, 95.0, -1.0, -4.0, 103.0),
    }
    episodes = []
    frame_rows = []
    camera_rows = []
    video_rows = []
    for episode_index in range(episode_count):
        episode_id = f"episode-{episode_index}__enhanced_v1"
        trajectory_id = f"scene/astar_data/low_short/trajectory-{episode_index}__enhanced_v1"
        offset = float(episode_index * 100)
        if dataset_key == "OpenFly_lerobot":
            reference_path = [[10.0 + offset + index, -20.0, -(30.0 + index), 0.0, 0.0, 0.0] for index in range(11)]
        else:
            reference_path = [[10.0 + offset + index, 20.0, 30.0 + index, 0.0, 0.0, 0.0] for index in range(11)]
        episodes.append(
            {
                "episode_id": episode_id,
                "trajectory_id": trajectory_id,
                "scene_id": scene_id,
                "start_position": reference_path[0][:3],
                "start_rotation": [1.0, 0.0, 0.0, 0.0],
                "goals": [{"position": reference_path[-1][:3]}],
                "reference_path": reference_path,
                "actions": [],
                "instruction": {"instruction_text": f"fly forward {episode_index}"},
            }
        )
        for image_index, waypoint_index in enumerate((0, 5, 10)):
            global_index = episode_index * 3 + image_index
            frame_rows.append(
                {
                    "episode_id": episode_id,
                    "image_index": image_index,
                    "index": global_index,
                    "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "position_xyz": reference_path[waypoint_index][:3],
                    "request_id": f"request-{episode_index}-{image_index}",
                    "scene_id": scene_id,
                    "source_episode_index": episode_index,
                    "status": "available",
                    "timestamp": float(waypoint_index),
                    "waypoint_index": waypoint_index,
                }
            )
        for view, (x, y, z, yaw, roll, pitch, fov) in views.items():
            camera_rows.append(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "view": view,
                    "camera_name": f"{view}_0",
                    "seed": episode_index + 1,
                    "base_pose": {"x": x, "y": y, "z": z, "yaw": yaw, "roll": roll, "pitch": pitch},
                    "pose_delta": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0},
                    "final_pose": {"x": x, "y": y, "z": z, "yaw": yaw, "roll": roll, "pitch": pitch},
                    "fov_degrees": fov,
                    "render_status": "complete",
                }
            )
            for image_index in range(3):
                global_index = episode_index * 3 + image_index
                video_rows.append(
                    {
                        "index": global_index,
                        "video_key": f"{view}_image",
                        "available": True,
                        "video_frame_index": global_index,
                        "chunk_index": 0,
                        "file_index": 0,
                    }
                )

    _write_jsonl(source / "trajectories" / "episodes.jsonl", episodes)
    _write_jsonl(source / "meta" / "navvla_multiview_frame_metadata.jsonl", frame_rows)
    _write_jsonl(source / "meta" / "navvla_episode_camera_parameters.jsonl", camera_rows)

    cameras = []
    for view, (x, y, z, yaw, roll, pitch, fov) in views.items():
        cameras.append(
            {
                "view": view,
                "camera_name": f"{view}_0",
                "video_key": f"{view}_image",
                "base_pose": {"x": x, "y": y, "z": z, "yaw": yaw, "roll": roll, "pitch": pitch},
                "base_fov_degrees": fov,
            }
        )
        video_dir = source / "videos" / f"{view}_image" / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "part-000.mp4").write_bytes(f"{dataset_key}:{view}".encode())
    camera_order = ["front_image", "back_image", "left_image", "right_image"]
    video_rows.sort(key=lambda row: (row["index"], camera_order.index(row["video_key"])))
    (source / "meta" / "navvla_cameras.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "calibration_status": "episode_randomized",
                "actual_episode_parameters": "navvla_episode_camera_parameters.jsonl",
                "cameras": cameras,
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(__import__("pyarrow").Table.from_pylist(video_rows), source / "meta" / "navvla_video_index.parquet")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_key": dataset_key,
                "trajectory_format": "aerialvln_episode_shell_absolute_pose_sequence",
                "source_pose_kind": source_pose_kind,
                "image_status": "collected",
                "complete_lerobot_split": False,
                "episode_count": episode_count,
                "render_request_count": 3 * episode_count,
                "missing_render_request_count": 0,
                "views": ["front", "back", "left", "right"],
                "render_coordinate_transform": render_transform,
            }
        ),
        encoding="utf-8",
    )
    return source


def test_openfly_render_poses_are_converted_to_first_body_aligned_frd() -> None:
    from tool.navvla.adapters.enhanced_vln import render_poses_to_training_local

    render = np.asarray(
        [
            [10.0, -20.0, -30.0, math.pi / 2.0],
            [10.0, -25.0, -32.0, math.pi / 2.0 + 0.2],
        ]
    )

    local = render_poses_to_training_local(render, dataset_key="OpenFly_lerobot")

    np.testing.assert_allclose(local[0], [0.0, 0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(local[1], [-5.0, 0.0, 2.0, 0.2], atol=1e-6)


def test_camera_pose_has_real_offset_body_angles_and_fov_in_radians() -> None:
    from tool.navvla.adapters.enhanced_vln import camera_pose7

    pose = camera_pose7(
        vehicle_pose=[2.0, 3.0, 4.0, math.pi / 2.0],
        camera_parameters={
            "final_pose": {"x": 0.25, "y": 0.1, "z": -0.2, "yaw": 10.0, "roll": 2.0, "pitch": -3.0},
            "fov_degrees": 100.0,
        },
    )

    np.testing.assert_allclose(
        pose,
        [1.9, 3.25, 3.8, math.radians(10.0), math.radians(2.0), math.radians(-3.0), math.radians(100.0)],
        atol=1e-6,
    )


def test_converter_writes_independent_seven_dimensional_multiview_lerobot(tmp_path: Path) -> None:
    from tool.navvla.adapters.enhanced_vln import convert_enhanced_vln_package

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="OpenFly_lerobot")
    source_video = source / "videos" / "front_image" / "chunk-000" / "part-000.mp4"
    source_hash = _sha256(source_video)
    output_parent = tmp_path / "output"

    summary = convert_enhanced_vln_package(
        source,
        output_root=output_parent,
        dataset_name="vln_train_enhanced_lerobot",
        max_episodes=None,
        build_derived_artifacts=False,
    )

    output = Path(summary["dataset_root"])
    data_path = output / "data" / "chunk-000" / "part-000.parquet"
    table = pq.read_table(data_path)
    rows = table.to_pylist()
    assert [row["timestamp"] for row in rows] == [0.0, 5.0, 10.0]
    assert [row["source_frame_index"] for row in rows] == [0, 5, 10]
    np.testing.assert_allclose(rows[0]["observation.state"], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(rows[1]["observation.state"], [5.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(rows[0]["action"][0], [1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(rows[0]["action"][7], [8.0, 0.0, 8.0, 0.0])
    np.testing.assert_allclose(rows[1]["action"][-1], [5.0, 0.0, 5.0, 0.0])
    assert all(not value for row in rows for value in row["action.padding_mask"])
    for view in ("front", "back", "left", "right"):
        assert len(rows[0][f"observation.camera_pose.{view}"]) == 7

    info = json.loads((output / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["fps"] == 1.0
    assert info["navvla"]["control_frequency_hz"] == 1.0
    assert info["navvla"]["camera_pose_dim"] == 7
    assert info["features"]["observation.camera_pose.front"]["names"][-1] == "fov"

    copied_video = output / "videos" / "front_image" / "chunk-000" / "part-000.mp4"
    assert _sha256(copied_video) == source_hash
    assert _sha256(source_video) == source_hash
    assert copied_video.stat().st_ino != source_video.stat().st_ino


def test_enhanced_vln_is_available_from_the_unified_convert_cli() -> None:
    from tool.navvla.adapters import get_adapter
    from tool.navvla.cli.convert_dataset import build_parser

    assert get_adapter("enhanced_vln").name == "enhanced_vln"
    args = build_parser().parse_args(
        [
            "--adapter",
            "enhanced_vln",
            "--source-root",
            "/tmp/source",
            "--output-root",
            "/tmp/output",
            "--dataset-name",
            "vln_train_enhanced_lerobot",
            "--no-visual-token-cache",
        ]
    )
    assert args.adapter == "enhanced_vln"


def test_partial_preview_video_is_trimmed_to_the_selected_prefix(tmp_path: Path) -> None:
    from tool.navvla.adapters.enhanced_vln import copy_video_prefix

    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=16x16:r=1:d=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    copy_video_prefix(source, target, selected_frame_count=2, source_frame_count=3, fps=1.0)

    capture = cv2.VideoCapture(str(target))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert capture.get(cv2.CAP_PROP_FPS) == 1.0
    finally:
        capture.release()


def test_conversion_report_paths_are_rebased_from_staging_to_final_output(tmp_path: Path) -> None:
    from tool.navvla.adapters.enhanced_vln import rebase_report_paths

    staging = tmp_path / ".preview.tmp"
    target = tmp_path / "preview"
    report = {
        "dataset_root": str(staging),
        "nested": {"path": str(staging / "meta" / "info.json")},
        "paths": [str(staging / "cache"), "unchanged"],
    }

    rebased = rebase_report_paths(report, staging=staging, target=target)

    assert rebased["dataset_root"] == str(target)
    assert rebased["nested"]["path"] == str(target / "meta" / "info.json")
    assert rebased["paths"] == [str(target / "cache"), "unchanged"]


def test_converter_builds_only_the_1024_context_index(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.adapters import enhanced_vln

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="AerialVLN_lerobot")
    captured: dict[str, object] = {}

    def fake_repair(dataset_root, **kwargs):
        captured.update(kwargs)
        return {"dataset_root": str(dataset_root), "validation": {"valid": True}}

    monkeypatch.setattr(enhanced_vln, "repair_navvla_dataset", fake_repair)
    enhanced_vln.convert_enhanced_vln_package(
        source,
        output_root=tmp_path / "output",
        dataset_name="preview",
        build_derived_artifacts=True,
    )

    assert captured["token_budgets"] == (1024,)


def test_enhanced_vln_adapter_accepts_multiple_load_workers() -> None:
    from tool.navvla.adapters.enhanced_vln import EnhancedVLNAdapter

    adapter = EnhancedVLNAdapter().configure(fps=1.0, action_horizon=8, load_workers=4)

    assert adapter.load_workers == 4


def test_multiprocess_conversion_matches_single_process_output(tmp_path: Path) -> None:
    from tool.navvla.adapters.enhanced_vln import convert_enhanced_vln_package

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="OpenFly_lerobot", episode_count=8)
    single_summary = convert_enhanced_vln_package(
        source,
        output_root=tmp_path / "single",
        dataset_name="vln_train_enhanced_lerobot",
        build_derived_artifacts=False,
        load_workers=1,
    )
    multi_summary = convert_enhanced_vln_package(
        source,
        output_root=tmp_path / "multi",
        dataset_name="vln_train_enhanced_lerobot",
        build_derived_artifacts=False,
        load_workers=2,
    )

    single = Path(single_summary["dataset_root"])
    multi = Path(multi_summary["dataset_root"])
    assert pq.read_table(single / "data/chunk-000/part-000.parquet").equals(
        pq.read_table(multi / "data/chunk-000/part-000.parquet")
    )
    for relative in (
        "meta/navvla_frame_metadata.jsonl",
        "meta/navvla_multiview_frame_metadata.jsonl",
        "meta/navvla_episode_camera_parameters.jsonl",
        "meta/navvla_tasks.jsonl",
    ):
        assert (single / relative).read_bytes() == (multi / relative).read_bytes()
    assert multi_summary["load_workers"] == 2
    assert multi_summary["video_copy_workers"] == 2


def test_keyboard_interrupt_removes_conversion_staging_directory(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.adapters import enhanced_vln

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="AerialVLN_lerobot")
    output = tmp_path / "output"

    def interrupt_copy(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(enhanced_vln, "_copy_videos", interrupt_copy)
    with pytest.raises(KeyboardInterrupt):
        enhanced_vln.convert_enhanced_vln_package(
            source,
            output_root=output,
            dataset_name="preview",
            build_derived_artifacts=False,
        )

    assert not list(output.glob(".preview.tmp-*"))


def test_finalize_staging_repairs_and_atomically_publishes_target(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.adapters import enhanced_vln

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="AerialVLN_lerobot")
    summary = enhanced_vln.convert_enhanced_vln_package(
        source,
        output_root=tmp_path / "output",
        dataset_name="preview",
        build_derived_artifacts=False,
    )
    target = Path(summary["dataset_root"])
    staging = target.with_name(".preview.tmp-resume-test")
    target.rename(staging)
    captured = {}

    def fake_repair(dataset_root, **kwargs):
        captured.update(kwargs)
        return {"dataset_root": str(dataset_root), "validation": {"valid": True}}

    monkeypatch.setattr(enhanced_vln, "repair_navvla_dataset", fake_repair)

    report = enhanced_vln.finalize_enhanced_vln_staging(staging)

    assert target.is_dir()
    assert not staging.exists()
    assert "cache_workers" not in captured
    assert report["dataset_root"] == str(target)
    assert report["derived_artifacts"]["dataset_root"] == str(target)


def test_derived_artifact_failure_preserves_completed_staging_for_resume(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.adapters import enhanced_vln

    source = _build_enhanced_fixture(tmp_path / "source", dataset_key="AerialVLN_lerobot")
    output = tmp_path / "output"

    def fail_repair(*_args, **_kwargs):
        raise RuntimeError("repair failed")

    monkeypatch.setattr(enhanced_vln, "repair_navvla_dataset", fail_repair)
    with pytest.raises(RuntimeError, match="repair failed"):
        enhanced_vln.convert_enhanced_vln_package(
            source,
            output_root=output,
            dataset_name="preview",
            build_derived_artifacts=True,
        )

    staging = list(output.glob(".preview.tmp-*"))
    assert len(staging) == 1
    assert (staging[0] / "conversion_report.json").is_file()
