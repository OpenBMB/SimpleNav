from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import NavVLAeval.plot_eval_trajectories as plot_eval_trajectories
from NavVLAeval.plot_eval_trajectories import (
    SourceEpisodeIndex,
    build_episode_comparison,
    comparison_payload,
    main,
    select_eval_infos,
    summarize_evaluation_performance,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _eval_info(*, benchmark: str, source_episode_id: str, input_root: Path, sr: float = 1.0) -> dict:
    return {
        "episode_uid": f"split:{source_episode_id}",
        "source_episode_id": source_episode_id,
        "benchmark": benchmark,
        "run_name": "fixture",
        "input_root": str(input_root),
        "input_namespace": "seen",
        "scene_id": "scene",
        "instruction": "move to the goal",
        "status": "completed",
        "termination_reason": "success" if sr else "max_steps",
        "metrics": {"SR": sr, "NE": 1.0},
        "paths": {"data": "data"},
    }


def test_openfly_comparison_transforms_reference_and_uses_dense_actual_waypoints(tmp_path: Path) -> None:
    source_path = tmp_path / "seen.json"
    _write_json(
        source_path,
        [
            {
                "episode_id": "000001",
                "pos": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "yaw": [0.25, 0.5],
            }
        ],
    )
    episode_dir = tmp_path / "run" / "logs" / "scene" / "seen" / "000001"
    eval_info_path = episode_dir / "eval_info.json"
    _write_json(eval_info_path, _eval_info(benchmark="openfly", source_episode_id="000001", input_root=source_path))
    _write_json(
        episode_dir / "data" / "000000.json",
        {
            "diagnostics": {
                "actual_waypoint_poses": [[1.5, -2.5, -3.5, -0.25], [2.0, -3.0, -4.0, -0.2]]
            },
            "state": [99.0, 99.0, 99.0, 0.0],
        },
    )

    comparison = build_episode_comparison(
        eval_info_path,
        config={"benchmark": {"name": "openfly"}, "input": {"source_z_sign": -1}},
        source_index=SourceEpisodeIndex(),
        allow_reference_only=False,
    )

    np.testing.assert_allclose(comparison.reference_poses[:, :3], [[1.0, -2.0, -3.0], [4.0, -5.0, -6.0]])
    np.testing.assert_allclose(
        comparison.executed_poses[:, :3],
        [[1.0, -2.0, -3.0], [1.5, -2.5, -3.5], [2.0, -3.0, -4.0]],
    )


def test_openfly_reference_supports_legacy_pose_converter_without_source_z_sign(monkeypatch: pytest.MonkeyPatch) -> None:
    def legacy_source_pose_to_canonical(values: list[float]) -> tuple[list[float], float]:
        x, y, z, yaw = values
        return [x, -y, z], -yaw

    monkeypatch.setattr(plot_eval_trajectories, "_source_pose_to_canonical", legacy_source_pose_to_canonical)

    reference = plot_eval_trajectories._openfly_reference_poses(
        {"pos": [[1.0, 2.0, 3.0]], "yaw": [0.25]},
        source_z_sign=-1.0,
    )

    np.testing.assert_allclose(reference, [[1.0, -2.0, -3.0, -0.25]])


def test_aerialvln_comparison_falls_back_to_step_state(tmp_path: Path) -> None:
    source_path = tmp_path / "val_seen.json"
    _write_json(
        source_path,
        {
            "episodes": [
                {
                    "episode_id": "episode-a",
                    "reference_path": [[0.0, 1.0, 2.0, 0.0, 0.0, 0.2], [3.0, 4.0, 5.0, 0.0, 0.0, 0.3]],
                }
            ]
        },
    )
    episode_dir = tmp_path / "run" / "logs" / "10" / "split" / "episode-a"
    eval_info_path = episode_dir / "eval_info.json"
    _write_json(eval_info_path, _eval_info(benchmark="aerialvln", source_episode_id="episode-a", input_root=source_path))
    _write_json(episode_dir / "data" / "000000.json", {"state": [1.0, 2.0, 3.0, 0.4]})

    comparison = build_episode_comparison(
        eval_info_path,
        config={"benchmark": {"name": "aerialvln"}},
        source_index=SourceEpisodeIndex(),
        allow_reference_only=False,
    )

    np.testing.assert_allclose(comparison.reference_poses, [[0.0, 1.0, 2.0, 0.2], [3.0, 4.0, 5.0, 0.3]])
    np.testing.assert_allclose(comparison.executed_poses, [[0.0, 1.0, 2.0, 0.2], [1.0, 2.0, 3.0, 0.4]])


def test_openfly_single_episode_file_uses_filename_as_source_id(tmp_path: Path) -> None:
    source_path = tmp_path / "episodes" / "001234.json"
    _write_json(source_path, {"frames": [{"state": [1.0, 2.0, 3.0, 0.4]}]})
    episode_dir = tmp_path / "run" / "logs" / "scene" / "seen" / "001234"
    eval_info_path = episode_dir / "eval_info.json"
    _write_json(eval_info_path, _eval_info(benchmark="openfly", source_episode_id="001234", input_root=tmp_path))

    comparison = build_episode_comparison(
        eval_info_path,
        config={"benchmark": {"name": "openfly"}, "input": {"source_z_sign": -1}},
        source_index=SourceEpisodeIndex(),
        allow_reference_only=True,
    )

    np.testing.assert_allclose(comparison.reference_poses[0, :3], [1.0, -2.0, -3.0])


def test_missing_steps_requires_explicit_reference_only_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "seen.json"
    _write_json(source_path, [{"episode_id": "000001", "pos": [[1.0, 2.0, 3.0]], "yaw": [0.0]}])
    eval_info_path = tmp_path / "run" / "logs" / "scene" / "seen" / "000001" / "eval_info.json"
    _write_json(eval_info_path, _eval_info(benchmark="openfly", source_episode_id="000001", input_root=source_path))
    config = {"benchmark": {"name": "openfly"}, "input": {"source_z_sign": -1}}

    with pytest.raises(FileNotFoundError, match="no per-step trajectory artifacts"):
        build_episode_comparison(
            eval_info_path,
            config=config,
            source_index=SourceEpisodeIndex(),
            allow_reference_only=False,
        )

    comparison = build_episode_comparison(
        eval_info_path,
        config=config,
        source_index=SourceEpisodeIndex(),
        allow_reference_only=True,
    )
    assert comparison.executed_poses is None
    assert comparison_payload(comparison, png_path=tmp_path / "plot.png")["trajectory_mode"] == "reference_only"


def test_balanced_selection_is_deterministic(tmp_path: Path) -> None:
    paths = []
    for index, sr in enumerate((1.0, 1.0, 0.0, 0.0)):
        path = tmp_path / str(index) / "eval_info.json"
        _write_json(path, _eval_info(benchmark="openfly", source_episode_id=str(index), input_root=tmp_path, sr=sr))
        paths.append(path)

    first = select_eval_infos(paths, episode_ids=None, sample_count=4, selection="balanced", seed=7)
    second = select_eval_infos(paths, episode_ids=None, sample_count=4, selection="balanced", seed=7)

    assert first == second
    selected_sr = [json.loads(path.read_text())["metrics"]["SR"] for path in first]
    assert selected_sr.count(1.0) == 2
    assert selected_sr.count(0.0) == 2


def test_automatic_selection_excludes_runtime_failures(tmp_path: Path) -> None:
    paths = []
    records = [
        {"source_episode_id": "success", "status": "completed", "failure": None, "metrics": {"SR": 1.0}},
        {"source_episode_id": "max-steps", "status": "completed", "failure": None, "metrics": {"SR": 0.0}},
        {
            "source_episode_id": "runtime-failure",
            "status": "failed",
            "failure": "TimeoutError: Request timed out",
            "failure_type": "benchmark_runtime",
            "metrics": {"SR": 0.0},
        },
    ]
    for index, record in enumerate(records):
        path = tmp_path / str(index) / "eval_info.json"
        _write_json(path, record)
        paths.append(path)

    selected = select_eval_infos(paths, episode_ids=None, sample_count=3, selection="balanced", seed=42)
    selected_ids = {json.loads(path.read_text(encoding="utf-8"))["source_episode_id"] for path in selected}

    assert selected_ids == {"success", "max-steps"}
    assert select_eval_infos(
        paths,
        episode_ids=["runtime-failure"],
        sample_count=1,
        selection="balanced",
        seed=42,
    ) == [paths[2]]


def test_main_writes_combined_views_and_performance_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source_path = tmp_path / "episodes.json"
    output_dir = tmp_path / "plots"
    _write_json(
        source_path,
        [
            {
                "episode_id": "000001",
                "frames": [
                    {"state": [0.0, 0.0, 0.0, 0.0]},
                    {"state": [1.0, 0.0, 0.0, 0.0]},
                ],
            },
            {
                "episode_id": "000002",
                "frames": [
                    {"state": [0.0, 0.0, 0.0, 0.0]},
                    {"state": [0.0, 2.0, 0.0, 0.0]},
                ],
            },
        ],
    )
    run_root.mkdir(parents=True)
    (run_root / "config.yaml").write_text(
        yaml.safe_dump({"benchmark": {"name": "openfly"}, "input": {"source_z_sign": 1.0}}),
        encoding="utf-8",
    )
    for episode_id, scene_id, sr, waypoint in (
        ("000001", "env_airsim_23", 1.0, [1.0, 0.0, 0.0, 0.0]),
        ("000002", "env_airsim_16", 0.0, [0.0, 1.0, 0.0, 0.0]),
    ):
        episode_root = run_root / "logs" / scene_id / "seen" / episode_id
        _write_json(
            episode_root / "eval_info.json",
            {
                "benchmark": "openfly",
                "run_name": "test-run",
                "episode_uid": f"seen:{episode_id}",
                "source_episode_id": episode_id,
                "input_root": str(source_path),
                "scene_id": scene_id,
                "status": "completed",
                "failure": None,
                "metrics": {"SR": sr, "NE": 0.0 if sr else 2.0},
                "termination_reason": "success" if sr else "max_steps",
                "paths": {"data": "data"},
            },
        )
        _write_json(
            episode_root / "data" / "000000.json",
            {"diagnostics": {"actual_waypoint_poses": [waypoint]}},
        )

    assert main(
        [
            "--run-root",
            str(run_root),
            "--output-dir",
            str(output_dir),
            "--episode-ids",
            "000001",
            "000002",
            "--dpi",
            "40",
        ]
    ) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["performance_summary"]["written_episodes"] == 2
    assert manifest["performance_summary"]["metrics"]["SR"] == 0.5
    assert manifest["performance_summary_path"] == str(output_dir / "performance_summary.json")
    assert manifest["combined_plot"] == str(output_dir / "combined_trajectories.png")
    assert manifest["combined_plots"] == {
        "xy": str(output_dir / "combined_xy.png"),
        "xz": str(output_dir / "combined_xz.png"),
        "yz": str(output_dir / "combined_yz.png"),
        "3d": str(output_dir / "combined_3d.png"),
    }
    assert (output_dir / "performance_summary.json").is_file()
    assert (output_dir / "env_airsim_23__000001.png").is_file()
    assert (output_dir / "env_airsim_16__000002.png").is_file()
    assert (output_dir / "combined_trajectories.png").stat().st_size > 1_000
    for view in ("xy", "xz", "yz", "3d"):
        assert (output_dir / f"combined_{view}.png").stat().st_size > 1_000


def test_performance_summary_excludes_runtime_failures_from_model_metrics(tmp_path: Path) -> None:
    paths = []
    records = [
        {
            "scene_id": "env_airsim_16",
            "status": "completed",
            "termination_reason": "success",
            "failure": None,
            "metrics": {"SR": 1.0, "OSR": 1.0, "NE": 10.0, "standard_SPL": 0.8},
        },
        {
            "scene_id": "env_airsim_23",
            "status": "completed",
            "termination_reason": "max_steps",
            "failure": None,
            "metrics": {"SR": 0.0, "OSR": 0.0, "NE": 50.0, "standard_SPL": 0.0},
        },
        {
            "scene_id": "env_airsim_16",
            "status": "failed",
            "termination_reason": "failure",
            "failure": "TimeoutError: Request timed out",
            "failure_type": "benchmark_runtime",
            "metrics": {"SR": 0.0, "OSR": 0.0, "NE": 0.0, "standard_SPL": 0.0},
        },
    ]
    for index, record in enumerate(records):
        path = tmp_path / "logs" / str(index) / "eval_info.json"
        _write_json(path, record)
        paths.append(path)
    run_plan_path = tmp_path / "run_plan.json"
    _write_json(run_plan_path, {"total_episode_uids": ["0", "1", "2", "3"]})

    summary = summarize_evaluation_performance(
        paths,
        run_plan_path=run_plan_path,
        metric_keys=("SR", "OSR", "NE", "standard_SPL"),
    )

    assert summary["planned_episodes"] == 4
    assert summary["written_episodes"] == 3
    assert summary["progress_percent"] == 75.0
    assert summary["completed_episodes"] == 2
    assert summary["failed_episodes"] == 1
    assert summary["metrics"] == {"SR": 0.5, "OSR": 0.5, "NE": 30.0, "standard_SPL": 0.4}
    assert summary["scene_metrics"]["env_airsim_16"]["metric_episodes"] == 1
    assert summary["failure_breakdown"] == {"benchmark_runtime": 1}
    assert summary["termination_breakdown"] == {"failure": 1, "max_steps": 1, "success": 1}
    assert summary["metrics_including_failures_as_zero"]["SR"] == 1.0 / 3.0
