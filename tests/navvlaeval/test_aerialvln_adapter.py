from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from NavVLAeval.common.config import EnvConfig, InputConfig, InputRootConfig, load_eval_config
from NavVLAeval.common.data.runtime_dataset import (
    AERIALVLN_PLATFORM_TEXT,
    AerialVLNRuntimeDatasetAdapter,
    VLNCE_PLATFORM_TEXT,
    get_runtime_dataset_adapter,
)
from NavVLAeval.common.runner import worker as worker_module
from NavVLAeval.common.runner.backend_plan import WorkerBackendPlan
from NavVLAeval.common.simulators.airsim.backend import AIRSIM_RPC_TIMEOUT_SEC
from NavVLAeval.common.simulators.airsim.process import (
    AirSimLaunchConfig,
    build_airsim_launch_command,
    copytree_with_hardlinks,
    resolve_airsim_start_script,
)
from NavVLAeval.common.simulators.airsim.settings import build_airsim_settings
from NavVLAeval.common.types import ActionPrediction, EnvironmentStepResult, EpisodeHistory, EpisodeResult, EvalEpisode, Pose4D, StepState
from NavVLAeval.aerialvln.benchmark import AerialVLNBenchmarkSpec
from NavVLAeval.aerialvln.inputs import AerialVLNJsonInputAdapter
from NavVLAeval.traveluav.inputs import TravelUAVJsonInputAdapter, TravelUAVLeRobotV3InputAdapter
from NavVLAeval.vlnce.benchmark import VLNCERuntime


def _write_aerialvln_json(path: Path) -> None:
    payload = {
        "episodes": [
            {
                "episode_id": 101,
                "scene_id": 26,
                "start_position": [1.0, 2.0, -3.0],
                "start_rotation": [1.0, 0.0, 0.0, 0.0],
                "goals": [{"position": [11.0, 22.0, -33.0]}],
                "reference_path": [
                    [1.0, 2.0, -3.0, 0.0, 0.0, 0.0],
                    [11.0, 22.0, -33.0, 0.0, 0.0, math.pi / 2],
                ],
                "instruction": {
                    "instruction_text": "Fly to the building entrance.",
                    "actions": ["forward"],
                },
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_traveluav_lerobot_root(tmp_path: Path) -> Path:
    root = tmp_path / "vln_val_seen"
    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode_index": 0,
                "episode_id": "town01",
                "scene_id": "Carla_Town01",
                "tasks": ["find target one"],
                "length": 1,
                "task_index": 0,
                "trajectory_id": "trajectory-town01",
            },
            {
                "episode_index": 1,
                "episode_id": "town04",
                "scene_id": "Carla_Town04",
                "tasks": ["find target four"],
                "length": 1,
                "task_index": 1,
                "trajectory_id": "trajectory-town04",
            },
        ]
    ).to_parquet(episodes_dir / "part-000.parquet", index=False)
    pd.DataFrame(
        [
            {
                "index": 0,
                "episode_index": 0,
                "frame_index": 0,
                "observation.state": [0.0, 0.0, 0.0, 0.0],
            },
            {
                "index": 1,
                "episode_index": 1,
                "frame_index": 0,
                "observation.state": [1.0, 1.0, 1.0, 0.0],
            },
        ]
    ).to_parquet(data_dir / "part-000.parquet", index=False)
    (root / "meta" / "navvla_frame_metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "source_frame_index": 0,
                        "source_metadata": {"source_state": [10.0, 20.0, -5.0, 0.0, 0.0, 0.25]},
                    }
                ),
                json.dumps(
                    {
                        "index": 1,
                        "source_frame_index": 0,
                        "source_metadata": {"source_state": [40.0, 50.0, -7.0, 0.0, 0.0, -0.5]},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_aerialvln_json_input_emits_normalized_episode(tmp_path: Path) -> None:
    source_path = tmp_path / "val_seen.json"
    _write_aerialvln_json(source_path)
    cfg = InputConfig(
        type="aerialvln_json",
        adapter_class_path="NavVLAeval.aerialvln.inputs:AerialVLNJsonInputAdapter",
        namespace="aerialvln_val_seen",
        path=source_path,
    )

    episodes = AerialVLNJsonInputAdapter().load_episodes(cfg, max_samples=None)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.episode_uid == "aerialvln_val_seen:101"
    assert episode.scene_id == "26"
    assert episode.instruction == "Fly to the building entrance."
    assert episode.payload["env_name"] == "env_26"
    assert episode.payload["start_pose"] == [1.0, 2.0, -3.0, 0.0]
    assert episode.payload["goal_position"] == [11.0, 22.0, -33.0]
    assert episode.payload["reference_path_m"][1] == [11.0, 22.0, -33.0, math.pi / 2]


def test_aerialvln_jsonl_input_emits_normalized_episode(tmp_path: Path) -> None:
    source_path = tmp_path / "val_seen_50.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in {
                "episodes": [
                    {
                        "episode_id": 101,
                        "scene_id": 26,
                        "start_position": [1.0, 2.0, -3.0],
                        "start_rotation": [1.0, 0.0, 0.0, 0.0],
                        "goals": [{"position": [11.0, 22.0, -33.0]}],
                        "reference_path": [
                            [1.0, 2.0, -3.0, 0.0, 0.0, 0.0],
                            [11.0, 22.0, -33.0, 0.0, 0.0, math.pi / 2],
                        ],
                        "instruction": {"instruction_text": "Fly to the building entrance."},
                    }
                ]
            }["episodes"]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = InputConfig(
        type="aerialvln_json",
        adapter_class_path="NavVLAeval.aerialvln.inputs:AerialVLNJsonInputAdapter",
        namespace="aerialvln_val_seen_50",
        path=source_path,
    )

    episodes = AerialVLNJsonInputAdapter().load_episodes(cfg, max_samples=None)

    assert [episode.source_episode_id for episode in episodes] == ["101"]
    assert episodes[0].payload["env_name"] == "env_26"


def test_aerialvln_json_input_filters_episode_ids_before_max_samples(tmp_path: Path) -> None:
    source_path = tmp_path / "val_seen.json"
    payload = {
        "episodes": [
            {
                "episode_id": "skip",
                "scene_id": 10,
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [1.0, 0.0, 0.0, 0.0],
                "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "instruction": "skip",
            },
            {
                "episode_id": "keep",
                "scene_id": 10,
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [1.0, 0.0, 0.0, 0.0],
                "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "instruction": "keep",
            },
        ]
    }
    source_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = InputConfig(
        type="aerialvln_json",
        adapter_class_path="NavVLAeval.aerialvln.inputs:AerialVLNJsonInputAdapter",
        namespace="aerialvln_val_seen",
        path=source_path,
        raw={"episode_ids": ["keep"]},
    )

    episodes = AerialVLNJsonInputAdapter().load_episodes(cfg, max_samples=1)

    assert [episode.source_episode_id for episode in episodes] == ["keep"]


def test_aerialvln_json_input_combines_namespaced_roots_with_global_limit(tmp_path: Path) -> None:
    seen_path = tmp_path / "val_seen.json"
    unseen_path = tmp_path / "val_unseen.json"
    _write_aerialvln_json(seen_path)
    _write_aerialvln_json(unseen_path)
    cfg = InputConfig(
        type="aerialvln_json",
        adapter_class_path="NavVLAeval.aerialvln.inputs:AerialVLNJsonInputAdapter",
        roots=(
            InputRootConfig(namespace="aerialvln_val_seen", path=seen_path),
            InputRootConfig(namespace="aerialvln_val_unseen", path=unseen_path),
        ),
    )

    adapter = AerialVLNJsonInputAdapter()
    episodes = adapter.load_episodes(cfg, max_samples=2)

    assert [episode.episode_uid for episode in episodes] == [
        "aerialvln_val_seen:101",
        "aerialvln_val_unseen:101",
    ]
    original_fingerprint = adapter.fingerprint(cfg)
    unseen_path.write_text(unseen_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert adapter.fingerprint(cfg) != original_fingerprint


def test_aerialvln_action_observations_fill_dense_history_with_waypoint_timestamps() -> None:
    adapter = AerialVLNRuntimeDatasetAdapter(
        history_selection="recent",
        history_image_frames=191,
        history_update_mode="action_observations",
        action_horizon=8,
    )
    prediction = ActionPrediction(
        normalized_actions=np.zeros((8, 4), dtype=np.float32),
        raw_actions=np.zeros((8, 4), dtype=np.float32),
    )

    def cached_observation(frame_index: int) -> dict:
        observation = adapter.prepare_observation_for_step(
            observation={"image": np.zeros((4, 4, 3), dtype=np.uint8)},
            step=frame_index,
        )
        observation["navvla_online_visual_tokens"] = {"front": np.zeros((4, 8), dtype=np.float16)}
        return observation

    history = EpisodeHistory()
    pre_observation = cached_observation(0)
    action_observations = [cached_observation(frame_index) for frame_index in range(1, 9)]
    update_observations = adapter.history_observations_for_update(
        pre_observation=pre_observation,
        post_observation=action_observations[-1],
        step_result=EnvironmentStepResult(
            next_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
            observation=action_observations[-1],
            data_done=False,
        ),
        action_observations=action_observations,
    )
    for observation in update_observations:
        history = adapter.update_history(
            history=history,
            observation=observation,
            prediction=prediction,
            instruction="go",
        )

    current_observation = adapter.prepare_observation_for_step(
        observation={"image": np.zeros((4, 4, 3), dtype=np.uint8)},
        step=8,
    )
    sample = adapter.build_example(observation=current_observation, history=history, instruction="go")

    assert [item["frame_index"] for item in sample["metadata"]["history_steps"]] == list(range(8))
    assert [item["timestamp"] for item in sample["metadata"]["history_steps"]] == [float(index) for index in range(8)]
    assert sample["metadata"]["timestamp"] == 8.0
    assert sample["history_cached_embeds"].shape == (8, 4, 8)


def test_aerialvln_prompt_prefix_is_configurable() -> None:
    with_prefix = get_runtime_dataset_adapter(
        {
            "runtime_adapter": "aerialvln",
            "include_aerialvln_prompt_prefix": True,
        }
    )
    without_prefix = get_runtime_dataset_adapter(
        {
            "runtime_adapter": "aerialvln",
            "include_aerialvln_prompt_prefix": False,
        }
    )

    assert with_prefix.platform_text == AERIALVLN_PLATFORM_TEXT
    assert with_prefix.platform_text.startswith("this is aerialvln. ")
    assert without_prefix.platform_text == AERIALVLN_PLATFORM_TEXT.removeprefix("this is aerialvln. ")


def test_traveluav_json_filters_scenes_before_max_samples(tmp_path: Path) -> None:
    source_path = tmp_path / "travel.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "episode_id": "town01",
                    "scene_id": "Carla_Town01",
                    "env_name": "Carla_Town01",
                    "instruction": "find target one",
                },
                {
                    "episode_id": "town04",
                    "scene_id": "Carla_Town04",
                    "env_name": "Carla_Town04",
                    "instruction": "find target four",
                },
            ]
        ),
        encoding="utf-8",
    )
    cfg = InputConfig(
        type="eval_json",
        adapter_class_path="NavVLAeval.traveluav.inputs:TravelUAVJsonInputAdapter",
        namespace="seen",
        path=source_path,
        raw={"scene_ids": ["Carla_Town04"]},
    )
    other_scene_cfg = InputConfig(
        type=cfg.type,
        adapter_class_path=cfg.adapter_class_path,
        namespace=cfg.namespace,
        path=cfg.path,
        raw={"scene_ids": ["Carla_Town01"]},
    )
    adapter = TravelUAVJsonInputAdapter()

    episodes = adapter.load_episodes(cfg, max_samples=1)

    assert [episode.source_episode_id for episode in episodes] == ["town04"]
    assert adapter.fingerprint(cfg) != adapter.fingerprint(other_scene_cfg)


def test_traveluav_lerobot_filters_scenes_before_max_samples(tmp_path: Path) -> None:
    root = _write_traveluav_lerobot_root(tmp_path)
    cfg = InputConfig(
        type="navvla_lerobot_v3",
        adapter_class_path="NavVLAeval.traveluav.inputs:TravelUAVLeRobotV3InputAdapter",
        roots=(InputRootConfig(namespace="vln_val_seen", path=root),),
        raw={"scene_ids": ["Carla_Town04"]},
    )
    other_scene_cfg = InputConfig(
        type=cfg.type,
        adapter_class_path=cfg.adapter_class_path,
        roots=cfg.roots,
        raw={"scene_ids": ["Carla_Town01"]},
    )
    adapter = TravelUAVLeRobotV3InputAdapter()

    episodes = adapter.load_episodes(cfg, max_samples=1)

    assert [episode.source_episode_id for episode in episodes] == ["town04"]
    assert episodes[0].payload["start_pose"] == [40.0, 50.0, -7.0, -0.5]
    assert episodes[0].payload["trajectory_raw_detailed"] == [[40.0, 50.0, -7.0, -0.5]]
    assert adapter.fingerprint(cfg) != adapter.fingerprint(other_scene_cfg)


def test_traveluav_lerobot_rejects_relative_state_without_absolute_source_pose(tmp_path: Path) -> None:
    root = _write_traveluav_lerobot_root(tmp_path)
    metadata_path = root / "meta" / "navvla_frame_metadata.jsonl"
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    records[1]["source_metadata"].pop("source_state")
    metadata_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    cfg = InputConfig(
        type="navvla_lerobot_v3",
        adapter_class_path="NavVLAeval.traveluav.inputs:TravelUAVLeRobotV3InputAdapter",
        roots=(InputRootConfig(namespace="vln_val_seen", path=root),),
        raw={"scene_ids": ["Carla_Town04"]},
    )

    with pytest.raises(ValueError, match="absolute source_state"):
        TravelUAVLeRobotV3InputAdapter().load_episodes(cfg, max_samples=1)


def test_aerialvln_settings_profile_and_layout(tmp_path: Path) -> None:
    env_dir = tmp_path / "env" / "env_26" / "LinuxNoEditor"
    env_dir.mkdir(parents=True)
    script = env_dir / "AirVLN.sh"
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    assert resolve_airsim_start_script(tmp_path / "env", "26", layout="aerialvln") == script
    settings = build_airsim_settings(api_server_port=41451, profile="aerialvln")

    assert settings["SimMode"] == "Multirotor"
    vehicle = settings["Vehicles"]["Drone_1"]
    assert vehicle["VehicleType"] == "SimpleFlight"
    front = vehicle["Cameras"]["front_0"]
    sizes_by_type = {item["ImageType"]: (item["Width"], item["Height"]) for item in front["CaptureSettings"]}
    assert sizes_by_type == {0: (448, 448), 2: (448, 448), 3: (448, 448)}

    command = build_airsim_launch_command(
        AirSimLaunchConfig(
            start_script=script,
            physical_gpu_id=0,
            settings_path=tmp_path / "settings.json",
            ue_args=[],
            settings_argument_style="space",
        )
    )
    assert "--settings" in command
    assert f"-settings={tmp_path / 'settings.json'}" not in command


def test_airsim_rpc_timeout_is_fixed_at_40_seconds() -> None:
    assert AIRSIM_RPC_TIMEOUT_SEC == 40


def test_openfly_worker_copy_detaches_binary_chmod_target_from_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    linux_root = source / "LinuxNoEditor"
    binary = linux_root / "scene" / "Binaries" / "Linux" / "scene"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    binary.chmod(0o644)
    start_script = linux_root / "start.sh"
    start_script.write_text(
        '#!/bin/sh\nchmod +x "$UE4_PROJECT_ROOT/scene/Binaries/Linux/scene"\n',
        encoding="utf-8",
    )
    shared_asset = linux_root / "scene" / "Content" / "asset.pak"
    shared_asset.parent.mkdir(parents=True)
    shared_asset.write_bytes(b"asset")
    worker = tmp_path / "worker"

    copytree_with_hardlinks(source, worker)

    worker_binary = worker / binary.relative_to(source)
    worker_asset = worker / shared_asset.relative_to(source)
    assert worker_binary.stat().st_ino != binary.stat().st_ino
    assert worker_asset.stat().st_ino == shared_asset.stat().st_ino
    worker_binary.chmod(0o755)
    assert binary.stat().st_mode & 0o777 == 0o644


def test_openfly_worker_copy_repairs_existing_hardlinked_binary(tmp_path: Path) -> None:
    source = tmp_path / "source"
    linux_root = source / "LinuxNoEditor"
    binary = linux_root / "scene" / "Binaries" / "Linux" / "scene"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")
    start_script = linux_root / "start.sh"
    start_script.write_text(
        '#!/bin/sh\nchmod +x "$UE4_PROJECT_ROOT/scene/Binaries/Linux/scene"\n',
        encoding="utf-8",
    )
    worker = tmp_path / "worker"
    worker_binary = worker / binary.relative_to(source)
    worker_binary.parent.mkdir(parents=True)
    worker_start_script = worker / start_script.relative_to(source)
    worker_start_script.parent.mkdir(parents=True, exist_ok=True)
    os.link(binary, worker_binary)
    os.link(start_script, worker_start_script)

    copytree_with_hardlinks(source, worker)

    assert worker_binary.stat().st_ino != binary.stat().st_ino


def test_airsim_timeout_retries_episode_once_and_invalidates_client(monkeypatch) -> None:
    episode = EvalEpisode(
        episode_uid="traveluav:episode-1",
        source_episode_id="episode-1",
        scene_id="Carla_Town04",
        instruction="Fly there.",
        source="traveluav",
        input_namespace="val_seen",
        input_root="/tmp/val_seen",
        payload={"env_name": "Carla_Town04"},
    )
    timeout_result = EpisodeResult(
        episode_uid=episode.episode_uid,
        source_episode_id=episode.source_episode_id,
        scene_id=episode.scene_id,
        instruction=episode.instruction,
        success=0,
        oracle_success=0,
        final_distance=0.0,
        path_length=0.0,
        gt_path_length=0.0,
        steps=1,
        failure="TimeoutError: request timed out",
        failure_type="benchmark_runtime",
        termination_reason="failure",
        failure_traceback="msgpackrpc.error.TimeoutError: request timed out",
    )
    success_result = EpisodeResult(
        episode_uid=episode.episode_uid,
        source_episode_id=episode.source_episode_id,
        scene_id=episode.scene_id,
        instruction=episode.instruction,
        success=1,
        oracle_success=1,
        final_distance=0.0,
        path_length=1.0,
        gt_path_length=1.0,
        steps=1,
        failure=None,
        failure_type=None,
        termination_reason="success",
    )
    results = iter([timeout_result, success_result])
    attempts = []

    def fake_run_episode(**kwargs):
        attempts.append(kwargs["episode"].episode_uid)
        return next(results)

    monkeypatch.setattr(worker_module, "_run_episode", fake_run_episode)
    backend = SimpleNamespace(close_calls=0)

    def close() -> None:
        backend.close_calls += 1

    backend.close = close
    worker = SimpleNamespace(backend=SimpleNamespace(type="airsim"))

    result = worker_module._run_episode_with_simulator_retry(
        cfg=None,
        worker=worker,
        store=None,
        episode=episode,
        runtime=None,
        model=None,
        env_backend=backend,
        action_codec=None,
        runtime_dataset=None,
        run_identity={},
    )

    assert attempts == [episode.episode_uid, episode.episode_uid]
    assert backend.close_calls == 1
    assert result is success_result


def test_airsim_timeout_invalidates_client_after_retry_is_exhausted(monkeypatch) -> None:
    timeout_result = EpisodeResult(
        episode_uid="traveluav:episode-1",
        source_episode_id="episode-1",
        scene_id="Carla_Town04",
        instruction="Fly there.",
        success=0,
        oracle_success=0,
        final_distance=0.0,
        path_length=0.0,
        gt_path_length=0.0,
        steps=1,
        failure="TimeoutError: request timed out",
        failure_type="benchmark_runtime",
        termination_reason="failure",
        failure_traceback="msgpackrpc.error.TimeoutError: request timed out",
    )
    attempts = []

    def fake_run_episode(**kwargs):
        attempts.append(kwargs["episode"].episode_uid)
        return timeout_result

    monkeypatch.setattr(worker_module, "_run_episode", fake_run_episode)
    backend = SimpleNamespace(close_calls=0)

    def close() -> None:
        backend.close_calls += 1

    backend.close = close
    worker = SimpleNamespace(backend=SimpleNamespace(type="airsim"))
    episode = EvalEpisode(
        episode_uid=timeout_result.episode_uid,
        source_episode_id=timeout_result.source_episode_id,
        scene_id=timeout_result.scene_id,
        instruction=timeout_result.instruction,
        source="traveluav",
        input_namespace="val_seen",
        input_root="/tmp/val_seen",
        payload={"env_name": "Carla_Town04"},
    )

    result = worker_module._run_episode_with_simulator_retry(
        cfg=None,
        worker=worker,
        store=None,
        episode=episode,
        runtime=None,
        model=None,
        env_backend=backend,
        action_codec=None,
        runtime_dataset=None,
        run_identity={},
    )

    assert attempts == [episode.episode_uid, episode.episode_uid]
    assert backend.close_calls == 2
    assert result is timeout_result


def test_eval_info_payload_starts_with_episode_uid_and_metrics() -> None:
    episode = EvalEpisode(
        episode_uid="traveluav:episode-1",
        source_episode_id="episode-1",
        scene_id="Carla_Town04",
        instruction="Fly there.",
        source="traveluav",
        input_namespace="val_seen",
        input_root="/tmp/val_seen",
        payload={"env_name": "Carla_Town04"},
    )
    result = EpisodeResult(
        episode_uid=episode.episode_uid,
        source_episode_id=episode.source_episode_id,
        scene_id=episode.scene_id,
        instruction=episode.instruction,
        success=1,
        oracle_success=1,
        final_distance=0.0,
        path_length=1.0,
        gt_path_length=1.0,
        steps=1,
        failure=None,
        failure_type=None,
        termination_reason="success",
    )
    cfg = SimpleNamespace(
        benchmark=SimpleNamespace(name="traveluav"),
        output=SimpleNamespace(run_name="test", metrics=("SR",)),
    )
    worker = SimpleNamespace(
        backend=SimpleNamespace(type="airsim", kwargs={"airsim_port": 41451}),
        episode_attempts={episode.episode_uid: "attempt-1"},
        worker_index=0,
        physical_gpu_id=5,
    )

    payload = worker_module._eval_info_payload(
        cfg,
        worker,
        episode,
        result,
        {"config_sha256": "config", "input_fingerprint": "input"},
    )

    assert list(payload)[:2] == ["episode_uid", "metrics"]


def test_r2r_config_uses_validated_direct_habitat_longmem_contract() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path("NavVLAeval/vlnce/r2r/config_portable.yaml"))

    assert cfg.benchmark.max_steps == 200
    assert cfg.benchmark.max_samples == 50
    assert cfg.benchmark.kwargs["stop_on_success_radius"] is False
    assert cfg.input.type == "vlnce_r2r"
    assert cfg.input.adapter_class_path == "NavVLAeval.vlnce.r2r.inputs:R2RInputAdapter"
    assert cfg.model.checkpoint == "../../../local/checkpoints/r2r/pytorch_model.pt"
    assert cfg.model.repo_root == "../../.."
    assert cfg.model.unnorm_key == "vln_train_train"
    assert cfg.dataset.action_type == "anchor_relative_body_frame_xyz_yaw"
    assert cfg.env.backend_class_path == "NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackend"
    assert cfg.env.planner_class_path == "NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackendPlanner"
    assert cfg.env.kwargs["continuous_control_mode"] == "collision_slide_pose_delta"
    assert cfg.env.kwargs["execute_waypoints_per_step"] == 8
    assert cfg.env.kwargs["habitat_lab_root"].endswith("local/simulators/VLN-CE/Evt-bench/habitat-lab")
    assert cfg.env.kwargs["habitat_sim_site_packages"].endswith(
        "local/simulators/VLN-CE/build_py310_habitat_sim_031/lib/python3.10/site-packages"
    )
    assert cfg.env.kwargs["capture_action_observations"] is True
    assert cfg.env.kwargs["action_adapter_kwargs"]["lateral_sign"] == -1.0
    assert cfg.env.kwargs["action_adapter_kwargs"]["yaw_sign"] == -1.0


def test_rxr_config_uses_same_habitat031_backend() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path("NavVLAeval/vlnce/rxr/config_portable.yaml"))

    assert cfg.env.type == "habitat"
    assert cfg.env.backend_class_path == "NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackend"
    assert cfg.env.planner_class_path == "NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackendPlanner"
    assert cfg.env.kwargs["task_name"] == "rxr"
    assert cfg.env.kwargs["roles"] == ["guide"]
    assert cfg.env.kwargs["languages"] == ["en-US", "en-IN"]
    assert cfg.env.kwargs["continuous_control_mode"] == "filtered_pose_delta"
    assert cfg.env.kwargs["action_adapter_kwargs"]["yaw_sign"] == 1.0


def test_vlnce_platform_text_matches_longmem_training_metadata() -> None:
    assert VLNCE_PLATFORM_TEXT == (
        "The platform is Indoor Robot for indoor robot navigation. The control frequency is 1 Hz. "
        "Please predict the next 8 local 3D waypoints (dx, dy, dz, dyaw) to execute the following task:"
    )


def test_continuous_habitat_adapter_reanchors_chunk_to_current_pose() -> None:
    from NavVLAeval.common.simulators.habitat.action_adapter import BodyFrameContinuousActionAdapter

    adapter = BodyFrameContinuousActionAdapter(min_delta=0.001, lateral_sign=-1.0, yaw_sign=1.0)
    decision = adapter.to_server_action(
        np.asarray([[0.2, 0.0, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0]], dtype=np.float32),
        action_index=1,
        anchor_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        current_pose=Pose4D(0.0, 0.2, 0.0, 0.0),
    )

    assert decision["server_payload"] == {
        "action": "MOVE_BY_POSE_DELTA",
        "action_args": {"dx": 0.0, "dy": 0.1, "dyaw": 0.0},
    }
    assert decision["log"]["delta_action"] == [0.1, 0.0, 0.0, 0.0]


def test_vlnce031_backend_executes_entire_received_action_chunk(tmp_path: Path) -> None:
    from NavVLAeval.common.simulators.habitat.backend import (
        VLNCE031HabitatBackend,
        VLNCE031HabitatBackendPlanner,
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.step_payloads = []

        def reset(self, payload: dict) -> dict:
            return _vlnce_habitat_payload(done=False, metrics={"distance_to_goal": 4.0})

        def step(self, payload: dict) -> dict:
            self.step_payloads.append(dict(payload))
            return _vlnce_habitat_payload(done=False, metrics={"distance_to_goal": 3.5})

        def close(self) -> None:
            return None

    runtime = FakeRuntime()
    cfg = EnvConfig(
        type="habitat",
        backend_class_path="NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackend",
        planner_class_path="NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackendPlanner",
        kwargs={
            "data_root": str(tmp_path),
            "split": "val_seen",
            "action_adapter_class_path": "NavVLAeval.common.simulators.habitat.action_adapter:BodyFrameContinuousActionAdapter",
            "action_adapter_kwargs": {"lateral_sign": -1.0, "yaw_sign": 1.0, "min_delta": 0.001},
            "capture_action_observations": True,
            "execute_waypoints_per_step": 1,
        },
    )
    plan = VLNCE031HabitatBackendPlanner().plan_worker_backend(
        cfg=cfg,
        store=object(),
        worker_index=0,
        physical_gpu_id=0,
    )
    backend = VLNCE031HabitatBackend(
        cfg=cfg,
        worker_backend=plan,
        physical_gpu_id=0,
        runtime_factory=lambda **_: runtime,
    )
    episode = EvalEpisode(
        episode_uid="r2r_val_seen:7",
        source_episode_id="7",
        scene_id="scene",
        instruction="go forward",
        source="vlnce_r2r",
        input_namespace="r2r_val_seen",
        input_root=str(tmp_path),
        payload={"episode_id": "7"},
    )

    backend.start_episode(episode, Pose4D(0.0, 0.0, 0.0, 0.0))
    result = backend.apply_action(
        Pose4D(0.0, 0.0, 0.0, 0.0),
        np.asarray([[0.2, 0.1, 0.0, 0.3], [0.5, 0.2, 0.0, 0.6]], dtype=np.float32),
    )

    assert plan == WorkerBackendPlan(type="habitat", kwargs={})
    assert runtime.step_payloads == [
        {"action": "MOVE_BY_POSE_DELTA", "action_args": {"dx": -0.1, "dy": 0.2, "dyaw": 0.3}},
        {"action": "MOVE_BY_POSE_DELTA", "action_args": {"dx": -0.2, "dy": 0.5, "dyaw": 0.6}},
    ]
    assert result.diagnostics["executed_model_waypoint_count"] == 2
    assert result.diagnostics["actual_waypoint_poses"] == [[0.0, 0.0, 0.0, 0.0]] * 2
    assert result.diagnostics["waypoint_control"][0]["target_pose"] == [-0.10000000149011612, 0.20000000298023224, 0.0, 0.30000001192092896]
    assert result.diagnostics["waypoint_control"][0]["actual_pose"] == [0.0, 0.0, 0.0, 0.0]
    assert result.diagnostics["waypoint_control"][0]["position_error_m"] > 0.0
    assert result.diagnostics["capture_action_observations"] is True
    assert len(result.action_observations) == 2

    from NavVLAeval.common.data.runtime_dataset import OnlineNavVLARuntimeDatasetAdapter

    history_adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_update_mode="action_observations",
    )
    history_candidates = history_adapter.history_observations_for_update(
        pre_observation={},
        post_observation=result.observation,
        step_result=result,
        action_observations=result.action_observations,
    )
    assert len(history_candidates) == 2
    assert history_candidates[0] is result.action_observations[0]
    assert history_candidates[1] is result.action_observations[1]

    no_capture_runtime = FakeRuntime()
    no_capture_cfg = EnvConfig(
        type="habitat",
        backend_class_path="NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackend",
        planner_class_path="NavVLAeval.common.simulators.habitat.backend:VLNCE031HabitatBackendPlanner",
        kwargs={**cfg.kwargs, "capture_action_observations": False},
    )
    no_capture_backend = VLNCE031HabitatBackend(
        cfg=no_capture_cfg,
        worker_backend=plan,
        physical_gpu_id=0,
        runtime_factory=lambda **_: no_capture_runtime,
    )
    no_capture_backend.start_episode(episode, Pose4D(0.0, 0.0, 0.0, 0.0))
    no_capture_result = no_capture_backend.apply_action(
        Pose4D(0.0, 0.0, 0.0, 0.0),
        np.asarray([[0.2, 0.1, 0.0, 0.3], [0.5, 0.2, 0.0, 0.6]], dtype=np.float32),
    )
    assert no_capture_result.diagnostics["capture_action_observations"] is False
    assert no_capture_result.diagnostics["executed_waypoint_count"] == 2
    assert no_capture_result.action_observations == []


def test_habitat_worker_process_receives_verified_egl_environment(tmp_path: Path, monkeypatch) -> None:
    from NavVLAeval.common.runner import parallel_runner
    from NavVLAeval.common.runner.parallel_runner import build_worker_subprocess_command

    monkeypatch.setattr(parallel_runner.shutil, "which", lambda _name: None)
    project_python = tmp_path / ".venv" / "bin" / "python"
    project_python.parent.mkdir(parents=True)
    project_python.write_text("", encoding="utf-8")

    egl_root = tmp_path / "nvidia-egl"
    lib_dir = egl_root / "lib"
    vendor_json = egl_root / "egl_vendor.d" / "10_nvidia.json"
    lib_dir.mkdir(parents=True)
    vendor_json.parent.mkdir(parents=True)
    vendor_json.write_text("{}", encoding="utf-8")

    env, _command = build_worker_subprocess_command(
        worker_plan_path=tmp_path / "worker.json",
        physical_gpu_id=1,
        repo_root=tmp_path,
        env_kwargs={"nvidia_egl_root": str(egl_root)},
    )

    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(lib_dir)
    assert env["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(vendor_json)
    assert _command[:4] == [str(project_python), "-u", "-m", "NavVLAeval.common.runner.worker"]


def test_habitat_initializes_environment_before_cuda_model(monkeypatch) -> None:
    events = []
    environment = SimpleNamespace(close=lambda: events.append("close_environment"))
    worker = SimpleNamespace(backend=SimpleNamespace(type="habitat"), physical_gpu_id=1)
    cfg = SimpleNamespace(env=SimpleNamespace(type="habitat"))

    monkeypatch.setattr(
        worker_module,
        "create_environment_backend",
        lambda **_: events.append("create_environment") or environment,
    )
    monkeypatch.setattr(worker_module, "build_model", lambda _cfg: events.append("build_model") or object())

    _model, actual_environment = worker_module._build_model_and_environment(cfg, worker)

    assert actual_environment is environment
    assert events == ["create_environment", "build_model"]


def test_legacy_habitat_rpc_stack_and_duplicate_builders_are_removed() -> None:
    from NavVLAeval.common.env import backends as env_backends
    from NavVLAeval.common.simulators.habitat import action_adapter, vlnce031_runtime

    habitat_root = Path("NavVLAeval/common/simulators/habitat")
    obsolete_files = [
        habitat_root / "client.py",
        habitat_root / "common" / "vlnce_server.py",
        habitat_root / "r2r" / "habitat_server.py",
        habitat_root / "rxr" / "habitat_server.py",
    ]

    assert not any(path.exists() for path in obsolete_files)
    assert not any((habitat_root / "habitat_extensions").rglob("*.py"))
    assert not hasattr(env_backends, "_worker_backend_env_type")
    assert not hasattr(action_adapter, "BodyFrameDiscreteTurnForwardActionAdapter")
    assert not hasattr(vlnce031_runtime, "camera_sensor_configs")
    assert not hasattr(vlnce031_runtime, "action_configs")
    assert not hasattr(vlnce031_runtime, "measurement_configs")


def test_vlnce_success_uses_euclidean_radius_without_early_stop() -> None:
    episode = EvalEpisode(
        episode_uid="r2r_val_seen:7",
        source_episode_id="7",
        scene_id="scene",
        instruction="go forward",
        source="vlnce_r2r",
        input_namespace="r2r_val_seen",
        input_root="/tmp/vlnce",
        payload={"gt_path_length": 4.0},
    )
    runtime = VLNCERuntime(task_name="r2r", success_distance=3.0, stop_on_success_radius=False)
    runtime.prepare_environment(episode, None, Pose4D(0.0, 0.0, 0.0, 0.0))
    pose = Pose4D(0.0, 0.0, 0.0, 0.0)
    state = StepState(
        episode=episode,
        step_index=0,
        artifact_step_index=0,
        instruction=episode.instruction,
        history=EpisodeHistory(),
        pre_observation={},
        post_observation={"done": False},
        pose_before=pose,
        pose_after=pose,
        prediction=ActionPrediction(
            normalized_actions=np.zeros((1, 4), dtype=np.float32),
            raw_actions=np.zeros((1, 4), dtype=np.float32),
        ),
        raw_action_chunk=np.zeros((1, 4), dtype=np.float32),
        world_waypoints=np.zeros((1, 4), dtype=np.float32),
        executed_action_count=1,
        distance_before=9.0,
        distance_after=9.0,
        path_length=6.0,
        diagnostics={
            "metrics": {
                "distance_to_goal": 9.0,
                "euclidean_distance_to_goal": 2.9,
                "success": 0,
                "oracle_success": 0,
                "path_length": 6.0,
            }
        },
    )

    termination = runtime.update_termination(state)

    assert termination.done is False
    assert termination.success == 1
    assert termination.oracle_success == 1
    assert runtime.is_success(pose, episode) is True
    assert termination.diagnostics["metrics"]["spl"] == 4.0 / 6.0


def _vlnce_habitat_payload(*, done: bool, metrics: dict) -> dict:
    return {
        "observation": {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb_left": np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb_right": np.zeros((2, 2, 3), dtype=np.uint8),
            "rgb_rear": np.zeros((2, 2, 3), dtype=np.uint8),
        },
        "pose": [0.0, 0.0, 0.0, -math.pi / 2.0],
        "instruction": "go forward",
        "episode_id": "7",
        "scene_id": "scene",
        "metrics": dict(metrics),
        "done": bool(done),
    }


def test_aerialvln_benchmark_uses_official_3d_goal_distance_and_success_radius() -> None:
    episode = EvalEpisode(
        episode_uid="aerialvln:101",
        source_episode_id="101",
        scene_id="26",
        instruction="Fly there.",
        source="aerialvln_json",
        input_namespace="aerialvln",
        input_root="/tmp/aerialvln.json",
        payload={
            "env_name": "env_26",
            "start_pose": [0.0, 0.0, -50.0, 0.0],
            "goal_position": [3.0, 4.0, -10.0],
            "reference_path_m": [[0.0, 0.0, -50.0, 0.0], [3.0, 4.0, -10.0, 0.0]],
        },
    )
    runtime = AerialVLNBenchmarkSpec(success_radius=45.0).create_runtime(None)

    assert np.isclose(runtime.distance_to_goal(Pose4D(0.0, 0.0, -50.0, 0.0), episode), np.sqrt(1625.0))
    assert runtime.is_success(Pose4D(0.0, 0.0, -50.0, 0.0), episode)

    state = StepState(
        episode=episode,
        step_index=0,
        artifact_step_index=0,
        instruction=episode.instruction,
        history=EpisodeHistory(),
        pre_observation={},
        post_observation={},
        pose_before=Pose4D(0.0, 0.0, -50.0, 0.0),
        pose_after=Pose4D(30.0, 40.0, -50.0, 0.0),
        prediction=ActionPrediction(
            normalized_actions=np.zeros((1, 4), dtype=np.float32),
            raw_actions=np.zeros((1, 4), dtype=np.float32),
        ),
        raw_action_chunk=np.ones((1, 4), dtype=np.float32),
        world_waypoints=np.zeros((1, 4), dtype=np.float32),
        executed_action_count=1,
        distance_before=50.0,
        distance_after=45.0,
        path_length=5.0,
    )

    termination = runtime.update_termination(state)
    assert termination.done is False
    assert termination.success == 0
    assert termination.reason == "running"


def test_aerialvln_max_steps_mode_records_success_without_early_stop() -> None:
    episode = EvalEpisode(
        episode_uid="aerialvln:101",
        source_episode_id="101",
        scene_id="26",
        instruction="Fly there.",
        source="aerialvln_json",
        input_namespace="aerialvln",
        input_root="/tmp/aerialvln.json",
        payload={
            "env_name": "env_26",
            "start_pose": [0.0, 0.0, 0.0, 0.0],
            "goal_position": [10.0, 0.0, 0.0],
            "reference_path_m": [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]],
        },
    )
    runtime = AerialVLNBenchmarkSpec(
        success_radius=20.0,
        termination_mode="max_steps",
    ).create_runtime(None)
    state = SimpleNamespace(episode=episode, pose_after=Pose4D(10.0, 0.0, 0.0, 0.0), distance_after=0.0)

    termination = runtime.update_termination(state)

    assert runtime.stop_at_first_success_waypoint() is False
    assert termination.done is False
    assert termination.success == 1
    assert termination.oracle_success == 1
    assert termination.reason == "running"


def test_aerialvln_action_stop_uses_final_segment_and_confirmations() -> None:
    episode = EvalEpisode(
        episode_uid="aerialvln:101",
        source_episode_id="101",
        scene_id="26",
        instruction="Fly there.",
        source="aerialvln_json",
        input_namespace="aerialvln",
        input_root="/tmp/aerialvln.json",
        payload={
            "env_name": "env_26",
            "start_pose": [0.0, 0.0, 0.0, 0.0],
            "goal_position": [100.0, 0.0, 0.0],
            "reference_path_m": [[0.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]],
        },
    )
    runtime = AerialVLNBenchmarkSpec(
        success_radius=20.0,
        termination_mode="action_or_max_steps",
        stop_action_threshold=0.292,
        stop_action_measure="final_segment_xyz_norm",
        stop_action_confirmations=2,
    ).create_runtime(None)
    action = np.zeros((8, 4), dtype=np.float32)
    action[-2, :3] = [10.0, 0.0, 0.0]
    action[-1, :3] = [10.2, 0.0, 0.0]
    state = SimpleNamespace(
        episode=episode,
        pose_after=Pose4D(0.0, 0.0, 0.0, 0.0),
        raw_action_chunk=action,
        distance_after=100.0,
    )

    first = runtime.update_termination(state)
    second = runtime.update_termination(state)

    assert runtime.stop_at_first_success_waypoint() is False
    assert first.done is False
    assert first.diagnostics["stop_action_streak"] == 1
    assert np.isclose(first.diagnostics["stop_action_value"], 0.2)
    assert second.done is True
    assert second.success == 0
    assert second.reason == "stop"
    assert second.diagnostics["stop_action_streak"] == 2
    artifacts = runtime.log_step_artifacts(state, None)
    assert np.isclose(artifacts["stop_action_values"]["final_segment_xyz_norm"], 0.2)

    reset_calls = []
    runtime.prepare_environment(
        episode,
        SimpleNamespace(reset_pose=reset_calls.append),
        Pose4D(0.0, 0.0, 0.0, 0.0),
    )
    assert len(reset_calls) == 1
    assert runtime.update_termination(state).diagnostics["stop_action_streak"] == 1


def test_aerialvln_rejects_unknown_termination_mode() -> None:
    with pytest.raises(ValueError, match="termination_mode"):
        AerialVLNBenchmarkSpec(termination_mode="unknown")
    with pytest.raises(ValueError, match="stop_action_measure"):
        AerialVLNBenchmarkSpec(stop_action_measure="unknown")
    with pytest.raises(ValueError, match="stop_action_confirmations"):
        AerialVLNBenchmarkSpec(stop_action_confirmations=0)


def test_aerialvln_runtime_dataset_state_is_opt_in() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    history = EpisodeHistory(
        images=[np.full((4, 4, 3), 64, dtype=np.uint8)],
        observations=[
            {
                "image": np.full((4, 4, 3), 64, dtype=np.uint8),
                "navvla_eval": {"episode_id": "101", "frame_index": 0, "timestamp": 0.0},
            }
        ],
        raw_actions=[np.asarray([1.0, 2.0, 3.0, 0.5], dtype=np.float32)],
    )

    default_adapter = AerialVLNRuntimeDatasetAdapter(history_image_frames=1, action_horizon=2)
    default_sample = default_adapter.build_example(
        observation={"image": image, "navvla_eval": {"episode_id": "101", "frame_index": 1}},
        history=history,
        instruction="Fly there.",
    )
    assert "state" not in default_sample
    assert sorted(default_sample["images"]) == ["front"]
    assert len(default_sample["history_images"]["front"]) == 1

    state_adapter = AerialVLNRuntimeDatasetAdapter(history_image_frames=1, action_horizon=2, include_state=True)
    state_sample = state_adapter.build_example(
        observation={"image": image, "navvla_eval": {"episode_id": "101", "frame_index": 1}},
        history=history,
        instruction="Fly there.",
    )

    assert state_sample["state"].shape == (4,)
    assert state_sample["state"].dtype == np.float32
