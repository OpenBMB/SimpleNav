from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from NavVLAeval.common.config import load_eval_config
from NavVLAeval.common.env.backends import create_environment_backend
from NavVLAeval.common.data.inputs import load_eval_episodes
from NavVLAeval.common.runner.planning import build_run_plan
from NavVLAeval.common.runtime_components import build_benchmark_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start UnrealZoo for one UAV-Flow episode and execute one zero-action step.")
    parser.add_argument("--config", default="NavVLAeval/uavflow/config_portable.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output", default="/tmp/navvlaeval_uavflow_unrealzoo_smoke.json")
    parser.add_argument("--start-process", action="store_true", help="Actually launch UnrealZoo/Gym. Omit for constructor-only smoke.")
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_argparser().parse_args(argv)
    cfg = load_eval_config(args.config, overrides=[*args.override, "benchmark.max_samples=1"])
    runtime = build_benchmark_runtime(cfg)
    planned = build_run_plan(cfg, dry_run=True)
    if not planned.worker_plans or not planned.worker_plans[0].episodes:
        raise RuntimeError("dry-run produced no worker episode")
    worker = planned.worker_plans[0]
    episode = worker.episodes[0]
    initial_pose = runtime.initial_pose(episode)
    backend = create_environment_backend(
        cfg=cfg.env,
        worker_backend=worker.backend,
        physical_gpu_id=worker.physical_gpu_id,
        start_process=bool(args.start_process),
    )
    payload = {
        "episode_uid": episode.episode_uid,
        "scene_id": episode.scene_id,
        "initial_pose_m": initial_pose.as_array().tolist(),
        "backend": worker.backend.to_jsonable(),
        "started": bool(args.start_process),
    }
    if args.start_process:
        try:
            backend.start_episode(episode, initial_pose)
            observation = backend.get_observation()
            step = backend.apply_action(initial_pose, np.zeros((1, 4), dtype=np.float32))
            payload.update(
                {
                    "image_shape": list(np.asarray(observation["image"]).shape),
                    "next_pose_m": step.next_pose.as_array().tolist(),
                    "diagnostics": step.diagnostics,
                }
            )
        finally:
            backend.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
