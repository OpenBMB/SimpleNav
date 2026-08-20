from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any

from NavVLAeval.common.config import EvalConfig
from NavVLAeval.common.log.artifacts import ArtifactStore, write_json_atomic
from NavVLAeval.common.log.metrics import summary_from_run_artifacts
from NavVLAeval.common.runner.planning import PlannedRun, build_run_plan
from NavVLAeval.common.types import WorkerPlan


def build_dry_run_summary(planned: PlannedRun) -> dict[str, Any]:
    scene_counts: dict[str, int] = {}
    for episode in planned.episodes:
        scene_counts[episode.scene_id] = scene_counts.get(episode.scene_id, 0) + 1
    return {
        "benchmark": planned.run_plan.benchmark,
        "run_name": planned.run_plan.run_name,
        "total_episodes": len(planned.run_plan.total_episode_uids),
        "skipped_episodes": len(planned.run_plan.skipped_episode_uids),
        "pending_episodes": len(planned.run_plan.pending_episode_uids),
        "scene_counts": scene_counts,
        "worker_count": len(planned.worker_plans),
        "workers": [
            {
                "worker_index": worker.worker_index,
                "physical_gpu_id": worker.physical_gpu_id,
                "item_count": len(worker.episodes),
                "episode_uids": [episode.episode_uid for episode in worker.episodes],
                "backend": _json_safe_backend(worker),
                "worker_log_path": str(worker.worker_log_path),
            }
            for worker in planned.worker_plans
        ],
    }


def run_eval_from_config(
    cfg: EvalConfig,
    *,
    dry_run: bool,
    repo_root: str | Path,
    worker_module: str = "NavVLAeval.common.runner.worker",
) -> dict[str, Any]:
    planned = build_run_plan(cfg, dry_run=dry_run)
    if dry_run:
        return build_dry_run_summary(planned)
    assert planned.lock is not None
    try:
        exit_codes = launch_worker_subprocesses(
            workers=planned.worker_plans,
            repo_root=repo_root,
            worker_module=worker_module,
            env_kwargs=cfg.env.kwargs,
        )
        summary = summary_from_run_artifacts(ArtifactStore(planned.run_root).run_plan_path, planned.run_root, metric_keys=cfg.output.metrics)
        write_json_atomic(planned.run_root / "summary.json", summary)
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"one or more workers failed: {exit_codes}")
        return summary
    finally:
        planned.lock.release()


def build_worker_subprocess_command(
    *,
    worker_plan_path: str | Path,
    physical_gpu_id: int,
    repo_root: str | Path,
    worker_module: str = "NavVLAeval.common.runner.worker",
    env_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    env["PYTHONUNBUFFERED"] = "1"
    _apply_worker_env_kwargs(env, env_kwargs or {})
    uv = shutil.which("uv")
    if uv is not None:
        command = [
            uv,
            "run",
            "--project",
            str(Path(repo_root)),
            "--no-sync",
            "python",
            "-u",
            "-m",
            str(worker_module),
            "--worker-plan",
            str(worker_plan_path),
        ]
    else:
        project_python = Path(repo_root) / ".venv" / "bin" / "python"
        python_executable = project_python if project_python.is_file() else Path(sys.executable)
        command = [
            str(python_executable),
            "-u",
            "-m",
            str(worker_module),
            "--worker-plan",
            str(worker_plan_path),
        ]
    return env, command


def launch_worker_subprocesses(
    *,
    workers: list[WorkerPlan],
    repo_root: str | Path,
    worker_module: str = "NavVLAeval.common.runner.worker",
    env_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    processes = []
    for worker in workers:
        worker_plan_path = worker.run_root / "worker_plans" / f"worker_{worker.worker_index}.json"
        env, command = build_worker_subprocess_command(
            worker_plan_path=worker_plan_path,
            physical_gpu_id=worker.physical_gpu_id,
            repo_root=repo_root,
            worker_module=worker_module,
            env_kwargs=env_kwargs,
        )
        worker.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = worker.worker_log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        thread = threading.Thread(target=_tee_process_output, args=(process, log_file), daemon=True)
        thread.start()
        processes.append((process, thread, log_file))
    exit_codes = []
    for process, thread, log_file in processes:
        exit_codes.append(process.wait())
        thread.join(timeout=5)
        log_file.close()
    return exit_codes


def _apply_worker_env_kwargs(env: dict[str, str], env_kwargs: dict[str, Any]) -> None:
    root_value = env_kwargs.get("nvidia_egl_root")
    if not root_value:
        return
    root = Path(str(root_value)).expanduser().resolve()
    lib_dir = Path(str(env_kwargs.get("nvidia_egl_lib_dir") or root / "lib")).expanduser().resolve()
    vendor_json = Path(
        str(env_kwargs.get("nvidia_egl_vendor_json") or root / "egl_vendor.d" / "10_nvidia.json")
    ).expanduser().resolve()
    if lib_dir.is_dir():
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(lib_dir))
    if vendor_json.is_file():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor_json)


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    values = [item for item in env.get(key, "").split(os.pathsep) if item]
    values = [item for item in values if item != value]
    env[key] = os.pathsep.join([value, *values])


def format_metric_summary_lines(summary: dict[str, Any], *, summary_path: str | Path | None = None) -> list[str]:
    metrics = dict(summary.get("metrics") or {})
    metric_text = " ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
    lines = [
        (
            "[eval-summary] "
            f"total_episodes={int(summary.get('total_episodes', 0))} "
            f"completed_episodes={int(summary.get('completed_episodes', 0))} "
            f"failed_episodes={int(summary.get('failed_episodes', 0))} "
            f"metric_episodes={int(summary.get('metric_episodes', 0))} "
            f"{metric_text}".rstrip()
        )
    ]
    for scene_id, scene_summary in sorted(dict(summary.get("scene_metrics", {}) or {}).items()):
        scene_metrics = dict(scene_summary.get("metrics") or {})
        scene_metric_text = " ".join(f"{key}={float(value):.4f}" for key, value in scene_metrics.items())
        lines.append(
            (
                "[scene-summary] "
                f"scene_id={scene_id} "
                f"total_episodes={int(scene_summary.get('total_episodes', 0))} "
                f"failed_episodes={int(scene_summary.get('failed_episodes', 0))} "
                f"metric_episodes={int(scene_summary.get('metric_episodes', 0))} "
                f"{scene_metric_text}"
            ).rstrip()
        )
    if summary_path is not None:
        lines.append(f"[eval-summary] summary_json={summary_path}")
    return lines


def print_metric_summary(summary: dict[str, Any], *, summary_path: str | Path | None = None) -> None:
    for line in format_metric_summary_lines(summary, summary_path=summary_path):
        print(line, flush=True)


def _tee_process_output(process: subprocess.Popen, log_file: Any) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        log_file.write(line)
        log_file.flush()
        print(line, end="", flush=True)


def _json_safe_backend(worker: WorkerPlan) -> dict[str, Any]:
    return worker.backend.to_jsonable()
