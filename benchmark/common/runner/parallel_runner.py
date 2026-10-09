from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from benchmark.common.config import EvalConfig, apply_driver_environment
from benchmark.common.log.artifacts import ArtifactStore, write_json_atomic
from benchmark.common.log.metrics import summary_from_run_artifacts
from benchmark.common.runner.planning import PlannedRun, build_run_plan
from benchmark.common.types import WorkerPlan


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
    worker_module: str = "benchmark.common.runner.worker",
) -> dict[str, Any]:
    if not dry_run:
        from tool.navvla.simulator_dependencies import require_simulator

        require_simulator(cfg.env.type)
    planned = build_run_plan(cfg, dry_run=dry_run)
    cfg = planned.cfg
    if dry_run:
        return build_dry_run_summary(planned)
    assert planned.lock is not None
    try:
        exit_codes = launch_worker_subprocesses(
            workers=planned.worker_plans,
            repo_root=repo_root,
            worker_module=worker_module,
            driver_environment=cfg.driver_environment,
            worker_timeout_sec=cfg.parallel.worker_timeout_sec,
        )
        summary = summary_from_run_artifacts(
            ArtifactStore(planned.run_root).run_plan_path, planned.run_root, metric_keys=cfg.output.metrics
        )
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
    worker_module: str = "benchmark.common.runner.worker",
    driver_environment: dict[str, str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    env["PYTHONUNBUFFERED"] = "1"
    apply_driver_environment(driver_environment or {}, env)
    env["UV_PROJECT_ENVIRONMENT"] = str(Path(repo_root) / ".venv")
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        str(Path(repo_root) / ".venv/bin/python"),
        "-u",
        "-m",
        str(worker_module),
        "--worker-plan",
        str(worker_plan_path),
    ]
    return env, command


def launch_worker_subprocesses(
    *, workers, repo_root, worker_module="benchmark.common.runner.worker", driver_environment=None, worker_timeout_sec=0
):
    processes = []
    try:
        for worker in workers:
            if not worker.episodes:
                continue
            env, command = build_worker_subprocess_command(
                worker_plan_path=worker.run_root / "worker_plans" / f"worker_{worker.worker_index}.json",
                physical_gpu_id=worker.physical_gpu_id,
                repo_root=repo_root,
                worker_module=worker_module,
                driver_environment=driver_environment,
            )
            worker.worker_log_path.parent.mkdir(parents=True, exist_ok=True)
            log = worker.worker_log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            except Exception:
                log.close()
                raise
            thread = threading.Thread(target=_tee_process_output, args=(process, log), daemon=True)
            processes.append((process, thread, log, time.monotonic()))
            thread.start()
        codes = []
        for process, thread, log, started in processes:
            timeout = max(0.01, worker_timeout_sec - (time.monotonic() - started)) if worker_timeout_sec > 0 else None
            codes.append(process.wait(timeout=timeout))
        return codes
    finally:
        for process, thread, log, _ in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            thread.join(timeout=5)
            log.close()


def format_metric_summary_lines(summary: dict[str, Any], *, summary_path: str | Path | None = None) -> list[str]:
    metrics = dict(summary.get("metrics") or {})
    metric_text = " ".join(
        (f"{key}={float(value):.4f}" if value is not None else f"{key}=unavailable") for key, value in metrics.items()
    )
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
        scene_metric_text = " ".join(
            (f"{key}={float(value):.4f}" if value is not None else f"{key}=unavailable")
            for key, value in scene_metrics.items()
        )
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
