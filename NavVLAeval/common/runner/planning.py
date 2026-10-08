from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from omegaconf import OmegaConf

from NavVLAeval.common.config import EvalConfig, _build_typed_config, load_class
from NavVLAeval.common.data.inputs import load_eval_episodes
from NavVLAeval.common.log.artifacts import (
    ArtifactStore,
    RunLock,
    acquire_run_lock,
    scan_eval_infos,
    validate_sanitized_episode_paths,
    write_json_atomic,
)
from NavVLAeval.common.runner.backend_plan import planner_from_config
from NavVLAeval.common.types import EvalEpisode, RunPlan, WorkerPlan


@dataclass(frozen=True)
class PlannedRun:
    cfg: EvalConfig
    run_plan: RunPlan
    worker_plans: list[WorkerPlan]
    episodes: list[EvalEpisode]
    skipped_episode_uids: set[str]
    pending_episodes: list[EvalEpisode]
    run_root: Path
    lock: RunLock | None = None


def partition_contiguous(episodes, worker_count):
    base, extra = divmod(len(episodes), worker_count)
    chunks, start = [], 0
    for index in range(worker_count):
        end = start + base + (index < extra)
        chunks.append(episodes[start:end])
        start = end
    return chunks


def build_run_plan(cfg: EvalConfig, *, dry_run: bool) -> PlannedRun:
    store = ArtifactStore(cfg.output.root / cfg.output.run_name)
    lock = None if dry_run else acquire_run_lock(store.run_root)
    try:
        manifest_path = store.run_root / "episodes.json"
        if store.config_path.exists():
            # Resume the saved experiment, never reinterpret new research parameters.
            saved = OmegaConf.to_container(OmegaConf.load(store.config_path), resolve=True)
            saved["parallel"] = cfg.raw["parallel"]
            saved["model"]["uri"] = cfg.model.uri
            saved["model"]["request_timeout_sec"] = cfg.model.request_timeout_sec
            saved["driver_environment"] = cfg.driver_environment
            cfg = _build_typed_config(saved, base_dir=store.run_root)
            episodes = [EvalEpisode(**item) for item in json.loads(manifest_path.read_text())]
        else:
            episodes = load_eval_episodes(cfg.input, max_samples=cfg.input.max_samples or cfg.benchmark.max_samples)
            validate_sanitized_episode_paths(episodes)
            benchmark = load_class(cfg.benchmark.class_path)(**cfg.benchmark.kwargs)
            for episode in episodes:
                benchmark.validate_episode(episode)
            if not dry_run:
                write_json_atomic(manifest_path, [asdict(episode) for episode in episodes])
        uids = {episode.episode_uid for episode in episodes}
        skipped = {
            record.payload["episode_uid"]
            for record in scan_eval_infos(store.run_root)
            if record.valid
            and record.payload["episode_uid"] in uids
            and record.payload["status"] == "completed"
            and record.payload["failure"] is None
        }
        pending = [episode for episode in episodes if episode.episode_uid not in skipped]
        ordinals = {episode.episode_uid: index for index, episode in enumerate(episodes)}
        planner = planner_from_config(cfg)
        workers = []
        for index, (gpu, chunk) in enumerate(
            zip(cfg.parallel.gpu_ids, partition_contiguous(pending, len(cfg.parallel.gpu_ids)))
        ):
            workers.append(
                WorkerPlan(
                    index,
                    gpu,
                    chunk,
                    store.run_root,
                    store.worker_log_path(index),
                    planner.plan_worker_backend(cfg=cfg.env, store=store, worker_index=index, physical_gpu_id=gpu),
                    {episode.episode_uid: uuid.uuid4().hex for episode in chunk},
                    {episode.episode_uid: ordinals[episode.episode_uid] for episode in chunk},
                )
            )
        plan = RunPlan(
            2,
            cfg.benchmark.name,
            cfg.output.run_name,
            [episode.episode_uid for episode in episodes],
            [episode.episode_uid for episode in episodes if episode.episode_uid in skipped],
            [episode.episode_uid for episode in pending],
            [str(store.worker_plan_path(w.worker_index)) for w in workers],
        )
        if not dry_run:
            OmegaConf.save(OmegaConf.create(cfg.raw), store.config_path)
            write_json_atomic(store.run_plan_path, asdict(plan))
            for worker in workers:
                write_json_atomic(store.worker_plan_path(worker.worker_index), asdict(worker))
        return PlannedRun(cfg, plan, workers, episodes, skipped, pending, store.run_root, lock)
    except Exception:
        if lock is not None:
            lock.release()
        raise
