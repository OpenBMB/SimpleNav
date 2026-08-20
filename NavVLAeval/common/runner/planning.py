from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import uuid

from omegaconf import OmegaConf

from NavVLAeval.common.log.artifacts import (
    ArtifactStore,
    RunIdentity,
    RunLock,
    acquire_run_lock,
    is_completed_skip_candidate,
    scan_eval_infos,
    validate_sanitized_episode_paths,
    write_json_atomic,
)
from NavVLAeval.common.config import EvalConfig, load_class
from NavVLAeval.common.data.inputs import compute_input_fingerprint, load_eval_episodes, load_input_adapter
from NavVLAeval.common.runner.backend_plan import planner_from_config
from NavVLAeval.common.types import (
    EvalEpisode,
    RunPlan,
    WorkerPlan,
)


@dataclass(frozen=True)
class PlannedRun:
    run_plan: RunPlan
    worker_plans: list[WorkerPlan]
    episodes: list[EvalEpisode]
    skipped_episode_uids: set[str]
    pending_episodes: list[EvalEpisode]
    run_root: Path
    lock: RunLock | None = None


def partition_contiguous(episodes: list[EvalEpisode], worker_count: int) -> list[list[EvalEpisode]]:
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}")
    base, extra = divmod(len(episodes), worker_count)
    chunks = []
    start = 0
    for index in range(worker_count):
        length = base + (1 if index < extra else 0)
        chunks.append(list(episodes[start : start + length]))
        start += length
    return chunks


def build_run_plan(cfg: EvalConfig, *, dry_run: bool) -> PlannedRun:
    run_root = cfg.output.root / cfg.output.run_name
    store = ArtifactStore(run_root)
    config_sha256 = config_identity_sha256(cfg)
    _validate_existing_config(store.config_path, config_sha256)

    lock = None
    if not dry_run:
        lock = acquire_run_lock(run_root)

    try:
        adapter = load_input_adapter(cfg.input)
        input_fingerprint = compute_input_fingerprint(adapter, cfg.input)
        episodes = load_eval_episodes(cfg.input, max_samples=cfg.input.max_samples or cfg.benchmark.max_samples)
        validate_sanitized_episode_paths(episodes)
        _validate_benchmark_episodes(cfg, episodes)
        identity = RunIdentity(
            benchmark=cfg.benchmark.name,
            run_name=cfg.output.run_name,
            config_sha256=config_sha256,
            input_fingerprint=input_fingerprint,
        )
        skipped = _scan_skipped_episode_uids(run_root, episodes, identity)
        pending = [episode for episode in episodes if episode.episode_uid not in skipped]
        worker_plans = _build_worker_plans(cfg, store, pending)
        run_plan = RunPlan(
            schema_version=1,
            benchmark=cfg.benchmark.name,
            run_name=cfg.output.run_name,
            config_sha256=config_sha256,
            input_fingerprint=input_fingerprint,
            total_episode_uids=[episode.episode_uid for episode in episodes],
            skipped_episode_uids=[episode.episode_uid for episode in episodes if episode.episode_uid in skipped],
            pending_episode_uids=[episode.episode_uid for episode in pending],
            worker_plan_paths=[str(store.worker_plan_path(worker.worker_index)) for worker in worker_plans],
        )
        if not dry_run:
            _write_run_artifacts(cfg, store, run_plan, worker_plans)
        return PlannedRun(
            run_plan=run_plan,
            worker_plans=worker_plans,
            episodes=episodes,
            skipped_episode_uids=skipped,
            pending_episodes=pending,
            run_root=run_root,
            lock=lock,
        )
    except Exception:
        if lock is not None:
            lock.release()
        raise


def config_identity_sha256(cfg: EvalConfig) -> str:
    payload = _jsonable_config(cfg)
    _drop_raw_fields(payload)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable_config(cfg: EvalConfig) -> dict:
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: convert(getattr(value, field.name)) for field in fields(value)}
        return value

    return convert(cfg)


def _validate_existing_config(config_path: Path, config_sha256: str) -> None:
    if not config_path.exists():
        return
    text = config_path.read_text(encoding="utf-8")
    if f"config_sha256: {config_sha256}" not in text:
        raise ValueError(f"existing config.yaml does not match current run identity: {config_path}")


def _validate_benchmark_episodes(cfg: EvalConfig, episodes: list[EvalEpisode]) -> None:
    spec_cls = load_class(cfg.benchmark.class_path)
    spec = spec_cls(**cfg.benchmark.kwargs)
    for episode in episodes:
        spec.validate_episode(episode, env=cfg.env, dataset=cfg.dataset)


def _scan_skipped_episode_uids(run_root: Path, episodes: list[EvalEpisode], identity: RunIdentity) -> set[str]:
    episodes_by_uid = {episode.episode_uid: episode for episode in episodes}
    skipped = set()
    for record in scan_eval_infos(run_root):
        if not record.valid or record.payload is None:
            continue
        episode_uid = str(record.payload.get("episode_uid"))
        episode = episodes_by_uid.get(episode_uid)
        if episode is None:
            continue
        if is_completed_skip_candidate(record.payload, episode, identity):
            skipped.add(episode_uid)
    return skipped


def _build_worker_plans(cfg: EvalConfig, store: ArtifactStore, pending: list[EvalEpisode]) -> list[WorkerPlan]:
    chunks = partition_contiguous(pending, len(cfg.parallel.gpu_ids))
    backend_planner = planner_from_config(cfg)
    plans = []
    for index, gpu_id in enumerate(cfg.parallel.gpu_ids):
        episodes = chunks[index]
        backend = backend_planner.plan_worker_backend(
            cfg=cfg.env,
            store=store,
            worker_index=index,
            physical_gpu_id=gpu_id,
        )
        plans.append(
            WorkerPlan(
                worker_index=index,
                physical_gpu_id=gpu_id,
                episodes=episodes,
                run_root=store.run_root,
                worker_log_path=store.worker_log_path(index),
                backend=backend,
                episode_attempts={
                    episode.episode_uid: f"worker{index}-{uuid.uuid4().hex[:12]}" for episode in episodes
                },
            )
        )
    return plans


def _write_run_artifacts(cfg: EvalConfig, store: ArtifactStore, run_plan: RunPlan, worker_plans: list[WorkerPlan]) -> None:
    config_payload = _config_yaml_payload(cfg)
    config_payload["config_sha256"] = run_plan.config_sha256
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config_payload), store.config_path)
    write_json_atomic(store.run_plan_path, _jsonable_config(run_plan))
    for worker in worker_plans:
        worker_payload = _jsonable_config(worker)
        worker_payload["backend"] = worker.backend.to_jsonable()
        write_json_atomic(store.worker_plan_path(worker.worker_index), worker_payload)


def _config_yaml_payload(cfg: EvalConfig) -> dict:
    payload = _jsonable_config(cfg.raw)
    payload.setdefault("benchmark", {})
    payload["benchmark"]["kwargs"] = _jsonable_config(cfg.benchmark.kwargs)
    payload.setdefault("input", {})
    if cfg.input.path is not None:
        payload["input"]["path"] = str(cfg.input.path)
    if cfg.input.data_root is not None:
        payload["input"]["data_root"] = str(cfg.input.data_root)
    if cfg.input.roots:
        payload["input"]["roots"] = [
            {"namespace": root.namespace, "path": str(root.path)}
            for root in cfg.input.roots
        ]
    payload.setdefault("model", {})
    payload["model"]["checkpoint"] = str(cfg.model.checkpoint)
    payload.setdefault("env", {})
    env_kwargs = payload["env"].setdefault("kwargs", {})
    for key, value in cfg.env.kwargs.items():
        if isinstance(value, Path):
            env_kwargs[key] = str(value)
        elif isinstance(value, tuple):
            env_kwargs[key] = list(value)
        else:
            env_kwargs[key] = value
    payload.setdefault("output", {})
    payload["output"]["root"] = str(cfg.output.root)
    return payload


def _drop_raw_fields(payload: object) -> None:
    if isinstance(payload, dict):
        payload.pop("raw", None)
        payload.pop("config_sha256", None)
        for value in payload.values():
            _drop_raw_fields(value)
    elif isinstance(payload, list):
        for value in payload:
            _drop_raw_fields(value)
