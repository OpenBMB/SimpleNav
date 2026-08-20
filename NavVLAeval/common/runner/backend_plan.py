from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from NavVLAeval.common.log.artifacts import ArtifactStore
    from NavVLAeval.common.config import EnvConfig


@dataclass(frozen=True)
class WorkerBackendPlan:
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": self.type, "kwargs": _to_jsonable(self.kwargs)}

    @classmethod
    def from_jsonable(cls, payload: Any) -> WorkerBackendPlan:
        if not isinstance(payload, dict):
            raise ValueError("worker backend plan must be a JSON object")
        backend_type = str(payload.get("type") or "")
        if not backend_type:
            raise ValueError("worker backend plan missing type")
        if "kwargs" not in payload:
            raise ValueError("worker backend plan missing kwargs")
        kwargs = payload.get("kwargs")
        if not isinstance(kwargs, dict):
            raise ValueError("worker backend plan kwargs must be an object")
        return cls(type=backend_type, kwargs=dict(kwargs))


class EnvironmentBackendPlanner(Protocol):
    def plan_worker_backend(
        self,
        *,
        cfg: EnvConfig,
        store: ArtifactStore,
        worker_index: int,
        physical_gpu_id: int,
    ) -> WorkerBackendPlan:
        ...


class OfflineBackendPlanner:
    def plan_worker_backend(
        self,
        *,
        cfg: EnvConfig,
        store: ArtifactStore,
        worker_index: int,
        physical_gpu_id: int,
    ) -> WorkerBackendPlan:
        del cfg, store, worker_index, physical_gpu_id
        return WorkerBackendPlan(type="offline", kwargs={})


def default_backend_planner(env_type: str) -> EnvironmentBackendPlanner:
    if env_type == "offline":
        return OfflineBackendPlanner()
    raise ValueError(f"env.planner_class_path is required for environment backend type: {env_type!r}")


def planner_from_config(cfg) -> EnvironmentBackendPlanner:
    if cfg.env.planner_class_path:
        from NavVLAeval.common.config import load_class

        planner_cls = load_class(cfg.env.planner_class_path)
        return planner_cls(**cfg.env.kwargs)
    return default_backend_planner(cfg.env.type)


def OfflineWorkerBackendPlan(*, type: str = "offline") -> WorkerBackendPlan:
    if type != "offline":
        raise ValueError(f"OfflineWorkerBackendPlan type must be 'offline', got {type!r}")
    return WorkerBackendPlan(type="offline", kwargs={})


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
