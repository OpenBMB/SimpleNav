from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from NavVLAeval.common.log.artifacts import scan_eval_infos
from NavVLAeval.common.types import EpisodeResult


DEFAULT_METRIC_KEYS = ("SR", "OSR", "NE", "SPL", "standard_SPL")
ALL_METRIC_KEYS = DEFAULT_METRIC_KEYS + ("nDTW", "path_length", "gt_path_length", "steps_taken")


def normalize_metric_keys(metric_keys: Sequence[str] | None) -> tuple[str, ...]:
    if not metric_keys:
        return DEFAULT_METRIC_KEYS
    keys = tuple(str(key).strip() for key in metric_keys if str(key).strip())
    unknown = sorted(set(keys) - set(ALL_METRIC_KEYS))
    if unknown:
        raise ValueError(f"Unsupported metric keys: {unknown}")
    return keys


def episode_metric_payload(result: EpisodeResult | Mapping[str, Any], *, metric_keys: Sequence[str] | None = None) -> dict[str, Any]:
    result_dict = _result_dict(result)
    all_metrics = _all_episode_metrics(result_dict)
    return {key: all_metrics[key] for key in normalize_metric_keys(metric_keys) if key in all_metrics}


def summarize_result_metrics(
    results: Sequence[EpisodeResult | Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    selected_keys = normalize_metric_keys(metric_keys)
    result_dicts = [_result_dict(result) for result in results]
    total = len(result_dicts)
    failed = sum(1 for result in result_dicts if result.get("failure") is not None)
    metric_results = [result for result in result_dicts if result.get("failure") is None]
    metric_total = len(metric_results)
    metrics = {key: 0.0 for key in selected_keys}
    if metric_total:
        per_episode = [_all_episode_metrics(result) for result in metric_results]
        metrics = {
            key: float(sum(float(metric.get(key, 0.0)) for metric in per_episode) / metric_total)
            for key in selected_keys
        }
    return {
        "total_episodes": total,
        "completed_episodes": sum(1 for result in result_dicts if result.get("failure") is None),
        "failed_episodes": failed,
        "metric_episodes": metric_total,
        "metrics": metrics,
    }


def summarize_results_by_scene(
    results: Sequence[EpisodeResult | Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        result_dict = _result_dict(result)
        groups[str(result_dict["scene_id"])].append(result_dict)
    return {
        scene_id: summarize_result_metrics(scene_results, metric_keys=metric_keys)
        for scene_id, scene_results in sorted(groups.items())
    }


def summary_from_run_artifacts(run_plan_path: str | Path, run_root: str | Path, *, metric_keys: Sequence[str] | None = None) -> dict[str, Any]:
    run_plan_path = Path(run_plan_path)
    run_root = Path(run_root)
    run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    total_uids = list(run_plan.get("total_episode_uids", []))
    skipped_uids = set(run_plan.get("skipped_episode_uids", []))
    pending_uids = set(run_plan.get("pending_episode_uids", []))
    valid_infos = []
    failure_breakdown: Counter[str] = Counter()
    for record in scan_eval_infos(run_root):
        if not record.valid or record.payload is None:
            continue
        payload = record.payload
        episode_uid = str(payload.get("episode_uid"))
        if episode_uid not in set(total_uids):
            continue
        if not _matches_run_identity(payload, run_plan):
            raise ValueError(f"eval_info run identity mismatch for {episode_uid}: {record.path}")
        status = str(payload.get("status") or "")
        if status not in {"completed", "failed"}:
            continue
        valid_infos.append(payload)
        if payload.get("failure") is not None:
            failure_breakdown[str(payload.get("failure_type") or "unknown")] += 1

    selected_keys = normalize_metric_keys(metric_keys or _summary_metric_keys_from_eval_infos(valid_infos))
    result_uids = {str(payload["episode_uid"]) for payload in valid_infos}
    base_summary = summarize_result_metrics(valid_infos, metric_keys=selected_keys)
    unresolved = len([uid for uid in pending_uids if uid not in result_uids])
    return {
        "schema_version": 1,
        "benchmark": run_plan["benchmark"],
        "run_name": run_plan["run_name"],
        "config_path": "config.yaml",
        "config_sha256": run_plan["config_sha256"],
        "input_fingerprint": run_plan["input_fingerprint"],
        "total_episodes": len(total_uids),
        "completed_episodes": base_summary["completed_episodes"],
        "failed_episodes": base_summary["failed_episodes"],
        "metric_episodes": base_summary["metric_episodes"],
        "skipped_episodes": len(skipped_uids),
        "pending_episodes": len(pending_uids),
        "unresolved_episodes": unresolved,
        "metrics": base_summary["metrics"],
        "scene_metrics": summarize_results_by_scene(valid_infos, metric_keys=selected_keys),
        "failure_breakdown": dict(sorted(failure_breakdown.items())),
    }


def ndtw_score(predicted_points: Sequence[Any], reference_points: Sequence[Any], *, success_distance: float = 1.0) -> float | None:
    if not predicted_points or not reference_points:
        return None
    predicted = np.asarray(predicted_points, dtype=np.float32).reshape(len(predicted_points), -1)[:, :3]
    reference = np.asarray(reference_points, dtype=np.float32).reshape(len(reference_points), -1)[:, :3]
    dtw_distance = _dtw_distance(predicted, reference)
    return float(math.exp(-dtw_distance / max(float(len(reference)) * float(success_distance), 1e-6)))


def _dtw_distance(predicted: np.ndarray, reference: np.ndarray) -> float:
    n, m = predicted.shape[0], reference.shape[0]
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(predicted[i - 1] - reference[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _result_dict(result: EpisodeResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EpisodeResult):
        return asdict(result)
    if is_dataclass(result):
        return asdict(result)
    return dict(result)


def _all_episode_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    success = float(int(result.get("success", 0)))
    oracle_success = float(int(result.get("oracle_success", 0)))
    final_distance = float(result.get("final_distance", 0.0))
    path_length = float(result.get("path_length", 0.0))
    gt_path_length = float(result.get("gt_path_length", 0.0))
    metrics = {
        "SR": success,
        "OSR": oracle_success,
        "NE": final_distance,
        "SPL": _spl(result),
        "standard_SPL": _standard_spl(result),
        "nDTW": _ndtw(result),
        "path_length": path_length,
        "gt_path_length": gt_path_length,
        "steps_taken": float(result.get("steps", 0)),
    }
    payload_metrics = result.get("metrics")
    if isinstance(payload_metrics, Mapping):
        for key, value in payload_metrics.items():
            if key in ALL_METRIC_KEYS and value is not None:
                metrics[str(key)] = float(value)
    return metrics


def _spl(result: Mapping[str, Any]) -> float:
    if not int(result.get("success", 0)):
        return 0.0
    gt_path_length = float(result.get("gt_path_length", 0.0))
    return gt_path_length / max(float(result.get("path_length", 0.0)), gt_path_length, 1e-6)


def _standard_spl(result: Mapping[str, Any]) -> float:
    if not int(result.get("success", 0)):
        return 0.0
    return float(result.get("gt_path_length", 0.0)) / max(
        float(result.get("path_length", 0.0)),
        float(result.get("gt_path_length", 0.0)),
        1e-6,
    )


def _ndtw(result: Mapping[str, Any]) -> float:
    if result.get("nDTW") is not None:
        return float(result["nDTW"])
    if result.get("ndtw") is not None:
        return float(result["ndtw"])
    gt_path_length = float(result.get("gt_path_length", 0.0))
    final_distance = max(float(result.get("final_distance", 0.0)), 0.0)
    if gt_path_length <= 1e-6:
        return 1.0 if final_distance <= 1e-6 else 0.0
    return float(math.exp(-final_distance / max(gt_path_length, 1e-6)))


def _matches_run_identity(payload: Mapping[str, Any], run_plan: Mapping[str, Any]) -> bool:
    for key in ("benchmark", "run_name", "config_sha256", "input_fingerprint"):
        if payload.get(key) != run_plan.get(key):
            return False
    return True


def _summary_metric_keys_from_eval_infos(payloads: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    for payload in payloads:
        metrics = payload.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return tuple(str(key) for key in metrics.keys() if str(key) in ALL_METRIC_KEYS)
    return DEFAULT_METRIC_KEYS


class MetricEvaluator:
    def __init__(self, *, metric_keys: Sequence[str] | None = None) -> None:
        self.metric_keys = normalize_metric_keys(metric_keys)
        self.results: list[EpisodeResult] = []

    def add(self, result: EpisodeResult) -> None:
        self.results.append(result)

    def summary(self) -> dict[str, Any]:
        return summarize_result_metrics(self.results, metric_keys=self.metric_keys)
