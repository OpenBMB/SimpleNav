from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from NavVLAeval.common.log.artifacts import scan_eval_infos
from NavVLAeval.common.types import EpisodeResult

DEFAULT_METRIC_KEYS = ("SR", "OSR", "NE", "SPL")
ALL_METRIC_KEYS = DEFAULT_METRIC_KEYS + ("nDTW", "path_length", "gt_path_length", "steps_taken")


def normalize_metric_keys(metric_keys: Sequence[str] | None) -> tuple[str, ...]:
    if not metric_keys:
        return DEFAULT_METRIC_KEYS
    keys = tuple(str(key).strip() for key in metric_keys if str(key).strip())
    unknown = sorted(set(keys) - set(ALL_METRIC_KEYS))
    if unknown:
        raise ValueError(f"Unsupported metric keys: {unknown}")
    return keys


def episode_metric_payload(
    result: EpisodeResult | Mapping[str, Any], *, metric_keys: Sequence[str] | None = None
) -> dict[str, Any]:
    result_dict = _result_dict(result)
    if result_dict.get("failure") is not None:
        return {key: None for key in normalize_metric_keys(metric_keys)}
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
    metrics = {key: None for key in selected_keys}
    if metric_total:
        per_episode = [_all_episode_metrics(result) for result in metric_results]
        metrics = {
            key: (
                sum(float(metric[key]) for metric in per_episode) / metric_total
                if all(metric.get(key) is not None for metric in per_episode)
                else None
            )
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


def summary_from_run_artifacts(
    run_plan_path: str | Path, run_root: str | Path, *, metric_keys: Sequence[str] | None = None
) -> dict[str, Any]:
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


def ndtw_score(
    predicted_points: Sequence[Any], reference_points: Sequence[Any], *, success_distance: float = 1.0
) -> float | None:
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
    success = float(int(result["success"]))
    oracle_success = float(int(result["oracle_success"]))
    final_distance = float(result["final_distance"])
    path_length = float(result["path_length"])
    gt_path_length = float(result["gt_path_length"])
    metrics = {
        "SR": success,
        "OSR": oracle_success,
        "NE": final_distance,
        "SPL": _spl(result),
        "nDTW": result.get("nDTW"),
        "path_length": path_length,
        "gt_path_length": gt_path_length,
        "steps_taken": float(result["steps"]),
    }
    return metrics


def _spl(result: Mapping[str, Any]) -> float:
    if not int(result["success"]):
        return 0.0
    gt_path_length = float(result["gt_path_length"])
    return gt_path_length / max(float(result["path_length"]), gt_path_length, 1e-6)


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
