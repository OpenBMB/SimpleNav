from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping

from tool.navvla.compute_bats_k import (
    BATSBudgetConfig,
    MINICPM_IMAGE_WRAPPER_TOKENS,
    compute_bats_row_budget,
    history_frame_capacity,
)


@dataclass(frozen=True)
class BATSSelectionResult:
    selected: list[dict[str, Any]]
    ranked_selected: list[dict[str, Any]]
    effective_k: float
    max_history_frames: int


def bats_keep_probability(*, history_step: int, current_step: int, k: float, epsilon: float = 0.1) -> float:
    eps = float(epsilon)
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    current = int(current_step)
    if current <= 0:
        return 0.0
    k_value = float(k)
    if not math.isfinite(k_value) or k_value < 0.0:
        raise ValueError(f"BATS k must be a finite non-negative value, got {k}")
    exponent = k_value * (float(history_step) - float(current)) / float(current)
    decay = 0.0 if exponent <= -745.0 else math.exp(min(exponent, 709.0))
    return float(min(1.0, max(0.0, (1.0 - eps) * decay + eps)))


def online_bats_history_budget(
    *,
    token_budget: int,
    budget_num_cameras: int,
    current_visual_tokens: int,
    history_visual_tokens: int,
    tvi_tokens: int,
    current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
    history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
) -> int:
    return history_frame_capacity(
        token_budget=token_budget,
        num_cameras=_positive_int(budget_num_cameras, name="budget_num_cameras"),
        current_visual_tokens=current_visual_tokens,
        history_visual_tokens=history_visual_tokens,
        tvi_tokens=tvi_tokens,
        current_wrapper_tokens=current_wrapper_tokens,
        history_wrapper_tokens=history_wrapper_tokens,
    )


def select_bats_history(
    *,
    candidates: list[tuple[int, dict[str, Any]]],
    anchor_frame_index: int,
    episode_id: str,
    dataset_name: str,
    seed: int,
    epsilon: float,
    k: float,
    use_dynamic_bats_k: bool = True,
    token_budget: int = 1024,
    budget_num_cameras: int = 1,
    current_visual_tokens: int = 64,
    history_visual_tokens: int = 4,
    tvi_tokens: int = 1,
    current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
    history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
    max_history_frames: int | None = None,
    sampling_mode: str = "priority_capped",
) -> BATSSelectionResult:
    budget_max_history_frames = online_bats_history_budget(
        token_budget=token_budget,
        budget_num_cameras=budget_num_cameras,
        current_visual_tokens=current_visual_tokens,
        history_visual_tokens=history_visual_tokens,
        tvi_tokens=tvi_tokens,
        current_wrapper_tokens=current_wrapper_tokens,
        history_wrapper_tokens=history_wrapper_tokens,
    )
    if max_history_frames is None:
        max_history_frames = budget_max_history_frames
    else:
        max_history_frames = min(budget_max_history_frames, max(0, int(max_history_frames)))
    effective_k = float(k)
    if use_dynamic_bats_k:
        effective_k = compute_bats_row_budget(
            len(candidates),
            num_cameras=budget_num_cameras,
            config=BATSBudgetConfig(
                token_budget=int(token_budget),
                epsilon=float(epsilon),
                current_visual_tokens=int(current_visual_tokens),
                history_visual_tokens=int(history_visual_tokens),
                tvi_tokens=int(tvi_tokens),
                current_wrapper_tokens=int(current_wrapper_tokens),
                history_wrapper_tokens=int(history_wrapper_tokens),
            ),
        ).k
    if max_history_frames <= 0 or not candidates:
        return BATSSelectionResult([], [], effective_k, max_history_frames)

    normalized_sampling_mode = str(sampling_mode).strip().lower()
    if normalized_sampling_mode not in {"priority_capped", "independent"}:
        raise ValueError(
            "BATS sampling_mode must be one of ['independent', 'priority_capped'], "
            f"got {sampling_mode!r}"
        )
    sampled: list[tuple[float, int, float, dict[str, Any]]] = []
    for frame_index, item in candidates:
        probability = bats_keep_probability(
            history_step=int(frame_index),
            current_step=int(anchor_frame_index),
            epsilon=epsilon,
            k=effective_k,
        )
        draw = random.Random(
            f"{int(seed)}:{dataset_name}:{episode_id}:{int(anchor_frame_index)}:{int(frame_index)}"
        ).random()
        if draw < probability and (
            normalized_sampling_mode != "independent" or len(sampled) < max_history_frames
        ):
            priority = (
                draw / probability if normalized_sampling_mode == "priority_capped" and probability > 0.0 else 0.0
            )
            sampled.append((priority, int(frame_index), probability, item))
    if normalized_sampling_mode == "priority_capped" and len(sampled) > max_history_frames:
        sampled = sorted(sampled, key=lambda entry: (entry[0], entry[1]))[:max_history_frames]
    sampled.sort(key=lambda entry: entry[1])
    ranked = sorted(sampled, key=lambda entry: (entry[2], entry[1]), reverse=True)
    return BATSSelectionResult(
        selected=[item for _priority, _frame_index, _probability, item in sampled],
        ranked_selected=[item for _priority, _frame_index, _probability, item in ranked],
        effective_k=effective_k,
        max_history_frames=max_history_frames,
    )


def select_online_bats_history(**kwargs: Any) -> list[dict[str, Any]]:
    return select_bats_history(**kwargs).selected


def select_long_memory_candidate(
    ranked_candidates: list[Mapping[str, Any]],
    *,
    memory_frame_indices: set[int],
) -> Mapping[str, Any] | None:
    memory_indices = {int(index) for index in memory_frame_indices}
    memory_max = max(memory_indices) if memory_indices else None
    for candidate in reversed(ranked_candidates):
        frame_index = _candidate_frame_index(candidate)
        if frame_index in memory_indices:
            continue
        if memory_max is not None and frame_index <= memory_max:
            continue
        return candidate
    return None


def _candidate_frame_index(candidate: Mapping[str, Any]) -> int:
    if "frame_index" in candidate:
        return int(candidate["frame_index"])
    metadata = candidate.get("navvla_eval")
    if isinstance(metadata, Mapping) and "frame_index" in metadata:
        return int(metadata["frame_index"])
    for container_key in ("traveluav_episode", "uavflow_pose"):
        container = candidate.get(container_key)
        if not isinstance(container, Mapping):
            continue
        metadata = container.get("navvla_eval")
        if isinstance(metadata, Mapping) and "frame_index" in metadata:
            return int(metadata["frame_index"])
    raise KeyError("BATS candidate is missing frame_index")


def _positive_int(value: int, *, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return integer


__all__ = [
    "BATSSelectionResult",
    "bats_keep_probability",
    "online_bats_history_budget",
    "select_bats_history",
    "select_long_memory_candidate",
    "select_online_bats_history",
]
