from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.optimize import brentq


MINICPM_IMAGE_WRAPPER_TOKENS = 3
QWEN35_CURRENT_IMAGE_WRAPPER_TOKENS = 2
QWEN35_CACHED_HISTORY_WRAPPER_TOKENS = 0


def visual_block_token_cost(
    *,
    visual_tokens: int,
    tvi_tokens: int,
    wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
) -> int:
    wrappers = _non_negative_int(wrapper_tokens, name="wrapper_tokens")
    return int(visual_tokens) + int(tvi_tokens) + wrappers


def resolve_visual_wrapper_tokens(
    *,
    visual_token_mode: str,
    visual_token_profile: str,
    current_wrapper_tokens: int | None = None,
    history_wrapper_tokens: int | None = None,
) -> tuple[int, int]:
    is_qwen35_cached_history = (
        str(visual_token_mode).strip().lower() == "cached_history_online_current"
        and str(visual_token_profile).strip().lower().startswith("qwen3_5_")
    )
    default_current = (
        QWEN35_CURRENT_IMAGE_WRAPPER_TOKENS
        if is_qwen35_cached_history
        else MINICPM_IMAGE_WRAPPER_TOKENS
    )
    default_history = (
        QWEN35_CACHED_HISTORY_WRAPPER_TOKENS
        if is_qwen35_cached_history
        else MINICPM_IMAGE_WRAPPER_TOKENS
    )
    return (
        _non_negative_int(
            default_current if current_wrapper_tokens is None else current_wrapper_tokens,
            name="current_wrapper_tokens",
        ),
        _non_negative_int(
            default_history if history_wrapper_tokens is None else history_wrapper_tokens,
            name="history_wrapper_tokens",
        ),
    )


def history_frame_capacity(
    *,
    token_budget: int,
    num_cameras: int,
    current_visual_tokens: int,
    history_visual_tokens: int,
    tvi_tokens: int,
    current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
    history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS,
) -> int:
    cameras = _positive_int(num_cameras, name="num_cameras")
    current_cost = cameras * visual_block_token_cost(
        visual_tokens=current_visual_tokens,
        tvi_tokens=tvi_tokens,
        wrapper_tokens=current_wrapper_tokens,
    )
    history_cost = cameras * visual_block_token_cost(
        visual_tokens=history_visual_tokens,
        tvi_tokens=tvi_tokens,
        wrapper_tokens=history_wrapper_tokens,
    )
    if history_cost <= 0:
        return 0
    return max(0, (int(token_budget) - current_cost) // history_cost)


@dataclass(frozen=True)
class BATSBudgetConfig:
    token_budget: int = 1024
    epsilon: float = 0.1
    current_visual_tokens: int = 64
    history_visual_tokens: int = 4
    tvi_tokens: int = 1
    current_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS
    history_wrapper_tokens: int = MINICPM_IMAGE_WRAPPER_TOKENS
    max_k: float = 1.0e6

    def target_history_frames(self, *, num_cameras: int) -> float:
        cameras = _positive_int(num_cameras, name="num_cameras")
        current_cost = visual_block_token_cost(
            visual_tokens=self.current_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            wrapper_tokens=self.current_wrapper_tokens,
        ) * cameras
        history_cost = visual_block_token_cost(
            visual_tokens=self.history_visual_tokens,
            tvi_tokens=self.tvi_tokens,
            wrapper_tokens=self.history_wrapper_tokens,
        ) * cameras
        return max(0.0, (float(self.token_budget) - float(current_cost)) / float(history_cost))


@dataclass(frozen=True)
class BATSRowBudget:
    k: float
    target_frames: float
    expected_frames: float
    budget_feasible: bool


def compute_bats_expected_frames(history_frames: int, *, k: float, epsilon: float) -> float:
    frames = _non_negative_int(history_frames, name="history_frames")
    if frames == 0:
        return 0.0
    eps = _validate_epsilon(epsilon)
    if float(k) <= 0.0:
        return float(frames)
    value = (1.0 - eps) * ((1.0 - math.exp(-float(k))) / float(k)) + eps
    return float(frames) * value


def solve_bats_k_for_history_frames(history_frames: int, *, num_cameras: int, config: BATSBudgetConfig) -> float:
    frames = _non_negative_int(history_frames, name="history_frames")
    if frames == 0:
        return 0.0

    target = min(float(frames), config.target_history_frames(num_cameras=num_cameras))
    floor = float(frames) * _validate_epsilon(config.epsilon)
    if target <= floor:
        return float(config.max_k)
    if target >= float(frames):
        return 0.0

    def objective(k: float) -> float:
        return compute_bats_expected_frames(frames, k=k, epsilon=config.epsilon) - target

    upper = 1.0
    while objective(upper) > 0.0:
        upper *= 2.0
        if upper > float(config.max_k):
            raise RuntimeError(f"failed to bracket BATS k for history_frames={frames}, target={target}")
    return float(brentq(objective, 1.0e-12, upper, xtol=1e-12, rtol=1e-12, maxiter=100))


def compute_bats_row_budget(history_frames: int, *, num_cameras: int, config: BATSBudgetConfig) -> BATSRowBudget:
    k = solve_bats_k_for_history_frames(history_frames, num_cameras=num_cameras, config=config)
    target_frames = min(float(history_frames), config.target_history_frames(num_cameras=num_cameras))
    budget_feasible = int(history_frames) == 0 or target_frames > float(history_frames) * _validate_epsilon(config.epsilon)
    return BATSRowBudget(
        k=k,
        target_frames=target_frames,
        expected_frames=compute_bats_expected_frames(history_frames, k=k, epsilon=config.epsilon),
        budget_feasible=budget_feasible,
    )


def update_bats_k_for_dataset_root(
    dataset_root: str | Path,
    *,
    config: BATSBudgetConfig = BATSBudgetConfig(),
    budget_num_cameras: int | None = None,
    dry_run: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    from tool.navvla.context_index import resolve_context_index_paths

    root = Path(dataset_root)
    context_paths = resolve_context_index_paths(root, token_budget=int(config.token_budget))
    context_path = context_paths.meta_path
    debug_path = context_paths.debug_path

    camera_count = _camera_count(root, budget_num_cameras=budget_num_cameras, debug_path=debug_path)
    context = pd.read_parquet(context_path)
    debug = pd.read_parquet(debug_path) if debug_path.exists() else None
    if debug is not None and len(debug) != len(context):
        raise ValueError(f"debug context row count {len(debug)} does not match main context row count {len(context)}")

    history_frames = _history_frame_counts(context, debug=debug, num_cameras=camera_count)
    budgets = [compute_bats_row_budget(int(count), num_cameras=camera_count, config=config) for count in history_frames]
    k_values = [budget.k for budget in budgets]
    target_frames = [budget.target_frames for budget in budgets]
    expected_frames = [budget.expected_frames for budget in budgets]
    budget_feasible = [budget.budget_feasible for budget in budgets]

    updated_context = context.copy()
    updated_context["bats_k"] = np.asarray(k_values, dtype=np.float64)

    updated_debug = None
    if debug is not None:
        updated_debug = debug.copy()
        updated_debug["bats_k"] = np.asarray(k_values, dtype=np.float64)
        updated_debug["bats_expected_frames"] = np.asarray(expected_frames, dtype=np.float64)
        updated_debug["bats_target_frames"] = np.asarray(target_frames, dtype=np.float64)
        updated_debug["bats_budget_tokens"] = int(config.token_budget)
        updated_debug["bats_epsilon"] = float(config.epsilon)
        updated_debug["bats_num_cameras"] = int(camera_count)
        updated_debug["bats_budget_feasible"] = np.asarray(budget_feasible, dtype=bool)

    if not dry_run:
        if backup:
            _backup_file(context_path)
            if debug_path.exists():
                _backup_file(debug_path)
        updated_context.to_parquet(context_path, index=False)
        if updated_debug is not None:
            updated_debug.to_parquet(debug_path, index=False)

    finite_k = np.asarray(k_values, dtype=np.float64)
    return {
        "dataset_root": str(root),
        "rows": int(len(context)),
        "num_cameras": int(camera_count),
        "token_budget": int(config.token_budget),
        "epsilon": float(config.epsilon),
        "current_visual_tokens": int(config.current_visual_tokens),
        "history_visual_tokens": int(config.history_visual_tokens),
        "tvi_tokens": int(config.tvi_tokens),
        "target_history_frames_per_row": float(config.target_history_frames(num_cameras=camera_count)),
        "history_frames_min": int(min(history_frames)) if history_frames else 0,
        "history_frames_max": int(max(history_frames)) if history_frames else 0,
        "budget_infeasible_rows": int(sum(not value for value in budget_feasible)),
        "k_min": float(finite_k.min()) if finite_k.size else math.inf,
        "k_max": float(finite_k.max()) if finite_k.size else math.inf,
        "dry_run": bool(dry_run),
        "wrote": [] if dry_run else [str(path) for path in [context_path, debug_path] if path.exists()],
    }


def _history_frame_counts(context: pd.DataFrame, *, debug: pd.DataFrame | None, num_cameras: int) -> list[int]:
    if debug is not None and "token_count_before" in debug.columns:
        return [max(0, int(round(float(value) / float(num_cameras)))) for value in debug["token_count_before"].tolist()]
    raise ValueError("cannot infer pre-sampling BATS candidate count; rebuild the compact context debug shard")


def _camera_count(root: Path, *, budget_num_cameras: int | None = None, debug_path: Path | None = None) -> int:
    if budget_num_cameras is not None:
        return _positive_int(budget_num_cameras, name="budget_num_cameras")
    if debug_path is not None and debug_path.exists():
        debug = pd.read_parquet(debug_path)
        if "bats_num_cameras" in debug.columns and len(debug):
            return _positive_int(int(debug["bats_num_cameras"].iloc[0]), name="bats_num_cameras")
    cameras_path = root / "meta" / "navvla_cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"missing cameras metadata: {cameras_path}")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"{cameras_path} must contain a non-empty camera object")
    return len(cameras)


def _backup_file(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def _validate_epsilon(value: float) -> float:
    epsilon = float(value)
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {value}")
    return epsilon


def _non_negative_int(value: int, *, name: str) -> int:
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return integer


def _positive_int(value: int, *, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return integer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute and backfill per-row NavVLA BATS k values with Brent root solving.")
    parser.add_argument("dataset_roots", type=Path, nargs="+", help="NavVLA LeRobot v3 split root(s), e.g. vln_train")
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--budget-num-cameras", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--current-visual-tokens", type=int, default=64)
    parser.add_argument("--history-visual-tokens", type=int, default=4)
    parser.add_argument("--tvi-tokens", type=int, default=1)
    parser.add_argument("--max-k", type=float, default=1.0e6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", help="Write .bak copies before overwriting parquet files")
    args = parser.parse_args(argv)

    config = BATSBudgetConfig(
        token_budget=args.token_budget,
        epsilon=args.epsilon,
        current_visual_tokens=args.current_visual_tokens,
        history_visual_tokens=args.history_visual_tokens,
        tvi_tokens=args.tvi_tokens,
        max_k=args.max_k,
    )
    summaries = [
        update_bats_k_for_dataset_root(
            root,
            config=config,
            budget_num_cameras=args.budget_num_cameras,
            dry_run=args.dry_run,
            backup=args.backup,
        )
        for root in args.dataset_roots
    ]
    print(json.dumps(summaries[0] if len(summaries) == 1 else summaries, indent=2))


if __name__ == "__main__":
    main()
