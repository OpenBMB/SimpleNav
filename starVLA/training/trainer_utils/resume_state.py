"""Small helpers for resumable StarVLA training state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterator


_STATE_DIR_RE = re.compile(r"steps_(\d+)_state$")


@dataclass
class TrainingProgressState:
    completed_steps: int = 0
    vla_epoch_count: int = 0
    vla_batch_in_epoch: int = 0
    global_batches_consumed: int = 0
    schema_version: int = 1

    def state_dict(self) -> dict[str, int]:
        return {
            "schema_version": int(self.schema_version),
            "completed_steps": int(self.completed_steps),
            "vla_epoch_count": int(self.vla_epoch_count),
            "vla_batch_in_epoch": int(self.vla_batch_in_epoch),
            "global_batches_consumed": int(self.global_batches_consumed),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.schema_version = int(state_dict.get("schema_version", 1))
        self.completed_steps = int(state_dict.get("completed_steps", 0))
        self.vla_epoch_count = int(state_dict.get("vla_epoch_count", 0))
        self.vla_batch_in_epoch = int(state_dict.get("vla_batch_in_epoch", 0))
        self.global_batches_consumed = int(state_dict.get("global_batches_consumed", 0))


def parse_training_state_step(path: str | Path) -> int | None:
    match = _STATE_DIR_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def completed_training_state_step(path: str | Path) -> int | None:
    checkpoint_path = Path(path)
    directory_step = parse_training_state_step(checkpoint_path)
    if directory_step is None:
        return None
    trainer_state_path = checkpoint_path / "trainer_state.json"
    try:
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        marker_step = int(payload["completed_steps"])
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return directory_step if marker_step == directory_step else None


def find_latest_training_state_checkpoint(checkpoint_dir: str | Path) -> tuple[Path | None, int]:
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.exists():
        return None, 0

    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir():
            continue
        step = completed_training_state_step(path)
        if step is None:
            continue
        candidates.append((step, path))

    if not candidates:
        return None, 0

    step, path = max(candidates, key=lambda item: item[0])
    return path, step


def set_dataloader_epoch(dataloader, epoch: int) -> None:
    pending = [dataloader]
    seen_targets: set[int] = set()
    while pending:
        target = pending.pop()
        if target is None:
            continue
        target_id = id(target)
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)

        setter = getattr(target, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))
        for attribute in ("dataloader", "base_dataloader", "sampler", "batch_sampler"):
            child = getattr(target, attribute, None)
            if child is not None:
                pending.append(child)


def make_resume_dataloader_iterator(
    dataloader,
    progress_state: TrainingProgressState,
    *,
    skip_first_batches_fn: Callable[[object, int], object] | None = None,
) -> Iterator:
    set_dataloader_epoch(dataloader, progress_state.vla_epoch_count)
    batches_to_skip = max(0, int(progress_state.vla_batch_in_epoch))
    if batches_to_skip and skip_first_batches_fn is not None:
        dataloader = skip_first_batches_fn(dataloader, batches_to_skip)
    return iter(dataloader)
