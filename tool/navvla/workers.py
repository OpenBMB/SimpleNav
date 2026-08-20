from __future__ import annotations

import os


def resolve_workers(workers: int | None) -> int:
    if workers is None:
        return min(32, os.cpu_count() or 1)
    if int(workers) < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    return int(workers)
