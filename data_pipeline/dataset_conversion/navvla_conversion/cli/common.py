from __future__ import annotations

import argparse
import json
from typing import Any


def print_json_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def add_apply_backup_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--apply", action="store_true", help="Apply changes in place. Default is dry-run when supported.")
    parser.add_argument("--backup", action="store_true", help="Create backup files before applying changes.")
