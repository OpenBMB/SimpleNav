from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool.navvla.repair import repair_navvla_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or apply BATS context repair to a NavVLA LeRobot root.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--token-budget", action="append", type=int, dest="token_budgets")
    parser.add_argument("--budget-num-cameras", type=int, default=None)
    parser.add_argument("--history-camera-names", nargs="+", default=None)
    parser.add_argument("--history-visual-tokens", type=int, default=4)
    parser.add_argument("--current-visual-tokens", type=int, default=64)
    parser.add_argument("--tvi-tokens", type=int, default=1)
    parser.add_argument("--context-epsilon", type=float, default=0.1)
    parser.add_argument("--context-seed", type=int, default=42)
    parser.add_argument("--no-long-memory", action="store_false", dest="include_long_memory")
    parser.set_defaults(include_long_memory=True)
    args = parser.parse_args()
    print(json.dumps(repair_navvla_dataset(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
