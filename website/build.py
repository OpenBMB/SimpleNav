#!/usr/bin/env python3
"""Build the SimpleNAV static project page into a clean deployment directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"


def build(output: Path) -> None:
    if output.exists():
        if output.is_dir() and output.resolve() == ROOT / "_site":
            shutil.rmtree(output)
        else:
            raise SystemExit(f"Refusing to replace unexpected output path: {output}")
    output.mkdir(parents=True)

    for name in ("index.html", "assets", "data"):
        source = WEBSITE / name
        target = output / name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    logo = ROOT / "docs/assets/logo.jpg"
    (output / "assets").mkdir(exist_ok=True)
    shutil.copy2(logo, output / "assets/logo.jpg")
    shutil.copytree(ROOT / "docs/assets/figures", output / "assets/figures")
    shutil.copytree(ROOT / "docs/assets/demos", output / "assets/demos")
    shutil.copytree(
        ROOT / "data_pipeline/docs/assets/trajectory_comparisons",
        output / "assets/augmentation",
    )
    (output / ".nojekyll").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    args = parser.parse_args()
    build(args.output.resolve())
    print(f"Built SimpleNAV project page at {args.output.resolve()}")


if __name__ == "__main__":
    main()
