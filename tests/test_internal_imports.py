from __future__ import annotations

import ast
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_ROOTS = ("starVLA", "examples", "deployment", "tool", "NavVLAeval")


def _internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
    return {name for name in imports if name.startswith(INTERNAL_ROOTS)}


def _module_exists(name: str, published_files: set[str]) -> bool:
    relative = "/".join(name.split("."))
    return (
        f"{relative}.py" in published_files
        or f"{relative}/__init__.py" in published_files
        or any(path.startswith(f"{relative}/") for path in published_files)
    )


def test_maintained_tree_has_no_new_missing_internal_imports() -> None:
    missing: set[str] = set()
    try:
        published_files = set(
            subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        published_files = {
            str(path.relative_to(REPO_ROOT))
            for relative_root in INTERNAL_ROOTS
            for path in (REPO_ROOT / relative_root).rglob("*.py")
        }
    for relative_path in published_files:
        if not relative_path.startswith(INTERNAL_ROOTS):
            continue
        path = REPO_ROOT / relative_path
        for name in _internal_imports(path):
            if not _module_exists(name, published_files):
                missing.add(name)

    assert missing == set()
