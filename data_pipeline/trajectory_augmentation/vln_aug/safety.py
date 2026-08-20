import hashlib
from pathlib import Path


def assert_safe_output_path(source_split: Path, output_split: Path) -> None:
    source = source_split.resolve()
    output = output_split.resolve()
    if output == source or source in output.parents:
        raise ValueError("output must not equal or be inside the source split")
    if output.parent != source.parent or output.name != f"{source.name}_enhanced":
        raise ValueError("output must be the required enhanced sibling of the source split")


def refuse_existing_output(output_split: Path) -> None:
    if output_split.exists() or output_split.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing enhanced split: {output_split}")


def assert_reports_outside_sources(reports_dir: Path, source_splits: list[Path]) -> None:
    reports = reports_dir.resolve()
    for source_split in source_splits:
        source = source_split.resolve()
        if reports == source or source in reports.parents:
            raise ValueError(f"reports directory must not be inside source split: {source}")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_sha256_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    base = root.resolve()
    manifest = {}
    for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix()
        stat = path.stat()
        manifest[relative] = {"size": stat.st_size, "sha256": _sha256_file(path)}
    return manifest


def build_file_stat_manifest(root: Path) -> dict[str, dict[str, int]]:
    base = root.resolve()
    manifest = {}
    for path in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.relative_to(base).as_posix()):
        stat = path.stat()
        manifest[path.relative_to(base).as_posix()] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return manifest
