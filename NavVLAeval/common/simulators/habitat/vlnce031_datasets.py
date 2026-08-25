from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import attr
from habitat.core.dataset import ALL_SCENES_MASK, Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import VLNEpisode


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"
R2R_DATASET_NAME = "R2R_VLNCE_v1-3_preprocessed"
RXR_DATASET_NAME = "RxR_VLNCE_v0"


def configure_dataset_config(
    cfg: Any,
    *,
    data_root: str | Path,
    task_name: str,
    split: str,
    roles: list[str] | tuple[str, ...] = (),
    languages: list[str] | tuple[str, ...] = (),
    content_scenes: list[str] | tuple[str, ...] = (),
) -> None:
    from omegaconf import OmegaConf

    root = Path(data_root).expanduser().resolve()
    role_values = tuple(str(role) for role in (roles or ("guide",)))
    language_values = tuple(str(language) for language in (languages or ("*",)))
    cfg.habitat.dataset.type = _dataset_type(task_name)
    cfg.habitat.dataset.split = str(split)
    cfg.habitat.dataset.data_path = vlnce_split_data_path(root, task_name=task_name, split=split, roles=role_values)
    cfg.habitat.dataset.scenes_dir = str(root / "scene_datasets")
    if content_scenes:
        cfg.habitat.dataset.content_scenes = [str(scene) for scene in content_scenes]
    OmegaConf.update(cfg, "habitat.dataset.roles", list(role_values), force_add=True)
    OmegaConf.update(cfg, "habitat.dataset.languages", list(language_values), force_add=True)


def vlnce_split_data_path(
    data_root: str | Path,
    *,
    task_name: str,
    split: str,
    roles: list[str] | tuple[str, ...] = (),
) -> str:
    return _split_file_path(data_root, task_name=task_name, split=split, suffix="", roles=roles)


def vlnce_split_gt_path(
    data_root: str | Path,
    *,
    task_name: str,
    split: str,
    roles: list[str] | tuple[str, ...] = (),
) -> str:
    return _split_file_path(data_root, task_name=task_name, split=split, suffix="_gt", roles=roles)


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: str | None = None
    language: str | None = None
    annotator_id: str | None = None
    edit_distance: float | None = None
    timed_instruction: list[dict[str, Any]] | None = None
    instruction_tokens: list[str] | None = None
    split: str | None = None


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    instruction: ExtendedInstructionData = attr.ib(default=None, validator=not_none_validator)
    trajectory_id: int | str = attr.ib(default=None, validator=not_none_validator)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Habitat 0.3.1 compatible loader for RxR VLN-CE splits."""

    annotation_roles = ("guide", "follower")
    supported_languages = ("en-US", "en-IN", "hi-IN", "te-IN")

    def __init__(self, config: Any | None = None) -> None:
        self.episodes: list[VLNExtendedEpisode] = []
        self.config = config
        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            filename = _format_data_path(config, role=role)
            with gzip.open(filename, "rt", encoding="utf-8") as handle:
                self.from_json(handle.read(), scenes_dir=str(_cfg(config, "scenes_dir", "")))

        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))
        language_filter = set(_cfg_list(config, "languages", default=(ALL_LANGUAGES_MASK,)))
        if ALL_LANGUAGES_MASK not in language_filter:
            self.episodes = [episode for episode in self.episodes if episode.instruction.language in language_filter]

    def from_json(self, json_str: str, scenes_dir: str | None = None) -> None:
        payload = json.loads(json_str)
        for episode_payload in payload.get("episodes", []):
            episode_dict = dict(episode_payload)
            episode_dict["episode_id"] = str(episode_dict["episode_id"])
            episode = VLNExtendedEpisode(**episode_dict)

            if scenes_dir:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(**episode.instruction)
            episode.instruction.split = str(_cfg(self.config, "split", ""))
            for index, goal in enumerate(episode.goals or []):
                episode.goals[index] = NavigationGoal(**goal)
            self.episodes.append(episode)

    @classmethod
    def get_scenes_to_load(cls, config: Any) -> list[str]:
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted({cls.scene_from_scene_path(episode.scene_id) for episode in dataset.episodes})

    @classmethod
    def extract_roles_from_config(cls, config: Any) -> list[str]:
        roles = _cfg_list(config, "roles", default=(ALL_ROLES_MASK,))
        if ALL_ROLES_MASK in roles:
            return list(cls.annotation_roles)
        unknown = set(roles) - set(cls.annotation_roles)
        if unknown:
            raise ValueError(f"Unsupported RxR annotation roles: {sorted(unknown)}")
        return list(roles)

    @classmethod
    def check_config_paths_exist(cls, config: Any) -> bool:
        return all(os.path.exists(_format_data_path(config, role=role)) for role in cls.extract_roles_from_config(config)) and os.path.exists(
            str(_cfg(config, "scenes_dir", ""))
        )


def _format_data_path(config: Any, *, role: str) -> str:
    data_path = str(_cfg(config, "data_path", ""))
    split = str(_cfg(config, "split", ""))
    return data_path.format(split=split, role=role)


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    lowered = key.lower()
    uppered = key.upper()
    try:
        return getattr(config, lowered)
    except Exception:
        pass
    try:
        return getattr(config, uppered)
    except Exception:
        return default


def _cfg_list(config: Any, key: str, *, default: tuple[str, ...]) -> list[str]:
    raw = _cfg(config, key, default)
    if raw is None:
        raw = default
    return [str(item) for item in raw]


def _dataset_type(task_name: str) -> str:
    task = str(task_name).lower()
    if task == "r2r":
        return "R2RVLN-v1"
    if task == "rxr":
        return "RxR-VLN-CE-v1"
    raise ValueError(f"Unsupported VLN-CE task_name: {task_name!r}")


def _split_file_path(
    data_root: str | Path,
    *,
    task_name: str,
    split: str,
    suffix: str,
    roles: list[str] | tuple[str, ...] = (),
) -> str:
    root = Path(data_root).expanduser().resolve()
    split_text = str(split)
    task = str(task_name).lower()
    if task == "r2r":
        split_dir = root / "datasets" / R2R_DATASET_NAME / split_text
        stem = f"{split_text}{suffix}"
    elif task == "rxr":
        role_values = tuple(str(role) for role in (roles or ("guide",)))
        if len(role_values) != 1:
            raise ValueError(f"Habitat031 RxR runtime expects exactly one role per worker, got {role_values!r}")
        split_dir = root / "datasets" / RXR_DATASET_NAME / split_text
        stem = f"{split_text}_{role_values[0]}{suffix}"
    else:
        raise ValueError(f"Unsupported VLN-CE task_name: {task_name!r}")
    gz_path = split_dir / f"{stem}.json.gz"
    if gz_path.exists():
        return str(gz_path)
    plain_path = split_dir / f"{stem}.json"
    if plain_path.exists():
        return str(_gzip_shadow_for_json(plain_path))
    return str(gz_path)


def _gzip_shadow_for_json(path: Path) -> Path:
    source = path.expanduser().resolve()
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()
    shadow_dir = Path(tempfile.gettempdir()) / "navvlaeval_habitat031_json_gz" / digest
    shadow_dir.mkdir(parents=True, exist_ok=True)
    target = shadow_dir / f"{source.name}.gz"
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        with source.open("rb") as src, gzip.open(str(target), "wb") as dst:
            shutil.copyfileobj(src, dst)
    return target
