from __future__ import annotations

from pathlib import Path

import pytest

from navvla_conversion.adapters import get_adapter
from navvla_conversion.adapters.cosfly_convert import build_parser as build_cosfly_parser
from navvla_conversion.cli import convert_dataset
from navvla_conversion.cli.render_vlnce_rgb import build_parser as build_render_parser


ADAPTERS = (
    "traveluav",
    "aerialvln",
    "vlnce_rendered",
    "flight",
    "indooruav",
    "huge",
    "embodiednav",
    "enhanced_vln",
    "openfly",
    "openscene",
    "nuscenes",
)


@pytest.mark.parametrize("name", ADAPTERS)
def test_all_unified_adapters_are_registered(name: str) -> None:
    assert get_adapter(name).name == name


def test_convert_cli_is_standalone_and_accepts_worker_alias(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    class FakeAdapter:
        def configure(self, **kwargs):
            captured["configure"] = kwargs
            return self

        def convert(self, **kwargs):
            captured["convert"] = kwargs
            return {"dataset_root": str(tmp_path / "out" / "vln_train")}

    monkeypatch.setattr(convert_dataset, "get_adapter", lambda _name: FakeAdapter())
    args = convert_dataset.build_parser().parse_args(
        [
            "--adapter",
            "traveluav",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--cache-workers",
            "2",
        ]
    )
    convert_dataset.convert_from_args(args)
    assert captured["convert"]["write_workers"] == 2
    assert captured["convert"]["write_visual_token_cache"] is False


def test_visual_cache_arguments_are_not_public() -> None:
    parser = convert_dataset.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--adapter",
                "traveluav",
                "--source-root",
                "source",
                "--output-root",
                "out",
                "--no-visual-token-cache",
            ]
        )


def test_cosfly_exposes_only_manifest_and_conversion_commands() -> None:
    parser = build_cosfly_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"prepare-manifest", "convert"}


def test_vlnce_renderer_requires_portable_paths() -> None:
    parser = build_render_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--family", "r2r", "--split", "train"])
