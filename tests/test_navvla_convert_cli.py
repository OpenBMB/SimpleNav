from __future__ import annotations

from pathlib import Path


def test_unified_convert_cli_dispatches_through_registered_adapter(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.cli import convert_dataset

    captured = {}

    class FakeAdapter:
        def configure(self, **kwargs):
            return self

        def convert(self, **kwargs):
            captured.update(kwargs)
            return {"dataset_root": str(tmp_path / "out" / "vln_train")}

    monkeypatch.setattr(convert_dataset, "get_adapter", lambda name: FakeAdapter())
    args = convert_dataset.build_parser().parse_args(
        [
            "--adapter",
            "traveluav",
            "--source-root",
            str(tmp_path / "source"),
            "--output-root",
            str(tmp_path / "out"),
            "--dataset-name",
            "vln_train",
            "--no-visual-token-cache",
            "--overwrite",
        ]
    )

    summary = convert_dataset.convert_from_args(args)

    assert summary["dataset_root"].endswith("vln_train")
    assert captured["source_root"] == tmp_path / "source"
    assert captured["output_root"] == tmp_path / "out"
    assert captured["dataset_name"] == "vln_train"
    assert captured["write_visual_token_cache"] is False


def test_unified_convert_cli_configures_uav_flow_adapter(tmp_path: Path, monkeypatch) -> None:
    from tool.navvla.cli import convert_dataset

    captured = {}

    class FakeAdapter:
        def configure(self, **kwargs):
            captured["configure"] = kwargs
            return self

        def convert(self, **kwargs):
            captured["convert"] = kwargs
            return {"dataset_root": str(tmp_path / "out" / "vln_train")}

    monkeypatch.setattr(convert_dataset, "get_adapter", lambda name: FakeAdapter())
    args = convert_dataset.build_parser().parse_args(
        [
            "--adapter",
            "uav_flow",
            "--source-root",
            str(tmp_path / "family"),
            "--source-root-is-family-root",
            "--variant",
            "sim",
            "--media-cache-root",
            str(tmp_path / "media"),
            "--instruction-field",
            "instruction_unified",
            "--load-workers",
            "3",
            "--reuse-media-cache",
            "--output-root",
            str(tmp_path / "out"),
            "--dataset-name",
            "vln_train",
            "--no-visual-token-cache",
            "--overwrite",
        ]
    )

    convert_dataset.convert_from_args(args)

    assert captured["configure"] == {
        "media_cache_root": tmp_path / "media",
        "variant": "sim",
        "fps": 5.0,
        "action_horizon": 8,
        "instruction_field": "instruction_unified",
        "reuse_media_cache": True,
        "load_workers": 3,
    }
    assert captured["convert"]["source_root"].name == "UAV-Flow-Sim"
