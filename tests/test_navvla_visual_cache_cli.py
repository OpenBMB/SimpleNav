from tool.navvla.cli import generate_visual_cache as gvc
from tool.navvla.cli.generate_visual_cache import parse_args


def test_visual_cache_preserves_minicpm_default_448_square_resize() -> None:
    args = parse_args(["/tmp/navvla_dataset"])

    assert args.input_resize == (448, 448)


def test_visual_cache_accepts_explicit_resize_override() -> None:
    args = parse_args(["/tmp/navvla_dataset", "--input-resize", "640x360"])

    assert args.input_resize == (640, 360)


def test_visual_cache_accepts_camera_filter() -> None:
    args = parse_args(["/tmp/navvla_dataset", "--camera-names", "front", "left", "right", "rear"])

    assert args.camera_names == ["front", "left", "right", "rear"]


def test_visual_cache_validate_before_runs_before_dry_run(tmp_path, monkeypatch) -> None:
    validated = []
    monkeypatch.setattr(gvc, "available_context_token_budgets", lambda _root: [512, 1024])
    monkeypatch.setattr(
        gvc,
        "validate_navvla_lerobot_dataset",
        lambda root, **kwargs: validated.append((root, kwargs["token_budget"])),
    )
    monkeypatch.setattr(gvc, "load_visual_cache_refs", lambda *_args, **_kwargs: [])

    gvc.main([str(tmp_path), "--validate-before", "--all-token-budgets", "--dry-run"])

    assert validated == [(tmp_path, 512), (tmp_path, 1024)]
