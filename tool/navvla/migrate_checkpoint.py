"""One-time conversion to SimpleNav assets; the source checkpoint is never modified."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf


def convert(source: Path, output: Path, *, assets_path: Path | None = None):
    import torch

    source = source.resolve()
    run = source.parents[1]
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    config_path = run / "config.full.yaml"
    if not config_path.exists():
        config_path = run / "config.yaml"
    config = OmegaConf.load(config_path)
    old_name = str(config.framework.name)
    kinds = {"navvla_qwen35_cpm": "qwen35", "navvla_cpm": "minicpm"}
    kind = kinds[old_name]
    config.framework.name = "simplenav"
    config.framework.qwenvl.type = kind
    head = config.framework.action_model
    if "action_horizon" not in head and "future_action_window_size" in head:
        head.action_horizon = int(head.future_action_window_size) + 1
    for alias in ("future_action_window_size", "past_action_window_size"):
        head.pop(alias, None)
    config.framework.action_model.type = "flow_matching"
    config.framework.action_model.padding_loss = "zero_target"
    if config.datasets.vla_data.get("CoT_prompt"):
        config.framework.navvla.prompt_template = config.datasets.vla_data.pop("CoT_prompt")
    config.datasets.vla_data.state_dim = int(config.framework.action_model.get("state_dim", 0))
    if assets_path is None:
        from starVLA.dataloader.cpm_lerobot.builder import build_cpm_dataset

        profiles = build_cpm_dataset(config.datasets.vla_data).model_input_profiles()
    else:
        profiles = json.loads(assets_path.read_text())
    if source.suffix == ".safetensors":
        from safetensors.torch import load_file

        weights = load_file(str(source))
    else:
        weights = torch.load(source, map_location="cpu", weights_only=True)
    prefix = "qwen35_vl_interface." if kind == "qwen35" else "minicpm_vl_interface."
    weights = {
        ("backbone." + key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in weights.items()
    }
    output.mkdir(parents=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save(weights, checkpoint_dir / "pytorch_model.pt")
    text = (
        OmegaConf.to_yaml(config).replace("qwen35_vl_interface", "backbone").replace("minicpm_vl_interface", "backbone")
    )
    (output / "config.full.yaml").write_text(text)
    (output / "model_assets.json").write_text(json.dumps(profiles, indent=2))
    shutil.copy2(run / "dataset_statistics.json", output / "dataset_statistics.json")
    return checkpoint_dir / "pytorch_model.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--assets",
        type=Path,
        help="Previously exported model_assets.json; otherwise export metadata once from training data",
    )
    args = parser.parse_args()
    print(convert(args.source, args.output, assets_path=args.assets))


if __name__ == "__main__":
    main()
