"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import importlib
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from transformers import PretrainedConfig, PreTrainedModel

from starVLA.model.framework import FRAMEWORK_REGISTRY
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
_FRAMEWORKS_IMPORTED = False
_RELEASE_FRAMEWORK_MODULES = ("starVLA.model.framework.VLM4A.simplenav",)


def _auto_import_framework_modules() -> None:
    global _FRAMEWORKS_IMPORTED
    if _FRAMEWORKS_IMPORTED:
        return

    for module_name in _RELEASE_FRAMEWORK_MODULES:
        importlib.import_module(module_name)

    _FRAMEWORKS_IMPORTED = True


def build_framework(cfg):  # The single entry point for building different model frameworks
    """
    Build a framework model from config.
    Args:
        cfg: Config object containing `cfg.framework.name`.
    Returns:
        nn.Module: Instantiated framework model.
    """
    if not hasattr(cfg, "framework") or not hasattr(cfg.framework, "name"):
        raise ValueError("Missing `cfg.framework.name`. The framework API now only accepts `framework.name`.")

    _auto_import_framework_modules()

    framework_id = cfg.framework.name
    if ":" in framework_id:
        module, name = framework_id.split(":")
        return getattr(importlib.import_module(module), name)(cfg)
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        available = sorted(FRAMEWORK_REGISTRY._registry.keys())
        raise NotImplementedError(f"Framework `{framework_id}` is not implemented. Available frameworks: {available}")

    model_class = FRAMEWORK_REGISTRY[framework_id]
    return model_class(cfg)


class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(self, hf_config=PretrainedConfig()) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """

        super().__init__(hf_config)

    # ------------------------------------------------------------------
    # Soft-constraint interface: subclasses should override these.
    # Default implementations raise NotImplementedError so that IDE
    # tooling (e.g. pylance, mypy) flags missing overrides, while
    # still allowing PreTrainedModel instantiation (no ABC).
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict], **kwargs) -> dict:
        """Training forward pass.

        Args:
            examples: List[dict], each dict requires at least:
                - image: List[PIL.Image]
                - lang: str
                - action: np.ndarray shaped [T, action_dim]

        Returns:
            dict: Must contain ``"action_loss"`` (torch.Tensor scalar).
                  May contain extra keys for logging (e.g. ``"kl_loss"``).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(examples) -> dict with 'action_loss' key."
        )

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Inference: predict future actions from observations.

        Args:
            examples: Same schema as *forward* (minus ``action`` which is optional).
            **kwargs: Framework-specific inference options.

        Returns:
            dict: Must contain ``"normalized_actions"`` (np.ndarray [B, T, action_dim]).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_action(examples) -> dict with 'normalized_actions' key."
        )

    # ------------------------------------------------------------------
    # Unified loss interface for Trainer
    # ------------------------------------------------------------------

    def supports_training_tag(self, tag: str) -> bool:
        """Return whether this framework can consume batches for *tag*."""
        if tag == "vla":
            return type(self).forward is not baseframework.forward
        if tag == "vlm":
            return type(self).forward_vlm is not baseframework.forward_vlm
        return False

    def compute_loss(self, tag: str, batch, loss_scale: dict = None) -> Dict[str, torch.Tensor] | None:
        """Unified forward entry-point: route to the right forward by *tag*.

        The trainer calls ``model.compute_loss(tag, batch)`` for every
        ``(tag, batch)`` pair produced by :class:`DataLoaderManager`.
        The model internally dispatches:

        - ``"vla"`` → ``self.forward(batch)``
        - ``"vlm"`` → ``self.forward_vlm(batch)``

        Subclasses can override this to add more tags (e.g. ``"world"``).

        Args:
            tag: dataset type tag (``"vla"``, ``"vlm"``, …)
            batch: the batch produced by the corresponding DataLoader.
            loss_scale: ``{"vla": 1.0, "vlm": 0.1}`` per-tag loss multiplier.
                        Defaults to 1.0 for unspecified tags.

        Returns:
            dict[str, Tensor] | None: keyed losses (e.g. ``{"action_loss": ...}``).
                Returns ``None`` when this framework does not support the
                incoming dataloader tag so the trainer can ``continue``.
        """
        if not self.supports_training_tag(tag):
            return None

        scale = (loss_scale or {}).get(tag, 1.0)

        if tag == "vla":
            out = self.forward(batch)
        elif tag == "vlm":
            out = self.forward_vlm(batch)
        else:
            return None

        # Apply loss scale and filter to Tensor values only
        return {k: v * scale for k, v in out.items() if isinstance(v, torch.Tensor)}

    def forward_vlm(self, batch):
        raise NotImplementedError("This model does not support VLM training")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        config_overrides: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats
        if config_overrides:
            allowed = {("framework", "qwenvl", "base_vlm"), ("framework", "qwenvl", "attn_implementation")}

            def leaves(value, prefix=()):
                for key, item in value.items():
                    if isinstance(item, dict):
                        yield from leaves(item, (*prefix, key))
                    else:
                        yield (*prefix, key)

            unsupported = set(leaves(config_overrides)) - allowed
            if unsupported:
                raise ValueError(f"Model structure is owned by checkpoint config; unsupported overrides: {unsupported}")
            model_config = OmegaConf.to_container(
                OmegaConf.merge(OmegaConf.create(model_config), OmegaConf.create(config_overrides)),
                resolve=True,
            )

        config = OmegaConf.create(model_config)
        model_config = config

        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        import json

        assets_path = pretrained_checkpoint.parents[1] / "model_assets.json"
        FrameworkModel.input_profiles = json.loads(assets_path.read_text())
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        if pretrained_checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            model_state_dict = load_file(str(pretrained_checkpoint))
        else:
            model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        FrameworkModel.load_state_dict(model_state_dict, strict=True)
        return FrameworkModel
