from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from starVLA.model.modules.action_model import GR00T_ActionHeader as action_header_module
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    _validate_loss_dim_weights,
    _weighted_action_mse,
)


def _flowmatching_action_head_config(*, loss_dim_weights=None) -> OmegaConf:
    action_model = {
        "action_model_type": "DiT-B",
        "action_dim": 4,
        "state_dim": 0,
        "action_horizon": 2,
        "num_inference_timesteps": 1,
        "hidden_size": 8,
        "num_target_vision_tokens": 1,
        "add_pos_embed": False,
        "max_seq_len": 2,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 10,
        "diffusion_model_cfg": {},
    }
    if loss_dim_weights is not None:
        action_model["loss_dim_weights"] = loss_dim_weights
    return OmegaConf.create({"framework": {"action_model": action_model}})


@pytest.mark.parametrize("raw_weights", [None, [1.0, 1.0, 1.0, 1.0]])
def test_uniform_action_loss_weights_match_original_mse(raw_weights) -> None:
    prediction = torch.tensor(
        [
            [[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]],
            [[-1.0, 2.0, -3.0, 4.0], [2.5, -1.5, 0.5, -3.5]],
        ]
    )
    target = torch.zeros_like(prediction)
    weights = _validate_loss_dim_weights(raw_weights, action_dim=4)

    actual = _weighted_action_mse(prediction, target, weights)

    torch.testing.assert_close(actual, ((prediction - target) ** 2).mean())


def test_action_loss_weights_broadcast_and_preserve_weight_normalized_scale() -> None:
    prediction = torch.zeros((2, 3, 4))
    target = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4).expand_as(prediction)
    weights = _validate_loss_dim_weights([1.0, 1.0, 1.0, 2.0], action_dim=4)

    loss = _weighted_action_mse(prediction, target, weights)

    expected = torch.tensor((1.0 + 4.0 + 9.0 + 2.0 * 16.0) / 5.0)
    torch.testing.assert_close(loss, expected)


def test_yaw_weight_doubles_gradient_relative_to_equal_error_position_dimension() -> None:
    prediction = torch.zeros((2, 3, 4), requires_grad=True)
    target = torch.ones_like(prediction)
    weights = _validate_loss_dim_weights([1.0, 1.0, 1.0, 2.0], action_dim=4)

    _weighted_action_mse(prediction, target, weights).backward()

    torch.testing.assert_close(prediction.grad[..., 3], prediction.grad[..., 0] * 2.0)


@pytest.mark.parametrize(
    ("raw_weights", "message"),
    [
        ([1.0, 1.0, 1.0], "exactly action_dim=4"),
        ([1.0, -1.0, 1.0, 1.0], "non-negative"),
        ([1.0, float("nan"), 1.0, 1.0], "finite"),
        ([1.0, float("inf"), 1.0, 1.0], "finite"),
        ([0.0, 0.0, 0.0, 0.0], "positive sum"),
    ],
)
def test_invalid_action_loss_weights_raise_clear_errors(raw_weights, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_loss_dim_weights(raw_weights, action_dim=4)


@pytest.mark.parametrize(
    ("configured_weights", "expected_weights"),
    [
        (None, [1.0, 1.0, 1.0, 1.0]),
        ([1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 2.0]),
    ],
)
def test_action_loss_weights_default_and_non_persistent_buffer(
    monkeypatch,
    configured_weights,
    expected_weights,
) -> None:
    class _FakeDiT(torch.nn.Module):
        def __init__(self, **_kwargs) -> None:
            super().__init__()
            self.config = SimpleNamespace(output_dim=8)

    monkeypatch.setattr(action_header_module, "DiT", _FakeDiT)
    config = _flowmatching_action_head_config(loss_dim_weights=configured_weights)

    head = FlowmatchingActionHead(config)

    torch.testing.assert_close(head.loss_dim_weights, torch.tensor(expected_weights))
    assert "loss_dim_weights" not in head.state_dict()


def test_qwen35_openfly_config_enables_yaw_weighting() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml"
    action_model_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["framework"]["action_model"]

    assert action_model_config["action_dim"] == 4
    assert action_model_config["loss_dim_weights"] == [1.0, 1.0, 1.0, 2.0]
