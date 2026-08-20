from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import torch

from NavVLAeval.common.model.model_wrappers import StarVLAEvalModel, _derive_prediction_seed


class FakeInnerModel:
    def __init__(self) -> None:
        self.calls = []

    def predict_action(self, *, examples, **kwargs):
        self.calls.append((examples, kwargs))
        return {"normalized_actions": np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32)}


def test_starvla_eval_model_wraps_single_example_predict_action() -> None:
    inner = FakeInnerModel()
    model = StarVLAEvalModel.from_loaded_model(inner, do_sample=False, num_ddim_steps=4)

    output = model.predict_action({"image": "frame"})

    assert output["normalized_actions"].shape == (1, 4)
    assert inner.calls == [([{"image": "frame"}], {"do_sample": False, "num_ddim_steps": 4})]


def test_prediction_seed_is_stable_and_episode_step_specific() -> None:
    first = _derive_prediction_seed(42, {"metadata": {"episode_uid": "episode-a", "frame_index": 8}})
    assert first == _derive_prediction_seed(42, {"metadata": {"episode_uid": "episode-a", "frame_index": 8}})
    assert first != _derive_prediction_seed(42, {"metadata": {"episode_uid": "episode-a", "frame_index": 16}})
    assert first != _derive_prediction_seed(42, {"metadata": {"episode_uid": "episode-b", "frame_index": 8}})


def test_starvla_eval_model_reseeds_flow_sampling_per_episode_frame() -> None:
    class NoiseModel:
        def predict_action(self, *, examples, **kwargs):
            del examples, kwargs
            return {"normalized_actions": torch.randn((1, 2, 4)).numpy()}

    model = StarVLAEvalModel.from_loaded_model(NoiseModel(), inference_seed=42)
    first = model.predict_action({"metadata": {"episode_uid": "episode-a", "frame_index": 8}})
    repeated = model.predict_action({"metadata": {"episode_uid": "episode-a", "frame_index": 8}})
    next_frame = model.predict_action({"metadata": {"episode_uid": "episode-a", "frame_index": 16}})

    np.testing.assert_array_equal(first["normalized_actions"], repeated["normalized_actions"])
    assert not np.array_equal(first["normalized_actions"], next_frame["normalized_actions"])
    assert first["metadata"]["inference_seed"] == repeated["metadata"]["inference_seed"]


def test_starvla_eval_model_passes_inference_tvi_mask_probability(monkeypatch, tmp_path: Path) -> None:
    inner = FakeInnerModel()
    monkeypatch.setattr(StarVLAEvalModel, "_load_model", lambda self: inner)
    model = StarVLAEvalModel(
        checkpoint=tmp_path / "checkpoint.pt",
        repo_root=tmp_path,
        inference_tvi_mask_probability=1.0,
    )

    model.predict_action({"image": "frame"})

    assert inner.calls == [([{"image": "frame"}], {"tvi_mask_probability": 1.0})]


def test_starvla_eval_model_resolves_final_model_directory_before_loading(monkeypatch, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "final_model"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "pytorch_model.pt"
    checkpoint_file.write_bytes(b"checkpoint")
    calls = []

    class FakeLoadedModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True

    class FakeBaseFramework:
        @staticmethod
        def from_pretrained(path: str, **kwargs):
            calls.append((Path(path), kwargs))
            return FakeLoadedModel()

    fake_framework_module = types.ModuleType("starVLA.model.framework.base_framework")
    fake_framework_module.baseframework = FakeBaseFramework
    monkeypatch.setitem(sys.modules, "starVLA.model.framework.base_framework", fake_framework_module)

    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    StarVLAEvalModel(
        checkpoint=checkpoint_dir,
        config_overrides={
            "framework": {
                "qwenvl": {"base_vlm": "/data1/wxwu/model/Qwen3-VL-2B-Instruct"},
                "navvla": {"history_visual_tokens": 32, "current_visual_tokens": 128},
            }
        },
    )

    assert calls == [
        (
            checkpoint_file.resolve(),
            {
                "config_overrides": {
                    "framework": {
                        "qwenvl": {"base_vlm": "/data1/wxwu/model/Qwen3-VL-2B-Instruct"},
                        "navvla": {"history_visual_tokens": 32, "current_visual_tokens": 128},
                    }
                }
            },
        )
    ]
