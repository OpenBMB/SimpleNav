from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from NavVLAeval.common.data.runtime_history import OnlineHistoryConfig, build_online_navvla_history_sample
from NavVLAeval.common.data.runtime_dataset import OnlineNavVLARuntimeDatasetAdapter
from NavVLAeval.common.model.action_codec import ActionCodec
from NavVLAeval.common.types import ActionPrediction, EnvironmentStepResult, EpisodeHistory, Pose4D
from starVLA.model.framework.VLM4A import navvla_cpm as navvla_cpm_module
from starVLA.model.framework.VLM4A.navvla_cpm import NavVLA_CPM
from starVLA.model.modules.bats import select_bats_history
from starVLA.model.modules.long_memory import LongMemoryTokenAggregator
from starVLA.model.modules import vlm as vlm_module


class _FakeMiniCPMVLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.IMAGE_TOKEN_INDEX = 999
        self.action_placeholder_token_id = 777
        self.model = _FakeMiniCPMBackbone()
        self.seen_images = None
        self.seen_instructions = None
        self.seen_suffixes = None

    def build_action_placeholder_suffix(self, num_placeholders: int) -> str:
        return "A" * int(num_placeholders)

    def build_qwenvl_inputs(self, images, instructions, action_suffixes=None, **_kwargs):
        self.seen_images = images
        self.seen_instructions = instructions
        self.seen_suffixes = action_suffixes
        batch_size = len(instructions)
        current_visual_tokens = 2
        num_placeholders = len(action_suffixes[0])
        row = [998] + [self.IMAGE_TOKEN_INDEX] * current_visual_tokens + [10] + [self.action_placeholder_token_id] * num_placeholders
        return {
            "input_ids": torch.tensor(row, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1),
            "attention_mask": torch.ones((batch_size, len(row)), dtype=torch.long),
        }

    def gather_action_placeholder_hidden_states(self, hidden_states, input_ids, *, num_placeholders: int, **_kwargs):
        del input_ids
        return hidden_states[:, -int(num_placeholders) :, :] + 10.0

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones((batch_size, seq_len, 16), dtype=torch.float32)
        return SimpleNamespace(hidden_states=[hidden], last_hidden_state=hidden)


class _FakeMiniCPMBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, vision_start_token_id=998)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.embedding = torch.nn.Embedding(1200, 16)

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, **_kwargs):
        return torch.ones((2, 16), dtype=torch.float32)

    def forward(self, *, inputs_embeds, attention_mask=None, **_kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds, hidden_states=[inputs_embeds])


class _MiniCPMNewlineTokenizer:
    unk_token_id = 248077

    def __init__(self) -> None:
        self.converted_tokens = []
        self.encoded_text = []

    def convert_tokens_to_ids(self, token: str) -> int:
        self.converted_tokens.append(token)
        return {"</image>": 248079, "\n": self.unk_token_id}[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encoded_text.append((text, add_special_tokens))
        return [198]


class _FakeActionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, vl_embs, actions, state=None, action_padding_mask=None):
        self.calls.append(
            {
                "vl_embs": vl_embs.detach().clone(),
                "actions": actions.detach().clone(),
                "state": None if state is None else state.detach().clone(),
                "action_padding_mask": None
                if action_padding_mask is None
                else action_padding_mask.detach().clone(),
            }
        )
        return vl_embs.sum() * 0.0 + actions.sum() * 0.0 + torch.tensor(1.0, device=vl_embs.device)


def _task6_cpm_config(*, tvi_mode: str, long_memory_visual_tokens: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "tvi_mode": tvi_mode,
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": long_memory_visual_tokens,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "cached_history_online_current"}},
        }
    )


def test_minicpm_single_image_suffix_uses_encoded_newline_token() -> None:
    tokenizer = _MiniCPMNewlineTokenizer()
    vlm_interface = SimpleNamespace(processor=SimpleNamespace(tokenizer=tokenizer))
    input_ids = torch.tensor([[998, 999, 999, 248079, 198, 55]], dtype=torch.long)

    suffix = navvla_cpm_module._minicpm_image_slot_suffix_token_ids(
        input_ids,
        image_token_id=999,
        vlm_interface=vlm_interface,
    )

    assert suffix == [248079, 198]
    assert tokenizer.converted_tokens == ["</image>"]
    assert tokenizer.encoded_text == [("\n", False)]


class _Float32PredictActionHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(16, 4)
        self.seen_vl_dtype = None
        self.seen_state_dtype = None

    def predict_action(self, vl_embs, state=None):
        self.seen_vl_dtype = vl_embs.dtype
        self.seen_state_dtype = None if state is None else state.dtype
        pooled = vl_embs.mean(dim=1)
        action = self.proj(pooled).unsqueeze(1).repeat(1, 2, 1)
        return action


def test_navvla_cpm_predict_action_casts_bfloat16_backbone_hidden_to_action_head_dtype() -> None:
    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    action_head = _Float32PredictActionHead()
    model.action_model = action_head
    model.action_horizon = 2
    model.action_dim = 4

    def _fake_forward_vlm_for_action(samples, **_kwargs):
        return torch.ones((len(samples), 2, 16), dtype=torch.bfloat16), []

    model._forward_vlm_for_action = _fake_forward_vlm_for_action
    sample = {
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
    }

    output = model.predict_action([sample])

    assert output["normalized_actions"].shape == (1, 2, 4)
    assert action_head.seen_vl_dtype == torch.float32
    assert action_head.seen_state_dtype == torch.float32


def test_navvla_cpm_online_history_images_are_not_treated_as_cached_blocks() -> None:
    history_image = Image.new("RGB", (8, 8), color=(20, 0, 0))
    current_image = Image.new("RGB", (8, 8), color=(40, 0, 0))
    sample = {
        "images": {"front": current_image},
        "history_images": {"front": [history_image]},
        "current_tvi": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "history_tvi": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "history_mask": np.asarray([True], dtype=bool),
        "metadata": {
            "history_blocks": [
                {
                    "camera_name": "front",
                    "step_index": 0,
                    "frame_index": 0,
                }
            ]
        },
    }

    online_images, blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(sample, required_cameras=["front"])

    assert online_images == [history_image, current_image]
    assert [block["is_history"] for block in blocks] == [True, False]
    assert [block["is_cached_history"] for block in blocks] == [False, False]


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, (0.0, 0.0)),
        (5, (0.5, 0.0)),
        (15, (0.5, 0.15)),
        (100, (0.5, 0.15)),
    ],
)
def test_history_augmentation_probabilities_follow_two_stage_warmup(step, expected) -> None:
    config = navvla_cpm_module.HistoryAugmentationConfig(
        enabled=True,
        shuffle_target_probability=0.5,
        shuffle_warmup_end_ratio=0.05,
        tvi_mask_target_probability=0.15,
        tvi_mask_warmup_start_ratio=0.05,
        tvi_mask_warmup_end_ratio=0.15,
    )

    actual = navvla_cpm_module.history_augmentation_probabilities(
        config,
        training_step=step,
        total_training_steps=100,
    )

    assert actual == pytest.approx(expected)


def test_history_augmentation_disabled_does_not_require_training_progress() -> None:
    config = navvla_cpm_module.HistoryAugmentationConfig(enabled=False)

    assert navvla_cpm_module.history_augmentation_probabilities(config) == (0.0, 0.0)


def test_history_augmentation_defaults_reach_target_probabilities_after_warmup() -> None:
    config = navvla_cpm_module.HistoryAugmentationConfig(enabled=True)

    assert navvla_cpm_module.history_augmentation_probabilities(
        config,
        training_step=100,
        total_training_steps=100,
    ) == pytest.approx((0.3, 0.1))


def test_history_augmentation_zero_length_warmups_jump_at_boundaries() -> None:
    config = navvla_cpm_module.HistoryAugmentationConfig(
        enabled=True,
        shuffle_target_probability=0.5,
        shuffle_warmup_end_ratio=0.0,
        tvi_mask_target_probability=0.15,
        tvi_mask_warmup_start_ratio=0.0,
        tvi_mask_warmup_end_ratio=0.0,
    )

    assert navvla_cpm_module.history_augmentation_probabilities(
        config,
        training_step=0,
        total_training_steps=100,
    ) == pytest.approx((0.5, 0.15))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shuffle_target_probability": -0.1},
        {"shuffle_target_probability": 1.1},
        {"tvi_mask_target_probability": float("nan")},
        {"shuffle_warmup_end_ratio": 0.6, "tvi_mask_warmup_start_ratio": 0.5},
        {"tvi_mask_warmup_start_ratio": 0.6, "tvi_mask_warmup_end_ratio": 0.5},
    ],
)
def test_history_augmentation_config_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError, match="history augmentation"):
        navvla_cpm_module.HistoryAugmentationConfig(enabled=True, **kwargs)


@pytest.mark.parametrize(
    ("training_step", "total_training_steps"),
    [(None, 100), (0, None), (-1, 100), (0, 0)],
)
def test_history_augmentation_enabled_requires_valid_progress(training_step, total_training_steps) -> None:
    config = navvla_cpm_module.HistoryAugmentationConfig(enabled=True)

    with pytest.raises(ValueError, match="training_step|total_training_steps"):
        navvla_cpm_module.history_augmentation_probabilities(
            config,
            training_step=training_step,
            total_training_steps=total_training_steps,
        )


@pytest.mark.parametrize("tvi_dim", [2, 7])
@pytest.mark.parametrize(
    "values",
    [
        [],
        np.asarray([], dtype=np.float64),
        np.zeros((0, 99), dtype=np.float32),
    ],
)
def test_as_numpy_tvi_accepts_empty_sequences_with_configured_width(values, tvi_dim) -> None:
    tvi = navvla_cpm_module._as_numpy_tvi(values, tvi_dim=tvi_dim)

    assert tvi.shape == (0, tvi_dim)
    assert tvi.dtype == np.float32


@pytest.mark.parametrize(
    ("values", "tvi_dim"),
    [
        (np.zeros((7,), dtype=np.float32), 7),
        (np.zeros((1, 2), dtype=np.float32), 7),
    ],
)
def test_as_numpy_tvi_keeps_nonempty_shape_validation_strict(values, tvi_dim) -> None:
    with pytest.raises(ValueError, match=rf"shape \[N, {tvi_dim}\]"):
        navvla_cpm_module._as_numpy_tvi(values, tvi_dim=tvi_dim)


def test_cached_visual_sequence_shuffles_only_complete_history_records() -> None:
    sample = {
        "images": {
            "front": Image.new("RGB", (4, 4), color=(90, 0, 0)),
            "left": Image.new("RGB", (4, 4), color=(91, 0, 0)),
        },
        "current_tvi": np.asarray([[9.0, 0.0], [9.0, 1.0]], dtype=np.float32),
        "history_cached_embeds": np.stack(
            [np.full((2, 4), value, dtype=np.float32) for value in (10, 20, 30, 40)]
        ),
        "history_cached_mask": np.asarray([True, True, True, True]),
        "history_tvi": np.asarray([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        "long_memory_tokens": np.full((1, 1, 4), 5.0, dtype=np.float32),
        "long_memory_tvi": np.asarray([[0.5, 0.0]], dtype=np.float32),
        "metadata": {
            "history_blocks": [
                {"camera_name": "front", "step_index": 0, "frame_index": 10},
                {"camera_name": "left", "step_index": 0, "frame_index": 11},
                {"camera_name": "front", "step_index": 1, "frame_index": 20},
                {"camera_name": "left", "step_index": 1, "frame_index": 21},
            ],
            "long_memory_blocks": [{"camera_name": "front", "step_index": 0}],
            "frame_index": 90,
        },
    }
    generator = torch.Generator().manual_seed(123)
    expected_generator = torch.Generator().manual_seed(123)
    assert torch.rand((), generator=expected_generator).item() < 1.0
    permutation = torch.randperm(4, generator=expected_generator).tolist()

    online_images, blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        sample,
        required_cameras=["front", "left"],
        history_shuffle_probability=1.0,
        generator=generator,
    )

    assert [block.get("long_memory_index") for block in blocks[:1]] == [0]
    history = blocks[1:5]
    assert [block["cached_history_index"] for block in history] == permutation
    assert [block["time"] for block in history] == [sample["history_tvi"][index, 0] for index in permutation]
    assert [block["phi"] for block in history] == [sample["history_tvi"][index, 1] for index in permutation]
    expected_cameras = [sample["metadata"]["history_blocks"][index]["camera_name"] for index in permutation]
    expected_frames = [sample["metadata"]["history_blocks"][index]["frame_index"] for index in permutation]
    assert [block["camera_name"] for block in history] == expected_cameras
    assert [block["frame_index"] for block in history] == expected_frames
    assert [block["camera_name"] for block in blocks[-2:]] == ["front", "left"]
    assert [block["time"] for block in blocks[-2:]] == [9.0, 9.0]
    assert [int(np.asarray(image)[0, 0, 0]) for image in online_images] == [90, 91]


def test_cached_visual_sequence_preserves_seven_dimensional_tvi_through_shuffle() -> None:
    current_tvi = np.asarray(
        [
            [9.0, 90.0, 91.0, 92.0, 0.9, 0.19, 0.29],
            [9.0, 80.0, 81.0, 82.0, 0.8, 0.18, 0.28],
        ],
        dtype=np.float32,
    )
    history_tvi = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.01, 0.02],
            [2.0, 20.0, 21.0, 22.0, 0.2, 0.03, 0.04],
        ],
        dtype=np.float32,
    )
    long_tvi = np.asarray([[0.5, 5.0, 6.0, 7.0, 0.05, 0.005, 0.006]], dtype=np.float32)
    sample = {
        "images": {
            "front": Image.new("RGB", (4, 4), color=(90, 0, 0)),
            "left": Image.new("RGB", (4, 4), color=(80, 0, 0)),
        },
        "current_tvi": current_tvi,
        "history_cached_embeds": np.ones((2, 2, 4), dtype=np.float32),
        "history_cached_mask": np.asarray([True, True]),
        "history_tvi": history_tvi,
        "long_memory_tokens": np.ones((1, 1, 4), dtype=np.float32),
        "long_memory_tvi": long_tvi,
        "metadata": {
            "history_blocks": [
                {"camera_name": "front", "step_index": 0, "frame_index": 10},
                {"camera_name": "left", "step_index": 1, "frame_index": 20},
            ],
            "long_memory_blocks": [{"camera_name": "front", "step_index": 0}],
            "frame_index": 90,
        },
    }
    generator = torch.Generator().manual_seed(17)
    expected_generator = torch.Generator().manual_seed(17)
    assert torch.rand((), generator=expected_generator).item() < 1.0
    permutation = torch.randperm(2, generator=expected_generator).tolist()

    online_images, blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        sample,
        required_cameras=["front", "left"],
        history_shuffle_probability=1.0,
        generator=generator,
        tvi_dim=7,
    )

    expected_tvi = [long_tvi[0], *[history_tvi[index] for index in permutation], *current_tvi]
    assert [block["camera_name"] for block in blocks[-2:]] == ["front", "left"]
    assert [int(np.asarray(image)[0, 0, 0]) for image in online_images] == [90, 80]
    for block, expected in zip(blocks, expected_tvi, strict=True):
        np.testing.assert_array_equal(block["tvi"], expected)
        assert block["time"] == pytest.approx(float(expected[0]))
        assert block["phi"] == pytest.approx(float(expected[4]))


def test_cached_visual_sequence_uses_width_aware_long_memory_and_legacy_current_fallback_rows() -> None:
    image = Image.new("RGB", (4, 4), color=(90, 0, 0))
    sample = {
        "images": {"front": image},
        "current_tvi": np.asarray([[9.0, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6]], dtype=np.float32),
        "long_memory_tokens": np.ones((1, 1, 4), dtype=np.float32),
        "metadata": {
            "long_memory_blocks": [{"camera_name": "front", "step_index": 4}],
            "timestamp": 9.0,
        },
    }

    _images, pose_blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        sample,
        required_cameras=["front"],
        tvi_dim=7,
    )
    legacy_sample = dict(sample)
    legacy_sample.pop("current_tvi")
    _images, legacy_blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        legacy_sample,
        required_cameras=["front"],
    )

    assert [block["tvi"].shape for block in pose_blocks] == [(7,), (7,)]
    np.testing.assert_array_equal(pose_blocks[0]["tvi"], [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(pose_blocks[1]["tvi"], sample["current_tvi"][0])
    assert [block["tvi"].shape for block in legacy_blocks] == [(2,), (2,)]
    np.testing.assert_array_equal(legacy_blocks[1]["tvi"], [9.0, 0.0])


@pytest.mark.parametrize(
    ("present_cameras", "tvi_row_count"),
    [
        (["front", "left"], 1),
        (["front"], 2),
    ],
)
def test_cached_visual_sequence_rejects_current_camera_pose_tvi_count_mismatch(
    present_cameras,
    tvi_row_count,
) -> None:
    sample = {
        "images": {
            camera_name: Image.new("RGB", (4, 4), color=(90, 0, 0))
            for camera_name in present_cameras
        },
        "current_tvi": np.zeros((tvi_row_count, 7), dtype=np.float32),
        "metadata": {},
    }

    with pytest.raises(
        ValueError,
        match=(
            rf"current_tvi row count {tvi_row_count} does not match "
            rf"present current camera count {len(present_cameras)}"
        ),
    ):
        navvla_cpm_module.build_navvla_cached_visual_sequence(
            sample,
            required_cameras=["front", "left"],
            tvi_dim=7,
        )


def test_cached_visual_sequence_maps_current_camera_pose_tvi_in_present_camera_order() -> None:
    current_tvi = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
            [2.0, 20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    sample = {
        "images": {
            "rear": Image.new("RGB", (4, 4), color=(70, 0, 0)),
            "front": Image.new("RGB", (4, 4), color=(90, 0, 0)),
            "left": None,
        },
        "current_tvi": current_tvi,
        "metadata": {},
    }

    _images, blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        sample,
        required_cameras=["front", "left", "rear"],
        tvi_dim=7,
    )

    assert [block["camera_name"] for block in blocks] == ["front", "rear"]
    np.testing.assert_array_equal(blocks[0]["tvi"], current_tvi[0])
    np.testing.assert_array_equal(blocks[1]["tvi"], current_tvi[1])


def test_online_visual_sequence_moves_history_image_with_its_tvi() -> None:
    history_images = [Image.new("RGB", (4, 4), color=(value, 0, 0)) for value in (10, 20)]
    current = Image.new("RGB", (4, 4), color=(90, 0, 0))
    sample = {
        "images": {"front": current},
        "current_tvi": np.asarray([[9.0, 0.0]], dtype=np.float32),
        "history_images": {"front": history_images},
        "history_tvi": np.asarray([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        "history_mask": np.asarray([True, True]),
        "metadata": {
            "history_blocks": [
                {"camera_name": "front", "step_index": 0, "frame_index": 10},
                {"camera_name": "front", "step_index": 1, "frame_index": 20},
            ]
        },
    }
    generator = torch.Generator().manual_seed(7)
    expected_generator = torch.Generator().manual_seed(7)
    torch.rand((), generator=expected_generator)
    permutation = torch.randperm(2, generator=expected_generator).tolist()

    online_images, blocks = navvla_cpm_module.build_navvla_cached_visual_sequence(
        sample,
        required_cameras=["front"],
        history_shuffle_probability=1.0,
        generator=generator,
    )

    assert [int(np.asarray(image)[0, 0, 0]) for image in online_images] == [
        10 if index == 0 else 20 for index in permutation
    ] + [90]
    assert [block["time"] for block in blocks[:2]] == [1.0 if index == 0 else 2.0 for index in permutation]


def test_online_history_sample_uses_in_memory_visual_token_cache() -> None:
    history = EpisodeHistory()
    history.observations.append(
        {
            "image": np.zeros((8, 8, 3), dtype=np.uint8),
            "images": {"front": np.zeros((8, 8, 3), dtype=np.uint8)},
            "navvla_eval": {"frame_index": 0, "timestamp": 0.0},
            "navvla_online_visual_tokens": {
                "front": np.full((2, 16), 3.0, dtype=np.float16),
            },
        }
    )
    current = {
        "image": np.ones((8, 8, 3), dtype=np.uint8),
        "images": {"front": np.ones((8, 8, 3), dtype=np.uint8)},
        "navvla_eval": {"frame_index": 1, "timestamp": 1.0},
    }

    sample = build_online_navvla_history_sample(
        observation=current,
        history=history,
        instruction="go forward",
        state=None,
        config=OnlineHistoryConfig(
            history_policy="recent",
            history_image_frames=1,
            required_cameras=("front",),
            history_visual_tokens=2,
            current_visual_tokens=2,
        ),
    )

    assert sample["history_images"]["front"] == []
    assert sample["history_cached_embeds"].shape == (1, 2, 16)
    assert sample["history_cached_embeds"].dtype == np.float16
    assert sample["history_cached_mask"].tolist() == [True]
    np.testing.assert_allclose(sample["history_cached_embeds"][0], np.full((2, 16), 3.0, dtype=np.float16))


def test_navvla_cpm_predict_action_returns_current_visual_token_cache(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _Float32PredictActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 0,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "online_images"}},
        }
    )
    model = NavVLA_CPM(cfg)
    image = Image.new("RGB", (8, 8))
    sample = {
        "images": {"front": image},
        "current_tvi": np.asarray([[2.0, 0.0]], dtype=np.float32),
        "history_tvi": np.zeros((0, 2), dtype=np.float32),
        "history_mask": np.zeros((0,), dtype=bool),
        "lang": "go forward",
        "platform_text": "uav",
        "action": np.zeros((2, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((2,), dtype=bool),
        "metadata": {"frame_index": 2},
    }

    output = model.predict_action([sample])

    records = output["metadata"]["online_current_visual_tokens"]
    assert len(records) == 1
    assert records[0]["camera_name"] == "front"
    assert records[0]["frame_index"] == 2
    assert records[0]["tokens"].shape == (2, 16)
    assert records[0]["tokens"].dtype == np.float16
    assert fake_vlm.seen_images == [[image]]


def test_navvla_cpm_encodes_intermediate_images_with_history_token_budget(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _Float32PredictActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 4,
                    "long_memory_visual_tokens": 0,
                    "current_visual_tokens": 64,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "cached_history_online_current"}},
        }
    )
    model = NavVLA_CPM(cfg)
    images = [Image.new("RGB", (8, 8), color=(value, 0, 0)) for value in (10, 20)]
    features = [
        torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16),
        torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16) + 1000,
    ]
    monkeypatch.setattr(
        model,
        "_encode_minicpm_current_image_features",
        lambda _inputs, *, online_block_count: features[:online_block_count],
    )

    cached = model.encode_history_images(images)

    assert fake_vlm.seen_images == [images]
    assert len(cached) == 2
    assert [tokens.shape for tokens in cached] == [(4, 16), (4, 16)]
    expected = [
        navvla_cpm_module.pool_minicpm_visual_tokens_to_count(feature, target_tokens=4).numpy()
        for feature in features
    ]
    for actual, target in zip(cached, expected, strict=True):
        np.testing.assert_allclose(actual, target.astype(np.float16), rtol=1e-3, atol=1e-3)


def test_navvla_cpm_predict_action_returns_incremental_long_memory_state(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _Float32PredictActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 1,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "online_images"}},
        }
    )
    model = NavVLA_CPM(cfg)
    image = Image.new("RGB", (8, 8))
    update_tokens = np.arange(4 * 16, dtype=np.float32).reshape(1, 4, 16)
    sample = {
        "images": {"front": image},
        "current_tvi": np.asarray([[2.0, 0.0]], dtype=np.float32),
        "history_tvi": np.zeros((0, 2), dtype=np.float32),
        "history_mask": np.zeros((0,), dtype=bool),
        "online_long_memory_update_tokens": update_tokens,
        "online_long_memory_update_tvi": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "online_long_memory_update_mask": np.asarray([True], dtype=bool),
        "lang": "go forward",
        "platform_text": "uav",
        "action": np.zeros((2, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((2,), dtype=bool),
        "metadata": {
            "frame_index": 2,
            "required_cameras": ["front"],
            "online_long_memory_update_frame_index": 0,
            "online_long_memory_update_blocks": [
                {"camera_name": "front", "frame_index": 0, "step_index": 0}
            ],
        },
    }

    output = model.predict_action([sample])

    updates = output["metadata"]["online_long_memory_updates"]
    assert len(updates) == 1
    assert updates[0]["frame_index"] == 0
    assert updates[0]["tokens"].shape == (1, 1, 16)
    assert updates[0]["tvi"].shape == (1, 2)
    assert updates[0]["blocks"][0]["camera_name"] == "front"


def test_action_codec_preserves_model_metadata_for_online_visual_cache() -> None:
    codec = ActionCodec(
        {
            "scale": [1.0, 1.0, 1.0, 1.0],
            "normalization_modes": ["scale", "scale", "scale", "scale"],
        },
        framework_name="navvla_cpm",
    )
    cache_record = {
        "camera_name": "front",
        "frame_index": 2,
        "tokens": np.ones((2, 16), dtype=np.float16),
    }

    prediction = codec.decode(
        {
            "normalized_actions": np.zeros((2, 4), dtype=np.float32),
            "metadata": {"online_current_visual_tokens": [cache_record]},
        }
    )

    assert prediction.metadata["framework_name"] == "navvla_cpm"
    assert prediction.metadata["online_current_visual_tokens"][0]["camera_name"] == "front"
    np.testing.assert_allclose(prediction.metadata["online_current_visual_tokens"][0]["tokens"], cache_record["tokens"])


def test_worker_attaches_model_visual_cache_to_current_observation() -> None:
    from NavVLAeval.common.runner.worker import _attach_online_visual_token_cache

    tokens = np.ones((4, 16), dtype=np.float16)
    prediction = ActionPrediction(
        normalized_actions=np.zeros((1, 4), dtype=np.float32),
        raw_actions=np.zeros((1, 4), dtype=np.float32),
        metadata={
            "online_current_visual_tokens": [
                {"camera_name": "front", "frame_index": 2, "tokens": tokens}
            ]
        },
    )

    prepared = _attach_online_visual_token_cache({"image": np.zeros((4, 4, 3), dtype=np.uint8)}, prediction)

    np.testing.assert_array_equal(prepared["navvla_online_visual_tokens"]["front"], tokens)


def test_online_runtime_history_update_keeps_cached_current_and_every_action_observation() -> None:
    adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_update_mode="action_observations",
    )
    pre_observation = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "navvla_online_visual_tokens": {"front": np.ones((2, 16), dtype=np.float16)},
    }
    post_observation = {"image": np.ones((8, 8, 3), dtype=np.uint8)}
    step_result = EnvironmentStepResult(
        next_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        observation=post_observation,
        data_done=False,
    )

    action_observations = [
        {"image": np.full((8, 8, 3), 2, dtype=np.uint8)},
        {"image": np.full((8, 8, 3), 3, dtype=np.uint8)},
        {"image": np.full((8, 8, 3), 4, dtype=np.uint8)},
    ]

    updates = adapter.history_observations_for_update(
        pre_observation=pre_observation,
        post_observation=post_observation,
        step_result=step_result,
        action_observations=action_observations,
    )

    assert updates[0] is pre_observation
    assert updates[1] is action_observations[0]
    assert updates[2] is action_observations[1]
    assert updates[3] is action_observations[2]
    assert len(updates) == 4


def test_worker_encodes_every_action_observation_into_four_token_cache() -> None:
    from NavVLAeval.common.runner.worker import _attach_history_visual_token_cache

    class _HistoryEncoder:
        def __init__(self) -> None:
            self.seen_images = None

        def encode_history_images(self, images):
            self.seen_images = list(images)
            return [
                np.full((4, 16), index + 2, dtype=np.float16)
                for index in range(len(images))
            ]

    adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_selection="recent",
        history_image_frames=4,
        history_update_mode="action_observations",
    )
    pre_observation = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "navvla_eval": {"frame_index": 0, "timestamp": 0.0},
        "navvla_online_visual_tokens": {"front": np.ones((4, 16), dtype=np.float16)},
    }
    final_observation = {
        "image": np.full((8, 8, 3), 4, dtype=np.uint8),
        "navvla_eval": {"frame_index": 3, "timestamp": 3.0},
    }
    action_observations = [
        {
            "image": np.full((8, 8, 3), value, dtype=np.uint8),
            "navvla_eval": {"frame_index": index, "timestamp": float(index)},
        }
        for index, value in enumerate((2, 3, 4), start=1)
    ]
    step_result = EnvironmentStepResult(
        next_pose=Pose4D(0.0, 0.0, 0.0, 0.0),
        observation=final_observation,
        data_done=False,
    )
    update_observations = adapter.history_observations_for_update(
        pre_observation=pre_observation,
        post_observation=final_observation,
        step_result=step_result,
        action_observations=action_observations,
    )
    encoder = _HistoryEncoder()

    cached_updates = _attach_history_visual_token_cache(
        model=encoder,
        runtime_dataset=adapter,
        observations=update_observations,
        instruction="go",
    )

    assert [int(np.asarray(image)[0, 0, 0]) for image in encoder.seen_images] == [2, 3, 4]
    np.testing.assert_array_equal(
        cached_updates[0]["navvla_online_visual_tokens"]["front"],
        np.ones((4, 16), dtype=np.float16),
    )
    np.testing.assert_array_equal(
        cached_updates[1]["navvla_online_visual_tokens"]["front"],
        np.full((4, 16), 2, dtype=np.float16),
    )
    np.testing.assert_array_equal(
        cached_updates[2]["navvla_online_visual_tokens"]["front"],
        np.full((4, 16), 3, dtype=np.float16),
    )
    np.testing.assert_array_equal(
        cached_updates[3]["navvla_online_visual_tokens"]["front"],
        np.full((4, 16), 4, dtype=np.float16),
    )

    history = EpisodeHistory()
    action = np.zeros((1, 4), dtype=np.float32)
    prediction = ActionPrediction(normalized_actions=action, raw_actions=action)
    for observation in cached_updates:
        adapter.update_history(history=history, observation=observation, prediction=prediction, instruction="go")
    assert len(history.observations) == 4
    sample = adapter.build_example(
        observation=final_observation,
        history=history,
        instruction="go",
    )

    assert sample["history_cached_embeds"].shape == (3, 4, 16)
    assert sample["history_images"]["front"] == []

    refreshed_current = dict(final_observation)
    refreshed_current["navvla_online_visual_tokens"] = {
        "front": np.full((4, 16), 9, dtype=np.float16)
    }
    adapter.update_history(
        history=history,
        observation=refreshed_current,
        prediction=prediction,
        instruction="go",
    )

    assert len(history.observations) == 4
    np.testing.assert_array_equal(
        history.observations[-1]["navvla_online_visual_tokens"]["front"],
        np.full((4, 16), 9, dtype=np.float16),
    )


def test_navvla_cpm_forward_consumes_memory_tvi_and_action_mask(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _FakeActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 1,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "cached_history_online_current"}},
        }
    )
    model = NavVLA_CPM(cfg)
    image = Image.new("RGB", (8, 8))
    sample = {
        "images": {"front": image},
        "current_tvi": np.asarray([[2.0, 0.0]], dtype=np.float32),
        "history_cached_embeds": np.ones((2, 2, 16), dtype=np.float32),
        "history_cached_mask": np.asarray([True, True], dtype=bool),
        "history_tvi": np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "long_memory_tokens": np.full((1, 1, 16), 2.0, dtype=np.float32),
        "long_memory_tvi": np.asarray([[0.5, 0.0]], dtype=np.float32),
        "lang": "go forward",
        "platform_text": "uav",
        "action": np.asarray([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]], dtype=np.float32),
        "action_padding_mask": np.asarray([False, True], dtype=bool),
        "metadata": {"history_blocks": [{"camera_name": "front", "step_index": 0}]},
    }

    output = model([sample])

    assert output["loss"].item() == 1.0
    assert fake_vlm.seen_images == [[image]]
    assert fake_vlm.seen_instructions == ["uav go forward"]
    assert fake_vlm.seen_suffixes == ["AA"]
    call = fake_action.calls[0]
    assert call["vl_embs"].shape == (1, 2, 16)
    assert call["action_padding_mask"].tolist() == [[False, True]]
    assert call["actions"][0, 1].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_navvla_cpm_forward_backbone_passes_exact_seven_dimensional_tvi_and_one_prefix_per_block(
    monkeypatch,
) -> None:
    class _RecordingTVIEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.seen = values.detach().clone()
            return torch.zeros((values.shape[0], 16), device=values.device, dtype=values.dtype)

    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    model.minicpm_vl_interface = _FakeMiniCPMVLM()
    model.minicpm_vl_interface.model.embedding.to(dtype=torch.bfloat16)
    model.tvi_dim = 7
    model.tvi_embedding = _RecordingTVIEmbedding()
    rows = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
            [2.0, 20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    blocks = [
        {"is_history": True, "tvi": rows[0]},
        {"is_history": False, "tvi": rows[1]},
    ]
    minicpm_inputs = {
        "input_ids": torch.tensor([[998, 999, 999, 10, 998, 999, 999, 11]], dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }
    monkeypatch.setattr(
        model,
        "_fuse_image_token_embeddings",
        lambda *_args, **_kwargs: (torch.ones((4, 16), dtype=torch.bfloat16), []),
    )

    outputs, _records = model._forward_backbone(minicpm_inputs, blocks)

    torch.testing.assert_close(model.tvi_embedding.seen, torch.from_numpy(rows))
    assert model.tvi_embedding.seen.dtype == torch.float32
    assert minicpm_inputs["input_ids"].shape == (1, 10)
    assert torch.count_nonzero(minicpm_inputs["input_ids"] == 0).item() == len(blocks)
    assert outputs.last_hidden_state.shape == (1, 10, 16)


def test_navvla_cpm_forward_backbone_batches_tvi_tensor_transfer(monkeypatch) -> None:
    class _RecordingTVIEmbedding(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.zeros((values.shape[0], 16), device=values.device, dtype=values.dtype)

    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    model.minicpm_vl_interface = _FakeMiniCPMVLM()
    model.tvi_dim = 7
    model.tvi_embedding = _RecordingTVIEmbedding()
    rows = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
            [2.0, 20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    blocks = [
        {"is_history": True, "tvi": rows[0]},
        {"is_history": False, "tvi": rows[1]},
    ]
    minicpm_inputs = {
        "input_ids": torch.tensor([[998, 999, 999, 10, 998, 999, 999, 11]], dtype=torch.long),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }
    monkeypatch.setattr(
        model,
        "_fuse_image_token_embeddings",
        lambda *_args, **_kwargs: (torch.ones((4, 16), dtype=torch.float32), []),
    )
    original_as_tensor = torch.as_tensor
    transfer_shapes = []

    def recording_as_tensor(values, *args, **kwargs):
        if isinstance(values, np.ndarray) and values.size and int(values.shape[-1]) == 7:
            transfer_shapes.append(tuple(values.shape))
        return original_as_tensor(values, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", recording_as_tensor)

    model._forward_backbone(minicpm_inputs, blocks)

    assert transfer_shapes == [(2, 7)]


def test_navvla_cpm_forward_passes_configured_tvi_dim_to_visual_builder(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _FakeActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    original_builder = navvla_cpm_module.build_navvla_cached_visual_sequence
    seen_tvi_dims = []

    def recording_builder(*args, **kwargs):
        seen_tvi_dims.append(kwargs.get("tvi_dim"))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(navvla_cpm_module, "build_navvla_cached_visual_sequence", recording_builder)
    model = NavVLA_CPM(
        _task6_cpm_config(tvi_mode="time_camera_pose", long_memory_visual_tokens=0)
    )
    sample = {
        "images": {"front": Image.new("RGB", (8, 8))},
        "current_tvi": np.asarray([[2.0, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6]], dtype=np.float32),
        "history_tvi": np.zeros((0, 7), dtype=np.float32),
        "history_mask": np.zeros((0,), dtype=bool),
        "lang": "go forward",
        "platform_text": "uav",
        "action": np.zeros((2, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((2,), dtype=bool),
        "metadata": {"required_cameras": ["front"]},
    }

    output = model([sample])

    assert output["loss"].item() == 1.0
    assert seen_tvi_dims == [7]


def test_navvla_cpm_padded_long_memory_sources_ignore_padding(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _FakeActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 1,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "cached_history_online_current"}},
        }
    )
    model = NavVLA_CPM(cfg)
    sample = {
        "long_memory_source_tokens": np.ones((5, 4, 16), dtype=np.float32),
        "long_memory_source_mask": np.asarray([True, True, False, False, False], dtype=bool),
        "long_memory_source_tvi": np.zeros((5, 2), dtype=np.float32),
        "metadata": {
            "long_memory_blocks": [
                {"camera_name": "front", "step_index": 0},
                {"camera_name": "front", "step_index": 1},
            ],
            "required_cameras": ["front"],
        },
    }

    model._attach_long_memory_tokens([sample])

    assert sample["long_memory_tokens"].shape == (1, 1, 16)
    assert sample["metadata"]["long_memory_blocks"] == [
        {"step_index": 1, "camera_name": "front", "source_block_count": 2}
    ]


def test_navvla_cpm_empty_long_memory_keeps_aggregator_in_backward(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _FakeActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 1,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "repeated_diffusion_steps": 1,
                },
            },
            "datasets": {"vla_data": {"visual_token_mode": "cached_history_online_current"}},
        }
    )
    model = NavVLA_CPM(cfg)
    image = Image.new("RGB", (8, 8))
    sample = {
        "images": {"front": image},
        "current_tvi": np.asarray([[2.0, 0.0]], dtype=np.float32),
        "history_tvi": np.zeros((0, 2), dtype=np.float32),
        "history_mask": np.zeros((0,), dtype=bool),
        "long_memory_source_tokens": np.zeros((0, 4, 16), dtype=np.float32),
        "long_memory_source_mask": np.zeros((0,), dtype=bool),
        "long_memory_source_tvi": np.zeros((0, 2), dtype=np.float32),
        "lang": "go forward",
        "platform_text": "uav",
        "action": np.asarray([[0.1, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0]], dtype=np.float32),
        "action_padding_mask": np.asarray([False, True], dtype=bool),
        "metadata": {"history_blocks": [], "long_memory_blocks": [], "required_cameras": ["front"]},
    }

    output = model([sample])
    output["loss"].backward()

    grad = model.long_memory_aggregator.projection.grad
    assert grad is not None
    assert grad.shape == model.long_memory_aggregator.projection.shape
    assert torch.count_nonzero(grad).item() == 0


def test_long_memory_aggregator_weighted_sum_before_projection(monkeypatch) -> None:
    aggregator = LongMemoryTokenAggregator(
        source_visual_tokens=4,
        long_memory_visual_tokens=8,
        decay=0.5,
        update_weight=0.25,
    )
    call_shapes = []
    original_project = aggregator.project_source_tokens

    def record_project_shape(source_tokens: torch.Tensor) -> torch.Tensor:
        call_shapes.append(tuple(source_tokens.shape))
        return original_project(source_tokens)

    monkeypatch.setattr(aggregator, "project_source_tokens", record_project_shape)
    source_tokens = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    source_tvi = torch.tensor([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    source_mask = torch.tensor([True, True, True])
    source_blocks = [
        {"step_index": 0, "camera_name": "front"},
        {"step_index": 1, "camera_name": "front"},
        {"step_index": 2, "camera_name": "front"},
    ]

    tokens, tvi, blocks = aggregator.aggregate_sample(
        source_tokens=source_tokens,
        source_tvi=source_tvi,
        source_mask=source_mask,
        source_blocks=source_blocks,
        required_cameras=["front"],
    )

    weights = torch.tensor([0.25, 0.125, 0.25])
    expected_source = (source_tokens * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True)
    torch.testing.assert_close(tokens, original_project(expected_source))
    torch.testing.assert_close(tvi, (source_tvi * weights.view(-1, 1)).sum(dim=0, keepdim=True))
    assert blocks == [{"step_index": 2, "camera_name": "front", "source_block_count": 3}]
    assert call_shapes == [(1, 4, 2)]


def test_long_memory_aggregator_preserves_seven_dimensional_weighted_tvi_for_aggregate_and_update() -> None:
    aggregator = LongMemoryTokenAggregator(
        source_visual_tokens=2,
        long_memory_visual_tokens=3,
        decay=0.5,
        update_weight=0.25,
        tvi_dim=7,
    )
    source_tokens = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    source_tvi = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    source_blocks = [{"camera_name": "front", "step_index": index} for index in range(3)]
    source_mask = torch.ones(3, dtype=torch.bool)

    aggregate_tokens, aggregate_tvi, aggregate_blocks = aggregator.aggregate_sample(
        source_tokens=source_tokens,
        source_tvi=source_tvi,
        source_mask=source_mask,
        source_blocks=source_blocks,
        required_cameras=["front"],
    )
    update_tokens = None
    update_tvi = None
    update_blocks: list[dict] = []
    for index in range(3):
        update_tokens, update_tvi, update_blocks = aggregator.update_state(
            previous_tokens=update_tokens,
            previous_tvi=update_tvi,
            previous_blocks=update_blocks,
            source_tokens=source_tokens[index : index + 1],
            source_tvi=source_tvi[index : index + 1],
            source_mask=source_mask[index : index + 1],
            source_blocks=source_blocks[index : index + 1],
            required_cameras=["front"],
        )

    weights = torch.tensor([0.25, 0.125, 0.25])
    expected_tvi = (source_tvi * weights.view(-1, 1)).sum(dim=0, keepdim=True)
    assert aggregate_tvi.shape == (1, 7)
    assert update_tvi.shape == (1, 7)
    torch.testing.assert_close(aggregate_tvi, expected_tvi)
    torch.testing.assert_close(update_tvi, expected_tvi)
    torch.testing.assert_close(update_tokens, aggregate_tokens)
    assert update_blocks == aggregate_blocks


def test_long_memory_aggregator_empty_outputs_keep_configured_tvi_width() -> None:
    aggregator = LongMemoryTokenAggregator(source_visual_tokens=2, long_memory_visual_tokens=3, tvi_dim=7)
    source_tokens = torch.zeros((0, 2, 4), dtype=torch.float32)
    source_tvi = torch.zeros((0, 7), dtype=torch.float32)
    source_mask = torch.zeros((0,), dtype=torch.bool)

    aggregate = aggregator.aggregate_sample(
        source_tokens=source_tokens,
        source_tvi=source_tvi,
        source_mask=source_mask,
        source_blocks=[],
        required_cameras=["front"],
    )
    update = aggregator.update_state(
        previous_tokens=None,
        previous_tvi=None,
        previous_blocks=[],
        source_tokens=source_tokens,
        source_tvi=source_tvi,
        source_mask=source_mask,
        source_blocks=[],
        required_cameras=["front"],
    )

    assert aggregate[1].shape == (0, 7)
    assert update[1].shape == (0, 7)


def test_long_memory_aggregator_rejects_wrong_source_tvi_width() -> None:
    aggregator = LongMemoryTokenAggregator(source_visual_tokens=2, long_memory_visual_tokens=3, tvi_dim=7)

    with pytest.raises(ValueError, match=r"source_tvi.*shape \[N, 7\]"):
        aggregator.aggregate_sample(
            source_tokens=torch.zeros((1, 2, 4)),
            source_tvi=torch.zeros((1, 2)),
            source_mask=torch.ones((1,), dtype=torch.bool),
            source_blocks=[{"camera_name": "front"}],
            required_cameras=["front"],
        )


def test_long_memory_aggregator_rejects_wrong_previous_tvi_width() -> None:
    aggregator = LongMemoryTokenAggregator(source_visual_tokens=2, long_memory_visual_tokens=3, tvi_dim=7)

    with pytest.raises(ValueError, match=r"previous_tvi.*shape \[N, 7\]"):
        aggregator.update_state(
            previous_tokens=torch.zeros((1, 3, 4)),
            previous_tvi=torch.zeros((1, 2)),
            previous_blocks=[{"camera_name": "front"}],
            source_tokens=torch.zeros((0, 2, 4)),
            source_tvi=torch.zeros((0, 7)),
            source_mask=torch.zeros((0,), dtype=torch.bool),
            source_blocks=[],
            required_cameras=["front"],
        )


def test_long_memory_aggregator_rejects_nonpositive_tvi_dim() -> None:
    with pytest.raises(ValueError, match="tvi_dim must be positive"):
        LongMemoryTokenAggregator(tvi_dim=0)


def test_navvla_cpm_configures_long_memory_aggregator_for_seven_dimensional_tvi(monkeypatch) -> None:
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: _FakeMiniCPMVLM())
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())

    model = NavVLA_CPM(
        _task6_cpm_config(tvi_mode="time_camera_pose", long_memory_visual_tokens=1)
    )

    assert model.long_memory_aggregator.tvi_dim == 7


def test_navvla_cpm_trims_seven_dimensional_long_memory_source_tvi_by_rows(monkeypatch) -> None:
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: _FakeMiniCPMVLM())
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    model = NavVLA_CPM(
        _task6_cpm_config(tvi_mode="time_camera_pose", long_memory_visual_tokens=1)
    )
    source_tvi = np.asarray(
        [
            [1.0, 10.0, 11.0, 12.0, 0.1, 0.2, 0.3],
            [2.0, 20.0, 21.0, 22.0, 0.4, 0.5, 0.6],
        ],
        dtype=np.float32,
    )
    sample = {
        "long_memory_source_tokens": np.ones((2, 4, 16), dtype=np.float32),
        "long_memory_source_tvi": source_tvi,
        "long_memory_source_mask": np.asarray([True, False], dtype=bool),
        "metadata": {
            "required_cameras": ["front"],
            "long_memory_blocks": [{"camera_name": "front", "step_index": 1}],
        },
    }

    model._attach_long_memory_tokens([sample])

    assert sample["long_memory_tvi"].shape == (1, 7)
    np.testing.assert_array_equal(sample["long_memory_tvi"][0], source_tvi[0])


@pytest.mark.parametrize(
    ("tvi_mode", "expected_tvi"),
    [
        ("time_yaw", np.asarray([[0.0, 1.25]], dtype=np.float32)),
        ("time_camera_pose", np.zeros((1, 7), dtype=np.float32)),
    ],
)
def test_navvla_cpm_missing_long_memory_dummy_tvi_is_mode_aware(
    monkeypatch,
    tvi_mode,
    expected_tvi,
) -> None:
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: _FakeMiniCPMVLM())
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    model = NavVLA_CPM(
        _task6_cpm_config(tvi_mode=tvi_mode, long_memory_visual_tokens=1)
    )
    sample = {
        "long_memory_source_tokens": np.zeros((0, 4, 16), dtype=np.float32),
        "metadata": {
            "required_cameras": ["front"],
            "long_memory_blocks": [],
            "camera": {"front": {"azimuth_rad": 1.25}},
        },
    }

    model._attach_long_memory_tokens([sample])

    np.testing.assert_array_equal(sample["long_memory_tvi"], expected_tvi)


def test_navvla_cpm_online_long_memory_default_tvi_uses_configured_width(monkeypatch) -> None:
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: _FakeMiniCPMVLM())
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    model = NavVLA_CPM(
        _task6_cpm_config(tvi_mode="time_camera_pose", long_memory_visual_tokens=1)
    )
    sample = {
        "online_long_memory_update_tokens": np.ones((1, 4, 16), dtype=np.float32),
        "metadata": {
            "required_cameras": ["front"],
            "long_memory_blocks": [],
            "online_long_memory_update_blocks": [{"camera_name": "front", "step_index": 4}],
            "online_long_memory_update_frame_index": 4,
        },
    }

    updates = model._compute_online_long_memory_updates([sample])

    assert updates[0]["tvi"].shape == (1, 7)
    np.testing.assert_array_equal(updates[0]["tvi"], np.zeros((1, 7), dtype=np.float32))


def test_get_vlm_model_dispatches_minicpm(monkeypatch) -> None:
    import starVLA.model.modules.vlm.MiniCPMV as minicpm_module

    class _FakeInterface:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(minicpm_module, "_MiniCPM_VL_Interface", _FakeInterface)
    cfg = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"}}})

    interface = vlm_module.get_vlm_model(cfg)

    assert isinstance(interface, _FakeInterface)


def test_navvla_cpm_default_config_uses_time_yaw_tvi_mode() -> None:
    from starVLA.model.modules.tvi import TIME_YAW_TVI_MODE

    assert navvla_cpm_module.NavVLACPMDefaultConfig().navvla["tvi_mode"] == TIME_YAW_TVI_MODE


def test_navvla_cpm_uses_default_history_augmentation_targets_when_omitted(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "long_memory_visual_tokens": 0,
                    "action_placeholder_count": 2,
                    "history_augmentation": {"enabled": True},
                },
                "action_model": {"action_horizon": 2, "action_dim": 4, "state_dim": 0},
            }
        }
    )

    model = NavVLA_CPM(cfg)

    assert model.history_augmentation.shuffle_target_probability == pytest.approx(0.3)
    assert model.history_augmentation.tvi_mask_target_probability == pytest.approx(0.1)


def test_navvla_cpm_constructor_uses_history_augmentation_probability_fallbacks(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    monkeypatch.setattr(
        navvla_cpm_module,
        "merge_framework_config",
        lambda _default_config_cls, config: config,
    )
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "long_memory_visual_tokens": 0,
                    "action_placeholder_count": 2,
                    "history_augmentation": {
                        "enabled": True,
                        "shuffle": {},
                        "tvi_mask": {},
                    },
                },
                "action_model": {
                    "action_horizon": 2,
                    "action_dim": 4,
                    "state_dim": 0,
                    "num_target_vision_tokens": 2,
                    "diffusion_model_cfg": {"cross_attention_dim": 16},
                },
            }
        }
    )

    model = NavVLA_CPM(cfg)

    assert model.history_augmentation.shuffle_target_probability == pytest.approx(0.3)
    assert model.history_augmentation.tvi_mask_target_probability == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("configured_mode", "expected_mode", "expected_dim"),
    [
        (None, "time_yaw", 2),
        ("learned_token", "learned_token", 2),
        ("time_camera_pose", "time_camera_pose", 7),
        ("metric_camera_pose", "metric_camera_pose", 7),
    ],
)
def test_navvla_cpm_builds_configured_tvi_embedding(
    monkeypatch,
    configured_mode,
    expected_mode,
    expected_dim,
) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    navvla_config = {"long_memory_visual_tokens": 0, "action_placeholder_count": 2}
    if configured_mode is not None:
        navvla_config["tvi_mode"] = configured_mode
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": navvla_config,
                "action_model": {"action_horizon": 2, "action_dim": 4, "state_dim": 0},
            }
        }
    )

    model = NavVLA_CPM(cfg)

    assert model.tvi_mode == expected_mode
    assert model.tvi_dim == expected_dim
    assert model.tvi_embedding.mode == expected_mode


def test_navvla_cpm_rejects_invalid_tvi_mode_before_building_models(monkeypatch) -> None:
    builder_calls = []

    def fail_if_called(builder_name):
        def _fail(*_args, **_kwargs):
            builder_calls.append(builder_name)
            raise AssertionError(f"{builder_name} builder must not be called")

        return _fail

    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", fail_if_called("VLM"))
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", fail_if_called("action"))
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "navvla": {"tvi_mode": "invalid"},
            }
        }
    )

    with pytest.raises(ValueError, match="unsupported TVI mode"):
        NavVLA_CPM(cfg)

    assert builder_calls == []


def test_tvi_module_preserves_embedding_parameter_layout() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding, sinusoidal_scalar_pe

    module = NavVLATVIEmbedding(hidden_size=8)
    assert not hasattr(module, "mask_token")
    expected_shapes = {
        "base": (8,),
        "time_mlp.0.weight": (8, 8),
        "time_mlp.0.bias": (8,),
        "time_mlp.2.weight": (8, 8),
        "time_mlp.2.bias": (8,),
        "angle_mlp.0.weight": (8, 8),
        "angle_mlp.0.bias": (8,),
        "angle_mlp.2.weight": (8, 8),
        "angle_mlp.2.bias": (8,),
    }
    state = {key: value.clone() for key, value in module.state_dict().items()}

    assert {key: tuple(value.shape) for key, value in state.items()} == expected_shapes
    fresh_module = NavVLATVIEmbedding(hidden_size=8)
    fresh_module.load_state_dict(state, strict=True)
    for key, value in fresh_module.state_dict().items():
        torch.testing.assert_close(value, state[key])
    assert sinusoidal_scalar_pe(torch.tensor([2.0]), dim=4).shape == (1, 4)


@pytest.mark.parametrize("hidden_size", [2, 6, 10])
def test_tvi_embedding_rejects_hidden_sizes_not_divisible_by_four(hidden_size) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    with pytest.raises(ValueError, match="divisible by 4"):
        NavVLATVIEmbedding(hidden_size=hidden_size)


@pytest.mark.parametrize("hidden_size", [4, 8])
def test_tvi_embedding_accepts_hidden_sizes_divisible_by_four(hidden_size) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=hidden_size)

    assert module(torch.tensor([[1.0, 0.3]])).shape == (1, hidden_size)


def test_tvi_camera_pose_mode_returns_one_hidden_vector_per_row() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, mode="time_camera_pose")
    tvi = torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7, 0.3, -0.6, 0.9],
            [2.0, -0.8, 0.5, 1.1, -0.7, 0.4, -1.0],
        ],
        dtype=torch.float32,
    )

    embeddings = module(tvi)

    assert embeddings.shape == (2, 8)
    assert embeddings.shape[0] == tvi.shape[0]


def test_tvi_learned_token_mode_uses_one_shared_trainable_vector() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, mode="learned_token")
    with torch.no_grad():
        module.base.copy_(torch.arange(8, dtype=torch.float32))
    tvi = torch.tensor(
        [[0.0, 0.0], [123.0, -2.5], [float("nan"), float("inf")]],
        requires_grad=True,
    )

    embeddings = module(tvi)

    assert {key: tuple(value.shape) for key, value in module.state_dict().items()} == {"base": (8,)}
    torch.testing.assert_close(embeddings, module.base.unsqueeze(0).expand(3, -1))
    embeddings.sum().backward()
    torch.testing.assert_close(module.base.grad, torch.full((8,), 3.0))
    assert tvi.grad is None


def test_tvi_learned_token_mode_ignores_history_mask_augmentation() -> None:
    from starVLA.model.modules.navvla_context import mask_history_tvi_embeddings
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    owner = SimpleNamespace(
        tvi_embedding=NavVLATVIEmbedding(
            hidden_size=8,
            mode="learned_token",
            enable_mask_token=True,
        )
    )
    assert set(owner.tvi_embedding.state_dict()) == {"base"}
    embeddings = owner.tvi_embedding(torch.zeros(2, 2))
    blocks = [{"is_history": True}, {"is_history": True}]

    masked = mask_history_tvi_embeddings(owner, embeddings, blocks, probability=1.0)

    torch.testing.assert_close(masked, embeddings)


def test_tvi_camera_pose_mode_routes_gradients_through_all_inputs() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    torch.manual_seed(0)
    module = NavVLATVIEmbedding(hidden_size=8, mode="time_camera_pose")
    tvi = torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7, 0.3, -0.6, 0.9],
            [2.0, -0.8, 0.5, 1.1, -0.7, 0.4, -1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    module(tvi).square().sum().backward()

    assert tvi.grad is not None
    for column in range(7):
        assert torch.count_nonzero(tvi.grad[:, column]).item() == tvi.shape[0]
    for parameter in module.pose_mlp.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() > 0


def test_metric_camera_pose_features_use_fixed_physical_multiscale_encoding() -> None:
    from starVLA.model.modules.tvi import metric_camera_pose_features

    tvi = torch.tensor(
        [[10.0, 16.0, -16.0, 0.0, torch.pi / 2, -torch.pi / 2, torch.pi]],
        dtype=torch.float32,
    )

    time_features, position_features, rotation_features = metric_camera_pose_features(tvi)

    assert time_features.shape == (1, 34)
    assert position_features.shape == (1, 102)
    assert rotation_features.shape == (1, 6)
    assert time_features.dtype == torch.float32
    assert position_features.dtype == torch.float32
    assert rotation_features.dtype == torch.float32
    torch.testing.assert_close(time_features[:, :2], torch.tensor([[10.0 / 4096.0, torch.log1p(torch.tensor(1.0))]]))
    torch.testing.assert_close(
        position_features[:, :6],
        torch.tensor([[16.0 / 2048.0, -16.0 / 2048.0, 0.0, torch.log(torch.tensor(2.0)), -torch.log(torch.tensor(2.0)), 0.0]]),
    )

    time_wavelengths = torch.logspace(torch.log10(torch.tensor(0.2)), torch.log10(torch.tensor(4096.0)), 16)
    expected_time_angles = 2.0 * torch.pi * torch.tensor(10.0) / time_wavelengths
    expected_time_fourier = torch.stack([torch.sin(expected_time_angles), torch.cos(expected_time_angles)], dim=-1)
    torch.testing.assert_close(time_features[:, 2:].reshape(1, 16, 2), expected_time_fourier.unsqueeze(0))

    position_wavelengths = torch.logspace(0.0, torch.log10(torch.tensor(2048.0)), 16)
    xyz = torch.tensor([[16.0, -16.0, 0.0]])
    expected_position_angles = 2.0 * torch.pi * xyz.unsqueeze(-1) / position_wavelengths
    expected_position_fourier = torch.stack(
        [torch.sin(expected_position_angles), torch.cos(expected_position_angles)], dim=-1
    )
    torch.testing.assert_close(
        position_features[:, 6:].reshape(1, 3, 16, 2),
        expected_position_fourier,
    )
    torch.testing.assert_close(
        rotation_features,
        torch.tensor([[1.0, 0.0, -1.0, 0.0, 0.0, -1.0]]),
        atol=1e-4,
        rtol=1e-4,
    )


def test_metric_camera_pose_mode_routes_gradients_through_all_inputs_and_gates() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    torch.manual_seed(0)
    module = NavVLATVIEmbedding(hidden_size=8, mode="metric_camera_pose")
    tvi = torch.tensor(
        [
            [1.0, 0.2, -0.4, 0.7, 0.3, -0.6, 0.9],
            [2.0, -0.8, 0.5, 1.1, -0.7, 0.4, -1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    embeddings = module(tvi)
    embeddings.square().sum().backward()

    assert embeddings.shape == (2, 8)
    assert tvi.grad is not None
    for column in range(7):
        assert torch.count_nonzero(tvi.grad[:, column]).item() == tvi.shape[0]
    for parameter_name in ("metric_time_gate", "metric_position_gate", "metric_rotation_gate"):
        parameter = getattr(module, parameter_name)
        assert parameter.item() == pytest.approx(1.0)
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 1


@pytest.mark.parametrize(
    ("mode", "tvi", "expected_shape"),
    [
        ("time_yaw", torch.zeros(2, 7), r"shape \[N, 2\]"),
        ("learned_token", torch.zeros(2, 7), r"shape \[N, 2\]"),
        ("time_camera_pose", torch.zeros(2, 2), r"shape \[N, 7\]"),
        ("metric_camera_pose", torch.zeros(2, 2), r"shape \[N, 7\]"),
    ],
)
def test_tvi_modes_validate_exact_input_width(mode, tvi, expected_shape) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, mode=mode)

    with pytest.raises(ValueError, match=expected_shape):
        module(tvi)


def test_tvi_embedding_rejects_unknown_mode_and_lists_supported_modes() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    with pytest.raises(ValueError) as exc_info:
        NavVLATVIEmbedding(hidden_size=8, mode="bad")

    message = str(exc_info.value)
    assert "time_yaw" in message
    assert "learned_token" in message
    assert "time_camera_pose" in message
    assert "metric_camera_pose" in message


@pytest.mark.parametrize(
    ("mode", "input_dim"),
    [
        ("time_yaw", 2),
        ("time_camera_pose", 7),
        ("metric_camera_pose", 7),
    ],
)
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_tvi_modes_reject_non_finite_inputs(mode, input_dim, non_finite) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, mode=mode)
    tvi = torch.zeros(2, input_dim)
    tvi[1, -1] = non_finite

    with pytest.raises(RuntimeError, match="TVI tensor must contain only finite values"):
        module(tvi)


def test_tvi_camera_pose_mode_only_adds_pose_mlp_parameters() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    legacy_shapes = {
        key: tuple(value.shape) for key, value in NavVLATVIEmbedding(hidden_size=8).state_dict().items()
    }
    pose_shapes = {
        key: tuple(value.shape)
        for key, value in NavVLATVIEmbedding(hidden_size=8, mode="time_camera_pose").state_dict().items()
    }

    assert pose_shapes == legacy_shapes | {
        "pose_mlp.0.weight": (8, 196),
        "pose_mlp.0.bias": (8,),
        "pose_mlp.2.weight": (8, 8),
        "pose_mlp.2.bias": (8,),
    }


def test_qwen_framework_default_tvi_embedding_has_no_mask_token(monkeypatch) -> None:
    from starVLA.model.framework.VLM4A import navvla_qwenpi_v3 as navvla_qwen_module
    from starVLA.model.framework.VLM4A.navvla_qwenpi_v3 import NavVLA_QwenPI_v3

    class FakeQwenVLInterface:
        def __init__(self) -> None:
            language_model = SimpleNamespace(layers=[object(), object()])
            self.model = SimpleNamespace(
                config=SimpleNamespace(hidden_size=8),
                model=SimpleNamespace(language_model=language_model),
            )

    class FakeActionHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = SimpleNamespace(transformer_blocks=[object(), object()])

    monkeypatch.setattr(navvla_qwen_module, "get_vlm_model", lambda config: FakeQwenVLInterface())
    monkeypatch.setattr(navvla_qwen_module, "get_action_model", lambda config: FakeActionHead())

    framework = NavVLA_QwenPI_v3(OmegaConf.create({"framework": {}}))

    assert "tvi_embedding.mask_token" not in framework.state_dict()


def test_tvi_module_optionally_registers_zero_initialized_mask_token() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, enable_mask_token=True)

    assert set(module.state_dict()) == {
        "base",
        "mask_token",
        "time_mlp.0.weight",
        "time_mlp.0.bias",
        "time_mlp.2.weight",
        "time_mlp.2.bias",
        "angle_mlp.0.weight",
        "angle_mlp.0.bias",
        "angle_mlp.2.weight",
        "angle_mlp.2.bias",
    }
    torch.testing.assert_close(module.mask_token, torch.zeros(8))


def test_tvi_mask_replacement_routes_gradients_by_selected_row() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, enable_mask_token=True)
    embeddings = module(torch.tensor([[1.0, 0.0], [2.0, 0.5]], dtype=torch.float32))
    embeddings.retain_grad()

    masked = module.replace_masked_rows(embeddings, [True, False])
    masked.sum().backward()

    assert torch.count_nonzero(module.mask_token.grad).item() == 8
    assert torch.count_nonzero(embeddings.grad[0]).item() == 0
    assert torch.count_nonzero(embeddings.grad[1]).item() == 8
    assert torch.count_nonzero(module.time_mlp[2].bias.grad).item() == 8


@pytest.mark.parametrize(
    ("embeddings", "row_mask", "error", "message"),
    [
        (torch.zeros(2, 2, 8), [True, False], ValueError, "shape \\[N, H\\]"),
        (torch.zeros(2, 6), [True, False], ValueError, "hidden dimension"),
        (torch.zeros(2, 8), [[True], [False]], ValueError, "row mask"),
        (torch.zeros(2, 8), [True], ValueError, "row mask"),
    ],
)
def test_tvi_mask_replacement_validates_shapes(embeddings, row_mask, error, message) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8, enable_mask_token=True)

    with pytest.raises(error, match=message):
        module.replace_masked_rows(embeddings, row_mask)


def test_tvi_mask_replacement_rejects_disabled_masking() -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    module = NavVLATVIEmbedding(hidden_size=8)

    with pytest.raises(RuntimeError, match="mask token is disabled"):
        module.replace_masked_rows(torch.zeros(2, 8), [True, False])


def test_navvla_cpm_masks_only_ordinary_history_tvi(monkeypatch) -> None:
    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    model.tvi_embedding = NavVLATVIEmbedding(hidden_size=8, enable_mask_token=True)
    embeddings = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    blocks = [
        {"is_history": True, "is_long_memory": True},
        {"is_history": True, "is_long_memory": False},
        {"is_history": True},
        {"is_history": False},
    ]
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0, 0.0, 0.9, 0.0]))

    masked = model._mask_history_tvi_embeddings(embeddings, blocks, probability=0.5)

    torch.testing.assert_close(masked[0], embeddings[0])
    torch.testing.assert_close(masked[1], model.tvi_embedding.mask_token)
    torch.testing.assert_close(masked[2], embeddings[2])
    torch.testing.assert_close(masked[3], embeddings[3])


def test_navvla_cpm_enabled_training_forward_requires_progress(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    fake_action = _FakeActionHead()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: fake_action)
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {
                    "history_visual_tokens": 2,
                    "long_memory_visual_tokens": 0,
                    "current_visual_tokens": 2,
                    "action_placeholder_count": 2,
                    "history_augmentation": {"enabled": True},
                },
                "action_model": {"action_horizon": 2, "action_dim": 4, "state_dim": 0},
            }
        }
    )
    model = NavVLA_CPM(cfg)
    sample = {
        "images": {"front": Image.new("RGB", (4, 4))},
        "current_tvi": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "history_tvi": np.zeros((0, 2), dtype=np.float32),
        "history_mask": np.zeros((0,), dtype=bool),
        "lang": "go",
        "platform_text": "uav",
        "action": np.zeros((2, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((2,), dtype=bool),
        "metadata": {"required_cameras": ["front"]},
    }

    with pytest.raises(ValueError, match="training_step"):
        model([sample])


def test_navvla_cpm_eval_forward_disables_enabled_augmentation(monkeypatch) -> None:
    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    model.history_augmentation = navvla_cpm_module.HistoryAugmentationConfig(enabled=True)
    model.action_horizon = 1
    model.action_dim = 4
    model.config = SimpleNamespace(framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 1}))
    model.action_model = _FakeActionHead()
    recorded = []

    def fake_forward(samples, **kwargs):
        recorded.append(kwargs)
        return torch.zeros((len(samples), 1, 4)), []

    model._forward_vlm_for_action = fake_forward
    model.eval()
    sample = {
        "action": np.zeros((1, 4), dtype=np.float32),
        "action_padding_mask": np.zeros((1,), dtype=bool),
        "metadata": {},
    }

    model([sample])

    assert recorded == [{"history_shuffle_probability": 0.0, "tvi_mask_probability": 0.0}]


def test_navvla_cpm_predict_action_disables_augmentation_while_model_is_training() -> None:
    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    model.action_model = _Float32PredictActionHead()
    model.history_augmentation = navvla_cpm_module.HistoryAugmentationConfig(enabled=True)
    recorded = []

    def fake_forward(samples, **kwargs):
        recorded.append(kwargs)
        return torch.ones((len(samples), 2, 16)), []

    model._forward_vlm_for_action = fake_forward
    model._compute_online_long_memory_updates = lambda samples: []
    model.train()

    model.predict_action([{}])

    assert recorded == [
        {
            "capture_online_current_cache": True,
            "history_shuffle_probability": 0.0,
            "tvi_mask_probability": 0.0,
        }
    ]


def test_navvla_cpm_predict_action_accepts_inference_tvi_mask_probability() -> None:
    model = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(model)
    model.action_model = _Float32PredictActionHead()
    recorded = []

    def fake_forward(samples, **kwargs):
        recorded.append(kwargs)
        return torch.ones((len(samples), 2, 16)), []

    model._forward_vlm_for_action = fake_forward
    model._compute_online_long_memory_updates = lambda samples: []

    model.predict_action([{}], tvi_mask_probability=1.0)

    assert recorded == [
        {
            "capture_online_current_cache": True,
            "history_shuffle_probability": 0.0,
            "tvi_mask_probability": 1.0,
        }
    ]


def test_navvla_cpm_mask_token_uses_tvi_embedding_learning_rate_group(monkeypatch) -> None:
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    fake_vlm = _FakeMiniCPMVLM()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _FakeActionHead())
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6"},
                "navvla": {"long_memory_visual_tokens": 0, "action_placeholder_count": 2},
                "action_model": {"action_horizon": 2, "action_dim": 4, "state_dim": 0},
            },
            "trainer": {
                "learning_rate": {"base": 1.0e-5, "tvi_embedding": 1.0e-4},
                "freeze_modules": "",
            },
        }
    )
    model = NavVLA_CPM(cfg)

    groups = build_param_lr_groups(model, cfg)
    mask_parameter_id = id(model.tvi_embedding.mask_token)

    matching = [group for group in groups if any(id(parameter) == mask_parameter_id for parameter in group["params"])]
    assert len(matching) == 1
    assert matching[0]["name"] == "tvi_embedding"
    assert matching[0]["lr"] == pytest.approx(1.0e-4)


def _tvi_checkpoint_model(*, framework_name: str, tvi_mode: str) -> torch.nn.Module:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding

    class _TVICheckpointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(framework=SimpleNamespace(name=framework_name))
            self.tvi_embedding = NavVLATVIEmbedding(8, mode=tvi_mode, enable_mask_token=True)
            self.other = torch.nn.Linear(8, 4)

    return _TVICheckpointModel()


def test_checkpoint_compatibility_loads_legacy_state_into_pose_minicpm() -> None:
    from starVLA.model.framework.base_framework import load_framework_state_dict_compatibly

    model = _tvi_checkpoint_model(framework_name="navvla_cpm", tvi_mode="time_camera_pose")
    legacy_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key != "tvi_embedding.mask_token" and not key.startswith("tvi_embedding.pose_mlp.")
    }

    load_framework_state_dict_compatibly(model, legacy_state)


@pytest.mark.parametrize(
    "config",
    [
        {"framework": {"name": "navvla_cpm"}},
        OmegaConf.create({"framework": {"name": "navvla_cpm"}}),
    ],
    ids=["dict", "omegaconf"],
)
def test_checkpoint_compatibility_recognizes_mapping_framework_configs(config) -> None:
    from starVLA.model.framework.base_framework import load_framework_state_dict_compatibly

    model = _tvi_checkpoint_model(framework_name="unused", tvi_mode="time_camera_pose")
    model.config = config
    legacy_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key != "tvi_embedding.mask_token" and not key.startswith("tvi_embedding.pose_mlp.")
    }

    load_framework_state_dict_compatibly(model, legacy_state)


def test_checkpoint_compatibility_keeps_unrelated_minicpm_keys_strict() -> None:
    from starVLA.model.framework.base_framework import load_framework_state_dict_compatibly

    model = _tvi_checkpoint_model(framework_name="navvla_cpm", tvi_mode="time_camera_pose")
    legacy_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key != "tvi_embedding.mask_token" and not key.startswith("tvi_embedding.pose_mlp.")
    }

    missing_other = {key: value for key, value in legacy_state.items() if key != "other.bias"}
    with pytest.raises(RuntimeError):
        load_framework_state_dict_compatibly(model, missing_other)
    unexpected = dict(legacy_state, unexpected=torch.zeros(1))
    with pytest.raises(RuntimeError):
        load_framework_state_dict_compatibly(model, unexpected)


def test_checkpoint_compatibility_does_not_allow_pose_keys_for_non_minicpm() -> None:
    from starVLA.model.framework.base_framework import load_framework_state_dict_compatibly

    model = _tvi_checkpoint_model(framework_name="navvla_qwenpi_v3", tvi_mode="time_camera_pose")
    legacy_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("tvi_embedding.pose_mlp.")
    }

    with pytest.raises(RuntimeError):
        load_framework_state_dict_compatibly(model, legacy_state)


def test_checkpoint_compatibility_rejects_pose_key_for_legacy_minicpm() -> None:
    from starVLA.model.framework.base_framework import load_framework_state_dict_compatibly

    model = _tvi_checkpoint_model(framework_name="navvla_cpm", tvi_mode="time_yaw")
    checkpoint_state = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_state["tvi_embedding.pose_mlp.made_up"] = torch.zeros(1)

    with pytest.raises(RuntimeError):
        load_framework_state_dict_compatibly(model, checkpoint_state)


def test_trainer_full_model_loader_uses_narrow_minicpm_compatibility(tmp_path, monkeypatch) -> None:
    from starVLA.model.modules.tvi import NavVLATVIEmbedding
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

    class _MiniCPMLike(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(framework=SimpleNamespace(name="navvla_cpm"))
            self.tvi_embedding = NavVLATVIEmbedding(8, enable_mask_token=True)
            self.other = torch.nn.Linear(8, 4)

    model = _MiniCPMLike()
    old_state = {key: value.clone() for key, value in model.state_dict().items() if key != "tvi_embedding.mask_token"}
    checkpoint = tmp_path / "old_minicpm.pt"
    torch.save(old_state, checkpoint)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    assert TrainerUtils.load_pretrained_backbones(model, str(checkpoint)) is model

    invalid = {key: value for key, value in old_state.items() if key != "other.bias"}
    torch.save(invalid, checkpoint)
    with pytest.raises(RuntimeError, match="loading full model failed"):
        TrainerUtils.load_pretrained_backbones(model, str(checkpoint))


def test_bats_selector_accepts_cache_only_history_records() -> None:
    from starVLA.model.modules.bats import select_bats_history

    candidates = []
    for frame_index, value in enumerate([10, 20, 30]):
        candidates.append(
            (
                frame_index,
                {
                    "navvla_online_visual_tokens": {"front": np.full((4, 8), value, dtype=np.float16)},
                    "navvla_eval": {"frame_index": frame_index},
                },
            )
        )
    result = select_bats_history(
        candidates=candidates,
        anchor_frame_index=3,
        episode_id="episode-a",
        dataset_name="unit",
        seed=7,
        epsilon=0.1,
        k=0.0,
        use_dynamic_bats_k=False,
        token_budget=84,
        budget_num_cameras=1,
        current_visual_tokens=64,
        history_visual_tokens=4,
        tvi_tokens=1,
    )

    assert result.max_history_frames == 2
    assert [item["navvla_eval"]["frame_index"] for item in result.selected] == [0, 2]
    assert len(result.ranked_selected) == 2


def test_bats_selector_overflow_uses_independent_frame_priorities() -> None:
    from starVLA.model.modules.bats import select_bats_history

    candidates = [(frame_index, {"frame_index": frame_index}) for frame_index in range(20)]

    result = select_bats_history(
        candidates=candidates,
        anchor_frame_index=20,
        episode_id="episode-a",
        dataset_name="unit",
        seed=42,
        epsilon=0.1,
        k=4.0,
        use_dynamic_bats_k=False,
        token_budget=108,
        budget_num_cameras=1,
        current_visual_tokens=64,
        history_visual_tokens=4,
        tvi_tokens=1,
    )

    assert [item["frame_index"] for item in result.selected] == [0, 14, 17]


def test_bats_priority_capped_mode_matches_tool_context_selection() -> None:
    from starVLA.model.modules.bats import bats_keep_probability, select_bats_history

    candidates = [(frame_index, {"frame_index": frame_index}) for frame_index in range(20)]
    result = select_bats_history(
        candidates=candidates,
        anchor_frame_index=20,
        episode_id="episode-a",
        dataset_name="unit",
        seed=42,
        epsilon=0.1,
        k=4.0,
        use_dynamic_bats_k=False,
        token_budget=108,
        budget_num_cameras=1,
        current_visual_tokens=64,
        history_visual_tokens=4,
        tvi_tokens=1,
        sampling_mode="priority_capped",
    )

    expected = []
    for frame_index in range(20):
        probability = bats_keep_probability(
            history_step=frame_index,
            current_step=20,
            epsilon=0.1,
            k=4.0,
        )
        draw = random.Random(f"42:unit:episode-a:20:{frame_index}").random()
        if draw < probability:
            expected.append((draw / probability, frame_index))

    expected = sorted(expected, key=lambda item: (item[0], item[1]))[: result.max_history_frames]
    expected_frame_indices = sorted(frame_index for _priority, frame_index in expected)
    assert [item["frame_index"] for item in result.selected] == expected_frame_indices
    assert any(frame_index < 15 for frame_index in expected_frame_indices)


def test_cpm_samples_from_batch_preserves_online_history_images(monkeypatch) -> None:
    fake_vlm = _FakeMiniCPMVLM()
    monkeypatch.setattr(navvla_cpm_module, "get_vlm_model", lambda config: fake_vlm)
    monkeypatch.setattr(navvla_cpm_module, "get_action_model", lambda config: _Float32PredictActionHead())
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "navvla_cpm",
                "qwenvl": {"base_vlm": "openbmb/MiniCPM-V-4.6", "downsample_mode": "16x"},
                "navvla": {
                    "required_cameras": ["front"],
                    "history_visual_tokens": 4,
                    "long_memory_visual_tokens": 0,
                    "current_visual_tokens": 64,
                },
                "action_model": {"action_horizon": 2, "action_dim": 4, "state_dim": 0},
            }
        }
    )
    model = NavVLA_CPM(cfg)
    history_image = Image.new("RGB", (448, 448), color=(10, 0, 0))

    samples = model._samples_from_batch(
        {
            "images": {"front": [Image.new("RGB", (448, 448), color=(20, 0, 0))]},
            "history_images": {"front": [[history_image]]},
            "current_tvi": [np.asarray([[1.0, 0.0]], dtype=np.float32)],
            "history_tvi": np.asarray([[[0.0, 0.0]]], dtype=np.float32),
            "history_mask": np.asarray([[True]], dtype=bool),
            "lang": ["go"],
            "platform_text": ["uav"],
            "action": np.zeros((1, 2, 4), dtype=np.float32),
            "action_padding_mask": np.zeros((1, 2), dtype=bool),
            "metadata": [{"history_blocks": [{"step_index": 0, "camera_name": "front"}]}],
        }
    )

    assert samples[0]["history_images"]["front"] == [history_image]


def test_bats_long_memory_candidate_matches_offline_promotion_rule() -> None:
    from starVLA.model.modules.bats import select_long_memory_candidate

    ranked = [
        {"frame_index": 10},
        {"frame_index": 11},
        {"frame_index": 13},
        {"frame_index": 14},
        {"frame_index": 12},
    ]

    assert select_long_memory_candidate(ranked, memory_frame_indices={10, 12}) == {"frame_index": 14}


def test_cpm_model_no_longer_imports_qwenpi_v3_implementation() -> None:
    source = Path("starVLA/model/framework/VLM4A/navvla_cpm.py").read_text(encoding="utf-8")

    assert "navvla_qwenpi_v3 import NavVLATVIEmbedding" not in source


def test_incremental_long_memory_matches_full_offline_recurrence() -> None:
    from starVLA.model.modules.long_memory import LongMemoryTokenAggregator

    aggregator = LongMemoryTokenAggregator(
        source_visual_tokens=2,
        long_memory_visual_tokens=3,
        decay=0.9,
        update_weight=0.1,
    )
    source_tokens = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    source_tvi = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    source_blocks = [{"camera_name": "front", "step_index": index} for index in range(3)]
    expected_tokens, expected_tvi, expected_blocks = aggregator.aggregate_sample(
        source_tokens=source_tokens,
        source_tvi=source_tvi,
        source_mask=torch.ones(3, dtype=torch.bool),
        source_blocks=source_blocks,
        required_cameras=["front"],
    )
    tokens = None
    tvi = None
    blocks: list[dict] = []
    for index in range(3):
        tokens, tvi, blocks = aggregator.update_state(
            previous_tokens=tokens,
            previous_tvi=tvi,
            previous_blocks=blocks,
            source_tokens=source_tokens[index : index + 1],
            source_tvi=source_tvi[index : index + 1],
            source_mask=torch.ones(1, dtype=torch.bool),
            source_blocks=source_blocks[index : index + 1],
            required_cameras=["front"],
        )

    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(tvi, expected_tvi)
    assert blocks == expected_blocks


def test_runtime_adapter_compacts_cached_history_and_reuses_tokens() -> None:
    adapter = OnlineNavVLARuntimeDatasetAdapter(
        required_cameras=("front",),
        history_selection="recent",
        history_image_frames=1,
    )
    history = EpisodeHistory()
    action = np.zeros((1, 4), dtype=np.float32)
    prediction = ActionPrediction(normalized_actions=action, raw_actions=action)
    observation = {
        "image": np.ones((4, 4, 3), dtype=np.uint8),
        "navvla_eval": {"frame_index": 0, "timestamp": 0.0},
        "navvla_online_visual_tokens": {"front": np.ones((4, 16), dtype=np.float16)},
    }

    adapter.update_history(history=history, observation=observation, prediction=prediction, instruction="go")

    stored = history.observations[0]
    assert "image" not in stored
    assert "images" not in stored
    assert history.images == []
    assert "navvla_online_dhash" not in stored
    sample = adapter.build_example(
        observation={
            "image": np.full((4, 4, 3), 2, dtype=np.uint8),
            "navvla_eval": {"frame_index": 1, "timestamp": 1.0},
        },
        history=history,
        instruction="go",
    )
    assert sample["history_cached_embeds"].shape == (1, 4, 16)
    assert sample["history_images"]["front"] == []


def test_runtime_history_rejects_raw_cache_mixture() -> None:
    history = EpisodeHistory(
        observations=[
            {"image": np.zeros((4, 4, 3), dtype=np.uint8), "navvla_eval": {"frame_index": 0}},
            {
                "navvla_online_visual_tokens": {"front": np.ones((4, 16), dtype=np.float16)},
                "navvla_eval": {"frame_index": 1},
            },
        ]
    )

    with pytest.raises(ValueError, match="must be entirely cached or entirely raw images"):
        build_online_navvla_history_sample(
            observation={"image": np.ones((4, 4, 3), dtype=np.uint8), "navvla_eval": {"frame_index": 2}},
            history=history,
            instruction="go",
            state=None,
            config=OnlineHistoryConfig(history_policy="recent", history_image_frames=2),
        )


def test_runtime_uniform_history_matches_training_continuous_uniform_selection() -> None:
    history = EpisodeHistory(
        observations=[
            {
                "navvla_online_visual_tokens": {"front": np.full((4, 2), index, dtype=np.float16)},
                "navvla_eval": {"frame_index": index, "timestamp": float(index)},
            }
            for index in range(10)
        ]
    )

    sample = build_online_navvla_history_sample(
        observation={
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "navvla_eval": {"frame_index": 10, "timestamp": 10.0},
        },
        history=history,
        instruction="go",
        state=None,
        config=OnlineHistoryConfig(history_policy="uniform", history_image_frames=4),
    )

    assert sample["history_tvi"][:, 0].tolist() == [0.0, 3.0, 6.0, 9.0]
    assert sample["history_cached_embeds"][:, 0, 0].tolist() == [0.0, 3.0, 6.0, 9.0]


@pytest.mark.parametrize("history_policy", ["recent", "uniform", "bats"])
def test_runtime_history_filters_source_stride_before_selection(history_policy: str) -> None:
    history = EpisodeHistory(
        observations=[
            {
                "navvla_online_visual_tokens": {"front": np.full((4, 2), index, dtype=np.float16)},
                "navvla_eval": {"frame_index": index, "timestamp": float(index)},
            }
            for index in range(16)
        ]
    )

    sample = build_online_navvla_history_sample(
        observation={
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "navvla_eval": {"frame_index": 16, "timestamp": 16.0},
        },
        history=history,
        instruction="go",
        state=None,
        config=OnlineHistoryConfig(
            history_policy=history_policy,
            history_candidate_source_stride=5,
        ),
    )

    assert sample["metadata"]["history_steps"] == [
        {"index": index, "frame_index": index, "timestamp": float(index)}
        for index in (0, 5, 10, 15)
    ]
    assert sample["metadata"]["history_candidate_source_stride"] == 5
    assert sample["history_cached_embeds"][:, 0, 0].tolist() == [0.0, 5.0, 10.0, 15.0]


def test_runtime_history_source_stride_prefers_explicit_source_frame_index() -> None:
    history = EpisodeHistory(
        observations=[
            {
                "navvla_online_visual_tokens": {"front": np.full((4, 2), frame_index, dtype=np.float16)},
                "navvla_eval": {
                    "frame_index": frame_index,
                    "source_frame_index": source_frame_index,
                    "timestamp": float(source_frame_index),
                },
            }
            for frame_index, source_frame_index in enumerate((0, 1, 5, 6, 10, 11))
        ]
    )

    sample = build_online_navvla_history_sample(
        observation={
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "navvla_eval": {"frame_index": 8, "source_frame_index": 16, "timestamp": 16.0},
        },
        history=history,
        instruction="go",
        state=None,
        config=OnlineHistoryConfig(
            history_policy="recent",
            history_candidate_source_stride=5,
        ),
    )

    assert sample["metadata"]["history_steps"] == [
        {"index": 0, "frame_index": 0, "timestamp": 0.0},
        {"index": 2, "frame_index": 2, "timestamp": 5.0},
        {"index": 4, "frame_index": 4, "timestamp": 10.0},
    ]


@pytest.mark.parametrize("stride", [0, -1])
def test_runtime_history_rejects_non_positive_source_stride(stride: int) -> None:
    with pytest.raises(ValueError, match="history_candidate_source_stride must be a positive integer"):
        OnlineHistoryConfig(history_candidate_source_stride=stride)


def test_runtime_adapter_propagates_history_candidate_source_stride() -> None:
    from NavVLAeval.common.data.runtime_dataset import get_runtime_dataset_adapter

    adapter = get_runtime_dataset_adapter(
        {
            "runtime_adapter": "openfly",
            "history_candidate_source_stride": 5,
        }
    )

    assert adapter.history_config.history_candidate_source_stride == 5


@pytest.mark.parametrize(
    ("history_policy", "expected_indices"),
    [
        ("recent", [8, 9]),
        ("uniform", [0, 9]),
    ],
)
def test_online_history_null_frame_cap_uses_token_budget_for_all_sampling_modes(
    history_policy: str,
    expected_indices: list[int],
) -> None:
    history = EpisodeHistory(
        observations=[
            {
                "navvla_online_visual_tokens": {"front": np.full((4, 2), index, dtype=np.float16)},
                "navvla_eval": {"frame_index": index, "timestamp": float(index)},
            }
            for index in range(10)
        ]
    )

    sample = build_online_navvla_history_sample(
        observation={
            "image": np.ones((4, 4, 3), dtype=np.uint8),
            "navvla_eval": {"frame_index": 10, "timestamp": 10.0},
        },
        history=history,
        instruction="go",
        state=None,
        config=OnlineHistoryConfig(
            history_policy=history_policy,
            history_image_frames=None,
            required_cameras=("front",),
            bats_token_budget=22,
            current_visual_tokens=2,
            history_visual_tokens=4,
            tvi_tokens=1,
        ),
    )

    assert sample["metadata"]["history_steps"] == [
        {"index": index, "frame_index": index, "timestamp": float(index)}
        for index in expected_indices
    ]
    assert sample["history_cached_embeds"].shape == (2, 4, 2)


def test_online_bats_null_frame_cap_uses_token_budget() -> None:
    candidates = [(index, {"frame_index": index}) for index in range(10)]

    selection = select_bats_history(
        candidates=candidates,
        anchor_frame_index=10,
        episode_id="episode",
        dataset_name="dataset",
        seed=42,
        epsilon=0.9,
        k=4.0,
        use_dynamic_bats_k=False,
        token_budget=22,
        budget_num_cameras=1,
        current_visual_tokens=2,
        history_visual_tokens=4,
        tvi_tokens=1,
        max_history_frames=None,
        sampling_mode="priority_capped",
    )

    assert selection.max_history_frames == 2
    assert len(selection.selected) <= 2


def test_shared_vlnce_runtime_and_configs_replace_independent_adapter() -> None:
    import yaml

    from NavVLAeval.common.data.runtime_dataset import get_runtime_dataset_adapter

    adapter = get_runtime_dataset_adapter(
        {
            "runtime_adapter": "vlnce",
            "dataset_name": "vlnce_r2r",
            "required_cameras": ["front", "left", "right", "rear"],
            "history_selection": "bats",
            "bats_token_budget": 1024,
        }
    )
    assert type(adapter) is OnlineNavVLARuntimeDatasetAdapter
    for path in (Path("NavVLAeval/vlnce/r2r/config_portable.yaml"), Path("NavVLAeval/vlnce/rxr/config_portable.yaml")):
        dataset = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]
        assert dataset["runtime_adapter"] == "vlnce"
        assert "runtime_dataset_class_path" not in dataset
        assert "runtime_dataset_kwargs" not in dataset
    assert not Path("NavVLAeval/vlnce/runtime_dataset.py").exists()


def test_eval_wrapper_does_not_forward_legacy_qwenpi_predict_kwargs() -> None:
    from NavVLAeval.common.model.model_wrappers import StarVLAEvalModel

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs = None

        def predict_action(self, *, examples, **kwargs):
            self.kwargs = kwargs
            return {"normalized_actions": np.zeros((1, 1, 4), dtype=np.float32)}

    inner = FakeModel()
    wrapper = StarVLAEvalModel.from_loaded_model(inner)

    assert wrapper.predict_action({"image": "frame"})["normalized_actions"].shape == (1, 4)
    assert inner.kwargs == {}


def test_eval_configs_drop_legacy_qwenpi_predict_options() -> None:
    import yaml

    legacy_keys = {"visual_token_mode", "use_bf16", "do_sample", "use_ddim", "num_ddim_steps"}
    for path in (
        Path("NavVLAeval/aerialvln/config_portable.yaml"),
        Path("NavVLAeval/openfly/config_portable.yaml"),
        Path("NavVLAeval/traveluav/config_portable.yaml"),
        Path("NavVLAeval/uavflow/config_template.yaml"),
        Path("NavVLAeval/vlnce/r2r/config_portable.yaml"),
        Path("NavVLAeval/vlnce/rxr/config_portable.yaml"),
    ):
        model_config = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        assert legacy_keys.isdisjoint(model_config), path
