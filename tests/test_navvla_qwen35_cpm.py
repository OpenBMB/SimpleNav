from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image

from NavVLAeval.common.data.runtime_history import _online_visual_record
from starVLA.dataloader.cpm_lerobot.cache import Qwen35PooledHistoryTokenStore
from starVLA.dataloader.cpm_lerobot.collate import NavVLACPMCollator, _pad_token_batches
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.VLM4A import navvla_qwen35_cpm as qwen35_cpm_module
from starVLA.model.framework.VLM4A.navvla_cpm import NavVLA_CPM
from starVLA.model.framework.VLM4A.navvla_qwen35_cpm import (
    QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE,
    NavVLA_Qwen35_CPM,
    NavVLAQwen35CPMDefaultConfig,
    _apply_qwen35_tvi_embeddings,
    _insert_qwen35_cached_visual_spans,
    _rewrite_qwen35_visual_spans,
)
from starVLA.model.modules.qwen35_vision import (
    BFLOAT16_BITS_STORAGE_ENCODING,
    bf16_to_numpy_bits,
    decode_qwen35_cache_tokens,
    encode_qwen35_postmerge_batched,
    encode_qwen35_postmerge_one_by_one,
    pool_qwen35_postmerge,
    qwen35_postmerge_token_count,
    qwen35_premerge_token_count,
)
from starVLA.model.modules.vlm import QWen3_5 as qwen35_module
from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY
from tool.navvla.cli import generate_visual_cache as generate_visual_cache_module
from tool.navvla.visual_token_cache import (
    QWEN35_POOLED_HISTORY_CACHE_STAGE,
    default_qwen35_pooled_history_visual_token_profile,
    write_profile_mmap_npy_cache,
)


def test_qwen35_256_grid_has_256_premerge_and_64_postmerge_tokens() -> None:
    grid = [1, 16, 16]
    assert qwen35_premerge_token_count(grid) == 256
    assert qwen35_postmerge_token_count(grid, spatial_merge_size=2) == 64
    config = NavVLAQwen35CPMDefaultConfig()
    assert config.name == "navvla_qwen35_cpm"
    assert config.navvla["visual_cache_stage"] == QWEN35_POOLED_HISTORY_CACHE_STAGE
    assert config.navvla["visual_cache_input_resize"] == [256, 256]
    assert "visual_source_postmerge_tokens" not in config.navvla
    assert "visual_cache_pooled_history_tokens" not in config.navvla
    assert config.qwenvl["attn_implementation"] == "flash_attention_2"
    assert config.qwenvl["max_text_tokens"] == 2048
    assert issubclass(NavVLA_Qwen35_CPM, baseframework)
    assert FRAMEWORK_REGISTRY["navvla_qwen35_cpm"].__name__ == "NavVLA_Qwen35_CPM"


def test_qwen35_postmerge_profile_and_training_config_freeze_merger() -> None:
    profile = default_qwen35_pooled_history_visual_token_profile(encoder_ckpt="checkpoint")
    assert profile.input_resize == (256, 256)
    assert profile.token_count == 4
    assert profile.cache_stage == QWEN35_POOLED_HISTORY_CACHE_STAGE
    assert profile.spatial_merge_size == 2
    assert profile.shard_size == 256
    assert profile.dtype == "uint16"
    assert profile.storage_encoding == BFLOAT16_BITS_STORAGE_ENCODING

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml").read_text(encoding="utf-8")
    )
    assert config["framework"]["navvla"]["visual_cache_input_resize"] == [256, 256]
    assert config["framework"]["navvla"]["visual_cache_stage"] == QWEN35_POOLED_HISTORY_CACHE_STAGE
    assert config["framework"]["qwenvl"]["base_vlm"] == "local/models/Qwen3.5-4B"
    assert (
        config["framework"]["navvla"]["visual_cache_encoder_ckpt"]
        == config["framework"]["qwenvl"]["base_vlm"]
    )
    assert config["datasets"]["vla_data"]["image_resize"] == [256, 256]
    assert config["framework"]["qwenvl"]["attn_implementation"] == "flash_attention_2"
    assert config["trainer"]["freeze_modules"] == "qwen35_vl_interface.model.model.visual"
    assert config["trainer"]["enable_gradient_checkpointing"] is False

    cache_tool = (root / "tool/navvla/generate_qwen35_postmerge_caches.sh").read_text(encoding="utf-8")
    assert "--skip-existing" in cache_tool
    assert "--camera-names" in cache_tool
    assert "rm -" not in cache_tool
    assert "qwen3_5_4b_postmerge_pool4_256_mmap" in cache_tool
    assert "--dtype uint16" in cache_tool


@pytest.mark.parametrize("enabled", [True, False])
def test_qwen35_reads_trainer_gradient_checkpointing_flag(monkeypatch, enabled: bool) -> None:
    class FakeTokenizer:
        padding_side = "right"

        @staticmethod
        def encode(token, *, add_special_tokens):
            assert not add_special_tokens
            return {
                "<|fim_pad|>": [1],
                "<|fim_prefix|>": [2],
                "<|fim_suffix|>": [3],
            }[token]

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @classmethod
        def from_pretrained(cls, model_id):
            assert model_id == "Qwen/Qwen3.5-4B"
            return cls()

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                hidden_size=None,
                image_token_id=248056,
                text_config=SimpleNamespace(hidden_size=8, use_cache=True),
            )
            self.gradient_checkpointing_calls = []

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_calls.append(gradient_checkpointing_kwargs)

    fake_model = FakeModel()

    class FakeQwen35:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "Qwen/Qwen3.5-4B"
            assert kwargs["attn_implementation"] == "sdpa"
            assert kwargs["torch_dtype"] == torch.bfloat16
            return fake_model

    monkeypatch.setattr(qwen35_module, "Qwen3_5ForConditionalGeneration", FakeQwen35)
    monkeypatch.setattr(qwen35_module, "AutoProcessor", FakeProcessor)

    config = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": "Qwen/Qwen3.5-4B",
                    "attn_implementation": "sdpa",
                }
            },
            "trainer": {"enable_gradient_checkpointing": enabled},
        }
    )
    interface = _QWen3_5_VL_Interface(config)

    assert interface.processor.tokenizer.padding_side == "left"
    if enabled:
        assert fake_model.gradient_checkpointing_calls == [{"use_reentrant": False}]
        assert fake_model.config.text_config.use_cache is False
    else:
        assert fake_model.gradient_checkpointing_calls == []
        assert fake_model.config.text_config.use_cache is True


def test_trainers_and_frameworks_share_forward_vlm_contract() -> None:
    loss = torch.tensor(2.0)
    qwen = NavVLA_Qwen35_CPM.__new__(NavVLA_Qwen35_CPM)
    torch.nn.Module.__init__(qwen)

    class FakeQwenInterface(torch.nn.Module):
        def forward(self, **_batch):
            return SimpleNamespace(loss=loss)

    qwen.qwen35_vl_interface = FakeQwenInterface()
    assert qwen.forward_vlm({}) == {"vlm_loss": loss}

    minicpm = NavVLA_CPM.__new__(NavVLA_CPM)
    torch.nn.Module.__init__(minicpm)
    minicpm.forward_pooled_vlm = lambda _batch: SimpleNamespace(loss=loss)
    assert minicpm.forward_vlm({}) == {"vlm_loss": loss}

    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "starVLA/training/train_starvla_cotrain.py",
        "starVLA/training/train_starvlm.py",
    ):
        trainer_source = (root / relative_path).read_text(encoding="utf-8")
        assert "unwrapped.forward_vlm(batch_vlm)" in trainer_source


def test_qwen35_navvla_pool_consumes_postmerge_tokens_without_running_merger() -> None:
    postmerge = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    output = pool_qwen35_postmerge(
        postmerge,
        torch.tensor([1, 4, 4]),
        target_tokens=1,
        spatial_merge_size=2,
    )
    expected = postmerge.mean(dim=0, keepdim=True)
    assert torch.equal(output, expected)


def test_offline_and_online_qwen35_cache_paths_use_same_pool_helper() -> None:
    assert generate_visual_cache_module.pool_qwen35_postmerge is pool_qwen35_postmerge
    assert generate_visual_cache_module.encode_qwen35_postmerge_one_by_one is encode_qwen35_postmerge_one_by_one


def test_qwen35_bfloat16_bit_cache_round_trip_is_exact() -> None:
    online = torch.tensor([[1.001, -2.003]], dtype=torch.bfloat16)
    stored = bf16_to_numpy_bits(online)
    restored = decode_qwen35_cache_tokens(
        stored,
        storage_encoding=BFLOAT16_BITS_STORAGE_ENCODING,
        device=torch.device("cpu"),
        model_dtype=torch.bfloat16,
    )
    assert stored.dtype == np.uint16
    assert torch.equal(online.view(torch.uint16), restored.view(torch.uint16))


def test_qwen35_cached_pool4_tokens_are_consumed_without_second_pool() -> None:
    model = NavVLA_Qwen35_CPM.__new__(NavVLA_Qwen35_CPM)
    torch.nn.Module.__init__(model)
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.model = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))

    model.qwen35_vl_interface = SimpleNamespace(model=FakeModel())
    model.config = SimpleNamespace(
        framework=SimpleNamespace(navvla={"history_visual_tokens": 4, "current_visual_tokens": 64})
    )
    model.hidden_size = 8
    cached_bf16 = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
    cached = bf16_to_numpy_bits(cached_bf16)

    output = model._cached_pooled_history_tokens(
        cached,
        [1, 16, 16],
        storage_encoding=BFLOAT16_BITS_STORAGE_ENCODING,
    )

    assert torch.equal(output, cached_bf16)


def test_qwen35_current_merger_tokens_enter_llm_without_pooling() -> None:
    model = NavVLA_Qwen35_CPM.__new__(NavVLA_Qwen35_CPM)
    torch.nn.Module.__init__(model)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.model = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))

    model.qwen35_vl_interface = SimpleNamespace(model=FakeModel())
    model.hidden_size = 8
    model._visual_token_budgets = lambda: (4, 32, 64)
    postmerge = torch.arange(64 * 8, dtype=torch.float32).reshape(64, 8).to(torch.bfloat16)
    model._encode_online_postmerge = lambda _inputs: [postmerge]
    model._pool_postmerge = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("current frame must not be pooled")
    )

    output, records = model._fuse_qwen35_visual_tokens(
        {
            "_nav_online_indices": [0],
            "_nav_online_image_grid_thw": torch.tensor([[1, 16, 16]]),
        },
        [{"is_history": False, "camera_name": "front", "frame_index": 0}],
        capture_online_current_cache=False,
    )

    assert torch.equal(output, postmerge)
    assert records == []


def test_qwen35_legacy_float16_cache_still_decodes_numerically() -> None:
    cached = np.asarray([[1.25, -2.5]], dtype=np.float16)
    output = decode_qwen35_cache_tokens(
        cached,
        storage_encoding="",
        device=torch.device("cpu"),
        model_dtype=torch.bfloat16,
    )
    assert torch.equal(output, torch.tensor([[1.25, -2.5]], dtype=torch.bfloat16))


def test_qwen35_bfloat16_bits_decoder_rejects_numeric_float_cache() -> None:
    with pytest.raises(TypeError, match="numpy uint16"):
        decode_qwen35_cache_tokens(
            np.ones((4, 8), dtype=np.float16),
            storage_encoding=BFLOAT16_BITS_STORAGE_ENCODING,
            device=torch.device("cpu"),
            model_dtype=torch.bfloat16,
        )


def test_qwen35_cache_visual_forward_remains_isolated_per_image() -> None:
    class FakeVisual:
        def __init__(self) -> None:
            self.calls = []

        def __call__(self, pixels, *, grid_thw, return_dict):
            self.calls.append((pixels.clone(), grid_thw.clone(), return_dict))
            return SimpleNamespace(pooler_output=pixels.sum(dim=-1, keepdim=True))

    visual = FakeVisual()
    grids = torch.tensor([[1, 2, 2], [1, 1, 2]])
    pixels = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    outputs = encode_qwen35_postmerge_one_by_one(visual, pixels, grids)

    assert [tuple(call[0].shape) for call in visual.calls] == [(4, 2), (2, 2)]
    assert all(tuple(call[1].shape) == (1, 3) for call in visual.calls)
    assert torch.equal(outputs[0], pixels[:4].sum(dim=-1, keepdim=True))
    assert torch.equal(outputs[1], pixels[4:].sum(dim=-1, keepdim=True))


def test_qwen35_current_visual_forward_batches_images_and_splits_outputs() -> None:
    class FakeVisual:
        dtype = torch.float32
        spatial_merge_size = 2

        def __init__(self) -> None:
            self.calls = []

        def __call__(self, pixels, *, grid_thw, return_dict):
            self.calls.append((pixels.clone(), grid_thw.clone(), return_dict))
            token_count = sum(
                qwen35_postmerge_token_count(grid, spatial_merge_size=self.spatial_merge_size)
                for grid in grid_thw
            )
            return SimpleNamespace(pooler_output=pixels[:token_count].clone())

    visual = FakeVisual()
    grids = torch.tensor([[1, 4, 4], [1, 2, 4]])
    pixels = torch.arange(48, dtype=torch.float32).reshape(24, 2)

    outputs = encode_qwen35_postmerge_batched(visual, pixels, grids)

    assert len(visual.calls) == 1
    assert tuple(visual.calls[0][0].shape) == (24, 2)
    assert torch.equal(visual.calls[0][1], grids)
    assert [tuple(output.shape) for output in outputs] == [(4, 2), (2, 2)]
    assert torch.equal(outputs[0], pixels[:4])
    assert torch.equal(outputs[1], pixels[4:6])


def test_qwen35_online_encoder_uses_one_packed_visual_forward() -> None:
    class FakeVisual:
        dtype = torch.float32
        spatial_merge_size = 2

        def __init__(self) -> None:
            self.call_count = 0

        def __call__(self, pixels, *, grid_thw, return_dict):
            self.call_count += 1
            return SimpleNamespace(pooler_output=pixels[:6].clone())

    model = NavVLA_Qwen35_CPM.__new__(NavVLA_Qwen35_CPM)
    torch.nn.Module.__init__(model)
    visual = FakeVisual()
    model.qwen35_vl_interface = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(visual=visual))
    )

    outputs = model._encode_online_postmerge(
        {
            "_nav_online_image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
            "pixel_values": torch.arange(48, dtype=torch.float32).reshape(24, 2),
        }
    )

    assert visual.call_count == 1
    assert [tuple(output.shape) for output in outputs] == [(4, 2), (2, 2)]


def test_qwen35_rewrite_keeps_left_padding_mrope_types_and_only_online_pixels() -> None:
    image_token_id = 99
    vision_start = 98
    vision_end = 97
    first = [vision_start, *([image_token_id] * 64), vision_end]
    second = [vision_start, *([image_token_id] * 64), vision_end]
    active = torch.tensor([11, *first, 12, *second, 13], dtype=torch.long)
    input_ids = torch.cat((torch.zeros(3, dtype=torch.long), active)).unsqueeze(0)
    attention_mask = torch.cat((torch.zeros(3, dtype=torch.long), torch.ones_like(active))).unsqueeze(0)
    token_types = torch.zeros_like(input_ids)
    token_types[input_ids == image_token_id] = 1
    pixel_values = torch.arange(512 * 2, dtype=torch.float32).reshape(512, 2)
    blocks = [
        {"is_history": True, "is_cached_history": True},
        {"is_history": False, "is_cached_history": False},
    ]
    output = _rewrite_qwen35_visual_spans(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": token_types,
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            "pixel_values": pixel_values,
        },
        blocks,
        image_token_id=image_token_id,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        history_visual_tokens=4,
        long_memory_visual_tokens=32,
        current_visual_tokens=64,
        merge_size=2,
        pad_token_id=0,
    )
    assert int((output["input_ids"] == image_token_id).sum()) == 68
    assert torch.equal(
        output["_nav_context_image_grid_thw"],
        torch.tensor([[1, 4, 4], [1, 16, 16]]),
    )
    assert int((output["mm_token_type_ids"] == 1).sum()) == 68
    assert torch.equal(output["pixel_values"], pixel_values[256:])
    assert int(output["_nav_tvi_mask"].sum()) == 2
    assert int((output["input_ids"] == vision_start).sum()) == 1
    assert int((output["input_ids"] == vision_end).sum()) == 1

    active_types = output["mm_token_type_ids"][output["attention_mask"].to(dtype=torch.bool)].tolist()
    visual_group_lengths = [
        end - start
        for start, end in _contiguous_value_spans(active_types, value=1)
    ]
    assert visual_group_lengths == [4, 64]


def _contiguous_value_spans(values: list[int], *, value: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(values):
        if int(values[cursor]) != int(value):
            cursor += 1
            continue
        start = cursor
        while cursor < len(values) and int(values[cursor]) == int(value):
            cursor += 1
        spans.append((start, cursor))
    return spans


def test_qwen35_rewrite_removes_wrappers_from_adjacent_cached_spans() -> None:
    image_token_id = 99
    vision_start = 98
    vision_end = 97
    wrapped = [vision_start, *([image_token_id] * 64), vision_end]
    input_ids = torch.tensor([[11, *wrapped, *wrapped, 13]], dtype=torch.long)
    token_types = torch.zeros_like(input_ids)
    token_types[input_ids == image_token_id] = 1
    output = _rewrite_qwen35_visual_spans(
        {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": token_types,
            "image_grid_thw": torch.tensor([[1, 16, 16], [1, 16, 16]]),
            "pixel_values": torch.zeros((512, 2), dtype=torch.float32),
        },
        [
            {"is_history": True, "is_cached_history": True, "is_long_memory": True},
            {"is_history": True, "is_cached_history": True},
        ],
        image_token_id=image_token_id,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        history_visual_tokens=4,
        long_memory_visual_tokens=32,
        current_visual_tokens=64,
        merge_size=2,
        pad_token_id=0,
    )

    active = output["attention_mask"].to(dtype=torch.bool)
    active_ids = output["input_ids"][active]
    active_types = output["mm_token_type_ids"][active].tolist()
    assert int((active_ids == vision_start).sum()) == 0
    assert int((active_ids == vision_end).sum()) == 0
    assert int(output["_nav_tvi_mask"].sum()) == 2
    assert [end - start for start, end in _contiguous_value_spans(active_types, value=1)] == [32, 4]
    assert output["pixel_values"].numel() == 0


def test_qwen35_cached_span_insertion_matches_placeholder_processor_output() -> None:
    image_token_id = 99
    vision_start = 98
    vision_end = 97
    wrapped = [vision_start, *([image_token_id] * 64), vision_end]
    blocks = [
        {"is_history": True, "is_cached_history": True, "is_long_memory": True},
        {"is_history": True, "is_cached_history": True},
        {"is_history": False, "is_cached_history": False},
        {"is_history": False, "is_cached_history": False},
    ]
    full_ids = torch.tensor([[11, *wrapped, *wrapped, *wrapped, *wrapped, 13]], dtype=torch.long)
    online_ids = torch.tensor([[11, *wrapped, *wrapped, 13]], dtype=torch.long)
    full_types = torch.zeros_like(full_ids)
    full_types[full_ids == image_token_id] = 1
    online_types = torch.zeros_like(online_ids)
    online_types[online_ids == image_token_id] = 1
    full_pixels = torch.arange(4 * 256 * 2, dtype=torch.float32).reshape(4 * 256, 2)
    online_pixels = full_pixels[2 * 256 :].clone()

    legacy = _rewrite_qwen35_visual_spans(
        {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "mm_token_type_ids": full_types,
            "image_grid_thw": torch.tensor([[1, 16, 16]] * 4),
            "pixel_values": full_pixels,
        },
        blocks,
        image_token_id=image_token_id,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        history_visual_tokens=4,
        long_memory_visual_tokens=32,
        current_visual_tokens=64,
        merge_size=2,
        pad_token_id=0,
    )
    optimized = _insert_qwen35_cached_visual_spans(
        {
            "input_ids": online_ids,
            "attention_mask": torch.ones_like(online_ids),
            "mm_token_type_ids": online_types,
            "image_grid_thw": torch.tensor([[1, 16, 16]] * 2),
            "pixel_values": online_pixels,
        },
        blocks,
        sample_block_counts=[4],
        image_token_id=image_token_id,
        vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        history_visual_tokens=4,
        long_memory_visual_tokens=32,
        current_visual_tokens=64,
        merge_size=2,
        pad_token_id=0,
    )

    for key in (
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "_nav_tvi_mask",
        "_nav_context_image_grid_thw",
        "_nav_original_image_grid_thw",
        "_nav_online_image_grid_thw",
        "image_grid_thw",
    ):
        assert torch.equal(optimized[key], legacy[key]), key
    assert optimized["_nav_online_indices"] == legacy["_nav_online_indices"] == [2, 3]


def test_qwen35_build_inputs_sends_only_online_images_to_processor(monkeypatch) -> None:
    class FakeInterface:
        def __init__(self) -> None:
            self.seen_images = None
            self.seen_move_to_device = None
            self.model = SimpleNamespace(
                device=torch.device("cpu"),
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        image_token_id=99,
                        vision_start_token_id=98,
                        vision_end_token_id=97,
                    ),
                    visual=SimpleNamespace(spatial_merge_size=2),
                ),
            )
            self.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))

        @staticmethod
        def build_action_placeholder_suffix(_count: int) -> str:
            return "action"

        def build_qwenvl_inputs(self, *, images, move_to_device, **_kwargs):
            self.seen_images = images
            self.seen_move_to_device = move_to_device
            return {"processor_output": torch.tensor(1)}

    cached_block = {"is_history": True, "is_cached_history": True}
    current_blocks = [
        {"is_history": False, "is_cached_history": False},
        {"is_history": False, "is_cached_history": False},
    ]
    online_images = [
        Image.new("RGB", (4, 4), color=(1, 2, 3)),
        Image.new("RGB", (4, 4), color=(4, 5, 6)),
    ]
    model = NavVLA_Qwen35_CPM.__new__(NavVLA_Qwen35_CPM)
    torch.nn.Module.__init__(model)
    model.qwen35_vl_interface = FakeInterface()
    model.action_placeholder_count = 1
    model.tvi_dim = 2
    model._visual_token_budgets = lambda: (1, 1, 4)
    model._validate_sample_profile = lambda _sample: None
    model._sample_required_cameras = lambda _sample: ["front", "left"]
    model._build_instruction = lambda _sample: "go"
    model._prepare_visual_image = lambda image: image

    captured = {}
    monkeypatch.setattr(
        qwen35_cpm_module,
        "build_navvla_cached_visual_sequence",
        lambda *_args, **_kwargs: (
            online_images,
            [cached_block, *current_blocks],
        ),
    )
    monkeypatch.setattr(
        qwen35_cpm_module,
        "_insert_qwen35_cached_visual_spans",
        lambda qwen_inputs, blocks, **kwargs: captured.update(
            qwen_inputs=qwen_inputs,
            blocks=blocks,
            kwargs=kwargs,
        )
        or {"rewritten": torch.tensor(2)},
    )
    qwen_inputs, blocks = model._build_qwen35_inputs(
        [{}],
        history_shuffle_probability=0.0,
    )

    assert model.qwen35_vl_interface.seen_images == [online_images]
    assert model.qwen35_vl_interface.seen_move_to_device is False
    assert blocks == [cached_block, *current_blocks]
    assert torch.equal(captured["qwen_inputs"]["processor_output"], torch.tensor(1))
    assert captured["blocks"] == blocks
    assert captured["kwargs"]["sample_block_counts"] == [3]
    assert torch.equal(qwen_inputs["rewritten"], torch.tensor(2))


def test_qwen35_mrope_consumes_adjacent_unwrapped_cached_spans_separately() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "spatial_merge_size": 2,
        },
    )
    model = transformers.Qwen3_5Model(config)
    input_ids = torch.tensor([[11, 0, *([99] * 4), 0, *([99] * 4), 13]], dtype=torch.long)
    token_types = torch.tensor([[0, 0, *([1] * 4), 0, *([1] * 4), 0]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    position_ids, _rope_deltas = model.get_rope_index(
        input_ids,
        mm_token_type_ids=token_types,
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long),
        attention_mask=attention_mask,
    )

    assert position_ids.shape == (3, 1, input_ids.shape[1])
    assert position_ids[:, 0, 2:6].tolist() == [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    assert position_ids[:, 0, 7:11].tolist() == [[5, 5, 5, 5], [5, 5, 6, 6], [5, 6, 5, 6]]


def test_qwen35_tvi_embeddings_replace_reserved_slots_without_changing_sequence() -> None:
    inputs = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor(
        [
            [False, True, False, False, False],
            [False, False, True, False, True],
        ]
    )
    tvi = torch.tensor(
        [
            [101.0, 102.0, 103.0],
            [201.0, 202.0, 203.0],
            [301.0, 302.0, 303.0],
        ]
    )

    output = _apply_qwen35_tvi_embeddings(inputs_embeds=inputs, tvi_mask=mask, tvi_embeds=tvi)

    assert output.shape == inputs.shape
    torch.testing.assert_close(output[mask], tvi)
    torch.testing.assert_close(output[~mask], inputs[~mask])


def test_qwen35_mmap_cache_round_trip_preserves_grid_and_stage(tmp_path) -> None:
    profile = default_qwen35_pooled_history_visual_token_profile(encoder_ckpt="checkpoint")
    index_path = write_profile_mmap_npy_cache(
        tmp_path,
        profile=profile,
        records=[
            {
                "ref": "episode/000001/front",
                "image_embeds": bf16_to_numpy_bits(
                    torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
                ),
                "image_key": "episode/000001/front",
                "grid_t": 1,
                "grid_h": 16,
                "grid_w": 16,
                "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
            }
        ],
    )
    assert index_path.exists()
    store = Qwen35PooledHistoryTokenStore(tmp_path, profile=profile.name)
    record = store.load_ref_record_batches([["episode/000001/front"]])[0]
    assert record.tokens.shape == (1, 4, 8)
    assert record.grid_thw.tolist() == [[1, 16, 16]]
    assert record.cache_stage == QWEN35_POOLED_HISTORY_CACHE_STAGE
    assert record.encoder_ckpt == "checkpoint"
    assert record.storage_encoding == BFLOAT16_BITS_STORAGE_ENCODING
    collator = NavVLACPMCollator()
    batches, _grids, _stage, _encoder, encoding = collator._load_token_batches(
        [
            {
                "metadata": {
                    "dataset_root": str(tmp_path),
                    "visual_token_profile": profile.name,
                    "history_token_refs": ["episode/000001/front"],
                }
            }
        ],
        metadata_key="history_token_refs",
    )
    padded, _mask = _pad_token_batches(batches)
    assert padded.dtype == np.uint16
    assert encoding == BFLOAT16_BITS_STORAGE_ENCODING
    assert np.array_equal(padded[0, 0], record.tokens[0])
    collator._stores.clear()
    store.close()


def test_qwen35_store_rejects_shard_dtype_that_disagrees_with_manifest(tmp_path) -> None:
    profile = default_qwen35_pooled_history_visual_token_profile(encoder_ckpt="checkpoint")
    write_profile_mmap_npy_cache(
        tmp_path,
        profile=profile,
        records=[
            {
                "ref": "episode/000001/front",
                "image_embeds": np.zeros((4, 8), dtype=np.uint16),
                "image_key": "episode/000001/front",
                "grid_t": 1,
                "grid_h": 16,
                "grid_w": 16,
                "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
            }
        ],
    )
    shard_path = next((tmp_path / "cache" / "visual_tokens" / profile.name / "shards").glob("*.npy"))
    np.save(shard_path, np.zeros((1, 4, 8), dtype=np.float16))

    store = Qwen35PooledHistoryTokenStore(tmp_path, profile=profile.name)
    with pytest.raises(TypeError, match="does not match manifest dtype"):
        store.load_ref_record_batches([["episode/000001/front"]])
    store.close()


def test_qwen35_max_text_tokens_only_truncates_text() -> None:
    interface = _QWen3_5_VL_Interface.__new__(_QWen3_5_VL_Interface)

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert not add_special_tokens
            return [int(value) for value in str(text).split()]

        def decode(self, values, **_kwargs):
            return " ".join(str(value) for value in values)

    interface.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    interface.config = SimpleNamespace(framework=SimpleNamespace(qwenvl={"max_text_tokens": 3}))
    assert interface._truncate_text("1 2 3 4 5") == "1 2 3"


def test_online_qwen35_cache_payload_matches_offline_contract() -> None:
    tokens = np.zeros((4, 8), dtype=np.uint16)
    observation = {
        "navvla_online_visual_tokens": {
            "front": {
                "tokens": tokens,
                "grid_thw": np.asarray([1, 16, 16]),
                "cache_stage": QWEN35_POOLED_HISTORY_CACHE_STAGE,
                "visual_token_profile": "qwen3_5_4b_postmerge_pool4_256_mmap",
                "encoder_ckpt": "Qwen/Qwen3.5-4B",
                "storage_encoding": BFLOAT16_BITS_STORAGE_ENCODING,
            }
        }
    }
    record = _online_visual_record(observation, "front")
    assert record is not None
    assert record["tokens"].shape == (4, 8)
    assert record["grid_thw"].tolist() == [1, 16, 16]
    assert record["cache_stage"] == QWEN35_POOLED_HISTORY_CACHE_STAGE
    assert record["encoder_ckpt"] == "Qwen/Qwen3.5-4B"
    assert record["storage_encoding"] == BFLOAT16_BITS_STORAGE_ENCODING
    assert QWEN35_LONG_MEMORY_SOURCE_POOLED_STAGE != record["cache_stage"]


def test_online_qwen35_cache_rejects_float_tokens_labeled_as_bfloat16_bits() -> None:
    observation = {
        "navvla_online_visual_tokens": {
            "front": {
                "tokens": np.zeros((4, 8), dtype=np.float16),
                "storage_encoding": BFLOAT16_BITS_STORAGE_ENCODING,
            }
        }
    }
    with pytest.raises(TypeError, match="numpy uint16"):
        _online_visual_record(observation, "front")
