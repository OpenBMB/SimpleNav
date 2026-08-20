from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from transformers import BertConfig, BertModel


def _load_bertwarper_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = (
        repo_root
        / "NavVLAeval"
        / "traveluav"
        / "3dparty"
        / "groundingdino"
        / "models"
        / "GroundingDINO"
        / "bertwarper.py"
    )
    spec = importlib.util.spec_from_file_location("traveluav_groundingdino_bertwarper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bert_model_warper_provides_head_mask_for_transformers_without_instance_method() -> None:
    module = _load_bertwarper_module()
    bert = BertModel(
        BertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
        )
    )
    assert not hasattr(bert, "get_head_mask")

    wrapper = module.BertModelWarper(bert)
    head_mask = wrapper.get_head_mask(None, wrapper.config.num_hidden_layers)

    assert head_mask == [None]


def test_bert_model_warper_forward_uses_transformers_5_attention_mask_signature() -> None:
    module = _load_bertwarper_module()
    bert = BertModel(
        BertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=16,
        )
    )
    wrapper = module.BertModelWarper(bert)

    output = wrapper(
        input_ids=torch.ones((1, 3), dtype=torch.long),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        return_dict=True,
    )

    assert output.last_hidden_state.shape == (1, 3, 8)
