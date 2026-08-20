from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from starVLA.training.train_starvla import VLATrainer


def _trainer(model: torch.nn.Module) -> VLATrainer:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(seed=123)
    trainer.model = model
    return trainer


def test_fixed_training_batch_mse_evaluation_is_removed() -> None:
    assert not hasattr(VLATrainer, "eval_action_model")


def test_openloop_eval_context_restores_model_modes_and_rng() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model.train()
    model[1].eval()
    trainer = _trainer(model)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    expected_after = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    with trainer._model_evaluation_mode():
        with trainer._fixed_evaluation_rng():
            first_eval_draw = (random.random(), float(np.random.random()), torch.rand(3))
            assert model.training is False
            assert model[1].training is False

    actual_after = (random.random(), float(np.random.random()), torch.rand(3))
    with trainer._model_evaluation_mode():
        with trainer._fixed_evaluation_rng():
            second_eval_draw = (random.random(), float(np.random.random()), torch.rand(3))

    assert first_eval_draw[0] == second_eval_draw[0]
    assert first_eval_draw[1] == second_eval_draw[1]
    torch.testing.assert_close(first_eval_draw[2], second_eval_draw[2])
    assert actual_after[0] == expected_after[0]
    assert actual_after[1] == expected_after[1]
    torch.testing.assert_close(actual_after[2], expected_after[2])
    assert model.training is True
    assert model[1].training is False


def test_openloop_eval_context_restores_original_eval_mode() -> None:
    model = torch.nn.Linear(2, 2)
    model.eval()
    trainer = _trainer(model)

    with trainer._model_evaluation_mode():
        assert model.training is False

    assert model.training is False
