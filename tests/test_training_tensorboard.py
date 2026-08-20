from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from starVLA.training.train_starvla import VLATrainer


def _trainer(*, trackers=None, is_main_process=True) -> VLATrainer:
    trainer = VLATrainer.__new__(VLATrainer)
    config = {
        "output_dir": "/tmp/navvla-run",
        "run_id": "test-run",
        "wandb_project": "test-project",
        "wandb_entity": "test-entity",
        "trainer": SimpleNamespace(logging_frequency=1),
    }
    if trackers is not None:
        config["trackers"] = trackers
    trainer.config = SimpleNamespace(**config)
    trainer.accelerator = SimpleNamespace(is_main_process=is_main_process)
    trainer.vla_epoch_count = 0
    trainer.vla_batch_in_epoch = 0
    trainer.tensorboard_writer = None
    return trainer


def test_missing_tracker_configuration_preserves_wandb_default() -> None:
    trainer = _trainer()

    assert trainer._configured_trackers() == frozenset({"wandb"})


def test_existing_jsonl_and_wandb_configuration_remains_valid() -> None:
    trainer = _trainer(trackers=["jsonl", "wandb"])

    assert trainer._configured_trackers() == frozenset({"jsonl", "wandb"})


def test_tensorboard_only_initialization_does_not_start_wandb() -> None:
    trainer = _trainer(trackers=["tensorboard"])
    writer = mock.Mock()

    with mock.patch("starVLA.training.train_starvla.wandb.init") as wandb_init:
        with mock.patch(
            "torch.utils.tensorboard.SummaryWriter",
            return_value=writer,
        ) as writer_cls:
            trainer._init_trackers()

    writer_cls.assert_called_once_with(log_dir="/tmp/navvla-run/tensorboard")
    wandb_init.assert_not_called()
    assert trainer.tensorboard_writer is writer


def test_tensorboard_logs_numeric_metrics_and_flushes_without_wandb() -> None:
    trainer = _trainer(trackers=["tensorboard"])
    trainer.completed_steps = 7
    trainer.tensorboard_writer = mock.Mock()
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [2.5e-5])
    trainer.vla_train_dataloader = [object(), object()]
    trainer.vla_epoch_count = 1
    trainer.vla_batch_in_epoch = 1

    with mock.patch("starVLA.training.train_starvla.wandb.log") as wandb_log, mock.patch(
        "starVLA.training.train_starvla.logger.info"
    ) as logger_info:
        trainer._log_metrics(
            {
                "action_dit_loss": 1.25,
                "data_time": 0.2,
                "diagnostic_text": "ignored",
            }
        )

    trainer.tensorboard_writer.add_scalar.assert_has_calls(
        [
            mock.call("train/action_dit_loss", 1.25, 7),
            mock.call("train/data_time", 0.2, 7),
            mock.call("train/learning_rate", 2.5e-5, 7),
            mock.call("train/epoch", 1.5, 7),
            mock.call("train/batch_in_epoch", 1.0, 7),
            mock.call("train/batches_per_epoch", 2.0, 7),
        ],
        any_order=True,
    )
    assert trainer.tensorboard_writer.add_scalar.call_count == 6
    trainer.tensorboard_writer.flush.assert_called_once_with()
    wandb_log.assert_not_called()
    logger_info.assert_called_once_with(
        "Step 7 | Epoch 1.5000 | Batch 1/2 | Metrics: "
        "{'action_dit_loss': 1.25, 'data_time': 0.2, "
        "'diagnostic_text': 'ignored', 'learning_rate': 2.5e-05}"
    )


def test_tensorboard_writes_a_readable_event_file(tmp_path) -> None:
    trainer = _trainer(trackers=["tensorboard"])
    trainer.config.output_dir = str(tmp_path)
    trainer.completed_steps = 3
    trainer.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [1.0e-4])
    trainer.vla_train_dataloader = [object()]

    trainer._init_trackers()
    with mock.patch("starVLA.training.train_starvla.logger.info"):
        trainer._log_metrics({"action_dit_loss": 0.75})
    trainer._finish_trackers()

    events = EventAccumulator(str(tmp_path / "tensorboard"))
    events.Reload()
    assert events.Scalars("train/action_dit_loss")[0].step == 3
    assert events.Scalars("train/action_dit_loss")[0].value == pytest.approx(0.75)


def test_finish_trackers_closes_tensorboard_without_finishing_wandb() -> None:
    trainer = _trainer(trackers=["tensorboard"])
    trainer.tensorboard_writer = mock.Mock()

    with mock.patch("starVLA.training.train_starvla.wandb.finish") as wandb_finish:
        trainer._finish_trackers()

    trainer.tensorboard_writer.close.assert_called_once_with()
    wandb_finish.assert_not_called()


def test_unsupported_tracker_fails_explicitly() -> None:
    trainer = _trainer(trackers=["tensorboard", "mlflow"])

    with pytest.raises(ValueError, match="Unsupported trackers: mlflow"):
        trainer._configured_trackers()
