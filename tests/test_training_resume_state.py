from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.utils.data import DataLoader

from starVLA.training.train_starvla import VLATrainer
from starVLA.training.trainer_utils.resume_state import (
    TrainingProgressState,
    find_latest_training_state_checkpoint,
    make_resume_dataloader_iterator,
    set_dataloader_epoch,
)


class _BucketDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 48) -> None:
        self.length = int(length)
        self.episode_ranges = [SimpleNamespace(dataset_index=0, episode_index=0, start=0, length=self.length)]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> int:
        return int(index)

    def encode_sample_index(self, dataset_index: int, sample_index: int) -> int:
        if int(dataset_index) != 0:
            raise IndexError(dataset_index)
        return int(sample_index)

    def history_frame_capacity_for_dataset(self, dataset_index: int) -> int:
        if int(dataset_index) != 0:
            raise IndexError(dataset_index)
        return self.length

    def prepare_history_frame_counts(self) -> None:
        return None

    def history_frame_count(self, index: int) -> int:
        return int(index)


def _prepared_bucket_loader(process_index: int, *, seed: int = 71):
    from accelerate.data_loader import prepare_data_loader
    from starVLA.dataloader.cpm_lerobot.sampler import LengthBucketedEpisodeBatchSampler

    dataset = _BucketDataset()
    sampler = LengthBucketedEpisodeBatchSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=seed,
        drop_last=True,
        bucket_width=8,
        buffer_size=8,
        sync_group_size=2,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    prepared = prepare_data_loader(
        dataloader,
        num_processes=2,
        process_index=int(process_index),
        split_batches=False,
        put_on_device=False,
        even_batches=False,
    )
    return prepared, sampler


def test_training_progress_state_round_trips_dataloader_position():
    state = TrainingProgressState()
    state.completed_steps = 7
    state.vla_epoch_count = 2
    state.vla_batch_in_epoch = 5
    state.global_batches_consumed = 23

    restored = TrainingProgressState()
    restored.load_state_dict(state.state_dict())

    assert restored.completed_steps == 7
    assert restored.vla_epoch_count == 2
    assert restored.vla_batch_in_epoch == 5
    assert restored.global_batches_consumed == 23


def test_find_latest_training_state_checkpoint_prefers_highest_step(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "steps_20_state").mkdir()
    (checkpoints / "steps_3_state").mkdir()
    (checkpoints / "steps_20_state" / "trainer_state.json").write_text(
        '{"completed_steps": 20}\n', encoding="utf-8"
    )
    (checkpoints / "steps_3_state" / "trainer_state.json").write_text(
        '{"completed_steps": 3}\n', encoding="utf-8"
    )
    (checkpoints / "steps_100_pytorch_model.pt").write_bytes(b"model-only")
    (checkpoints / "not_a_checkpoint").mkdir()

    latest_path, latest_step = find_latest_training_state_checkpoint(checkpoints)

    assert latest_path == checkpoints / "steps_20_state"
    assert latest_step == 20


def test_find_latest_training_state_checkpoint_ignores_incomplete_state_dir(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    complete = checkpoints / "steps_20_state"
    complete.mkdir()
    (complete / "trainer_state.json").write_text('{"completed_steps": 20}\n', encoding="utf-8")
    (checkpoints / "steps_30_state").mkdir()

    latest_path, latest_step = find_latest_training_state_checkpoint(checkpoints)

    assert latest_path == complete
    assert latest_step == 20


def test_find_latest_training_state_checkpoint_ignores_invalid_or_mismatched_marker(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    complete = checkpoints / "steps_20_state"
    complete.mkdir()
    (complete / "trainer_state.json").write_text('{"completed_steps": 20}\n', encoding="utf-8")
    invalid = checkpoints / "steps_30_state"
    invalid.mkdir()
    (invalid / "trainer_state.json").write_text("not-json\n", encoding="utf-8")
    mismatched = checkpoints / "steps_40_state"
    mismatched.mkdir()
    (mismatched / "trainer_state.json").write_text('{"completed_steps": 39}\n', encoding="utf-8")

    latest_path, latest_step = find_latest_training_state_checkpoint(checkpoints)

    assert latest_path == complete
    assert latest_step == 20


def test_explicit_resume_state_checkpoint_requires_completed_state(tmp_path: Path):
    checkpoint = tmp_path / "steps_20_state"
    checkpoint.mkdir()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.checkpoint_dir = str(tmp_path)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_checkpoint=str(checkpoint))
    )

    with pytest.raises(ValueError, match="incomplete training-state checkpoint"):
        trainer._resolve_resume_state_checkpoint()


def test_explicit_resume_state_checkpoint_rejects_mismatched_step_marker(tmp_path: Path):
    checkpoint = tmp_path / "steps_20_state"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"completed_steps": 19}\n', encoding="utf-8")
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.checkpoint_dir = str(tmp_path)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(resume_checkpoint=str(checkpoint))
    )

    with pytest.raises(ValueError, match="incomplete training-state checkpoint"):
        trainer._resolve_resume_state_checkpoint()


def test_make_resume_dataloader_iterator_skips_consumed_batches():
    dataloader = DataLoader(torch.arange(10), batch_size=2, shuffle=False)
    progress = TrainingProgressState(vla_epoch_count=0, vla_batch_in_epoch=2)

    iterator = make_resume_dataloader_iterator(
        dataloader,
        progress,
        skip_first_batches_fn=_skip_first_batches_for_test,
    )

    assert next(iterator).tolist() == [4, 5]


def test_set_dataloader_epoch_reaches_length_sampler_through_accelerate_wrappers() -> None:
    dataloader, sampler = _prepared_bucket_loader(process_index=0)

    set_dataloader_epoch(dataloader, 0)
    epoch_zero = [batch.tolist() for batch in dataloader]
    set_dataloader_epoch(dataloader, 1)
    assert sampler.epoch == sampler._source.epoch == dataloader.iteration == 1
    epoch_one = [batch.tolist() for batch in dataloader]

    assert sampler.epoch == sampler._source.epoch == 1
    assert dataloader.iteration == 2
    assert epoch_one != epoch_zero


def test_accelerate_ranks_share_global_order_and_receive_distinct_shards() -> None:
    rank_zero_loader, rank_zero_sampler = _prepared_bucket_loader(process_index=0)
    rank_one_loader, rank_one_sampler = _prepared_bucket_loader(process_index=1)

    set_dataloader_epoch(rank_zero_loader, 2)
    set_dataloader_epoch(rank_one_loader, 2)
    rank_zero = [batch.tolist() for batch in rank_zero_loader]
    rank_one = [batch.tolist() for batch in rank_one_loader]

    expected_sampler = _prepared_bucket_loader(process_index=0)[1]
    expected_sampler.set_epoch(2)
    expected = list(expected_sampler)

    assert rank_zero_sampler.epoch == rank_one_sampler.epoch == 2
    assert rank_zero == expected[0::2]
    assert rank_one == expected[1::2]
    assert set(index for batch in rank_zero for index in batch).isdisjoint(
        index for batch in rank_one for index in batch
    )


def test_resume_after_accelerate_skip_matches_uninterrupted_epoch() -> None:
    from accelerate.data_loader import skip_first_batches

    baseline_loader, _ = _prepared_bucket_loader(process_index=0)
    set_dataloader_epoch(baseline_loader, 3)
    baseline = [batch.tolist() for batch in baseline_loader]

    resumed_loader, resumed_sampler = _prepared_bucket_loader(process_index=0)
    progress = TrainingProgressState(vla_epoch_count=3, vla_batch_in_epoch=2)
    resumed = [
        batch.tolist()
        for batch in make_resume_dataloader_iterator(
            resumed_loader,
            progress,
            skip_first_batches_fn=skip_first_batches,
        )
    ]

    assert resumed_sampler.epoch == 3
    assert resumed == baseline[2:]


def _skip_first_batches_for_test(dataloader, num_batches):
    class _Skipped:
        def __iter__(self):
            iterator = iter(dataloader)
            for _ in range(num_batches):
                next(iterator)
            return iterator

    return _Skipped()


def test_vla_trainer_create_iterator_resumes_from_saved_batch_offset():
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.vla_train_dataloader = DataLoader(torch.arange(10), batch_size=2, shuffle=False)
    trainer.accelerator = SimpleNamespace(skip_first_batches=_skip_first_batches_for_test)
    trainer.training_progress = TrainingProgressState(vla_epoch_count=0, vla_batch_in_epoch=2)
    trainer.completed_steps = 0
    trainer.vla_epoch_count = 0
    trainer.vla_batch_in_epoch = 2
    trainer.global_batches_consumed = 2

    trainer._create_data_iterators()
    batch = trainer._get_next_batch()

    assert batch.tolist() == [4, 5]
    assert trainer.vla_batch_in_epoch == 3
    assert trainer.global_batches_consumed == 3


@pytest.mark.parametrize(
    ("save_training_state", "force_training_state"),
    [(True, False), (False, True)],
)
def test_vla_trainer_save_checkpoint_writes_accelerate_state_and_legacy_model(
    tmp_path: Path,
    save_training_state: bool,
    force_training_state: bool,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = 4
    trainer.vla_epoch_count = 1
    trainer.vla_batch_in_epoch = 2
    trainer.global_batches_consumed = 6
    trainer.training_progress = TrainingProgressState()
    trainer.checkpoint_dir = str(checkpoint_dir)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.config = SimpleNamespace(
        output_dir=str(tmp_path),
        trainer=SimpleNamespace(
            save_format="pt",
            save_training_state=save_training_state,
        ),
    )

    saved_state_dirs = []
    marker_states_at_barrier = []

    class _Accelerator:
        is_main_process = True

        def get_state_dict(self, model):
            return {"weight": torch.ones(1)}

        def save_state(self, output_dir, safe_serialization=True):
            saved_state_dirs.append((Path(output_dir), safe_serialization))
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        def print(self, *_args, **_kwargs):
            return None

        def wait_for_everyone(self):
            marker_states_at_barrier.append(
                (checkpoint_dir / "steps_4_state" / "trainer_state.json").exists()
            )

    trainer.accelerator = _Accelerator()

    trainer._save_checkpoint(force_training_state=force_training_state)

    assert saved_state_dirs == [(checkpoint_dir / "steps_4_state", False)]
    assert marker_states_at_barrier == [False, True]
    assert (checkpoint_dir / "steps_4_pytorch_model.pt").exists()
    trainer_state = (checkpoint_dir / "steps_4_state" / "trainer_state.json").read_text(encoding="utf-8")
    assert '"completed_steps": 4' in trainer_state
    assert '"vla_batch_in_epoch": 2' in trainer_state


def test_vla_trainer_stop_at_step_saves_full_state_once_and_exits() -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            max_train_steps=10,
            stop_at_step=2,
            save_on_stop=True,
            save_interval=2,
        )
    )
    trainer.accelerator = SimpleNamespace(
        sync_gradients=True,
        is_local_main_process=False,
        wait_for_everyone=mock.Mock(),
    )
    trainer._log_training_config = mock.Mock()
    trainer._should_run_openloop_eval = mock.Mock(return_value=False)
    trainer._create_data_iterators = mock.Mock()
    trainer._get_next_batch = mock.Mock(return_value={"batch": 1})
    trainer._train_step = mock.Mock(return_value={"action_dit_loss": 1.0})
    trainer._log_metrics = mock.Mock()
    trainer._save_checkpoint = mock.Mock()
    trainer._finish_trackers = mock.Mock()
    trainer._finalize_training = mock.Mock()

    trainer.train()

    assert trainer.completed_steps == 2
    assert trainer._train_step.call_count == 2
    trainer._save_checkpoint.assert_called_once_with(force_training_state=True)
    trainer._finish_trackers.assert_called_once_with()
    trainer.accelerator.wait_for_everyone.assert_called_once_with()
    trainer._finalize_training.assert_not_called()


@pytest.mark.parametrize("stop_at_step", [0, -1, 11])
def test_vla_trainer_rejects_invalid_stop_at_step(stop_at_step: int) -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(max_train_steps=10, stop_at_step=stop_at_step)
    )

    with pytest.raises(ValueError, match="stop_at_step"):
        trainer._configured_stop_at_step()


def test_vla_trainer_passes_completed_optimization_step_to_model() -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.completed_steps = 37
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(
            max_train_steps=100,
            gradient_clipping=None,
        )
    )
    calls = []

    class _Model(torch.nn.Module):
        def forward(self, batch, **kwargs):
            calls.append((batch, kwargs))
            parameter = next(self.parameters())
            loss = parameter.sum() * 0.0 + 1.0
            return {"action_loss": loss}

    model = _Model()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(())))
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.0)

    class _Accumulate:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    trainer.accelerator = SimpleNamespace(
        accumulate=lambda _model: _Accumulate(),
        backward=lambda loss: loss.backward(),
        sync_gradients=True,
    )

    trainer._train_step({"batch": 1})

    assert calls == [
        (
            {"batch": 1},
            {"training_step": 37, "total_training_steps": 100},
        )
    ]


def test_prepare_training_keeps_scheduler_native_and_registers_its_state() -> None:
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = SimpleNamespace(
        seed=42,
        trainer=SimpleNamespace(freeze_modules=None),
    )
    trainer.model = object()
    trainer.optimizer = object()
    trainer.vla_train_dataloader = object()
    trainer.lr_scheduler = mock.Mock()
    trainer.training_progress = TrainingProgressState()
    trainer.resume_state_checkpoint_path = None
    trainer.accelerator = SimpleNamespace(register_for_checkpointing=mock.Mock())
    trainer._save_initial_configs = mock.Mock()
    trainer._init_checkpointing = mock.Mock()
    trainer._adjust_lr_scheduler_for_resume = mock.Mock()
    trainer.freeze_backbones = mock.Mock(side_effect=lambda model, freeze_modules: model)
    trainer.print_trainable_parameters = mock.Mock()
    trainer.setup_distributed_training = mock.Mock(
        return_value=(trainer.model, trainer.optimizer, trainer.vla_train_dataloader)
    )
    trainer._init_trackers = mock.Mock()

    with mock.patch("starVLA.training.train_starvla.dist.is_initialized", return_value=False):
        trainer.prepare_training()

    trainer.setup_distributed_training.assert_called_once_with(
        trainer.accelerator,
        trainer.model,
        trainer.optimizer,
        trainer.vla_train_dataloader,
    )
    trainer.accelerator.register_for_checkpointing.assert_called_once_with(
        trainer.training_progress,
        trainer.lr_scheduler,
    )


def test_scheduler_state_resume_matches_uninterrupted_warmup() -> None:
    from transformers import get_scheduler

    def build():
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 2.5e-5}])
        scheduler = get_scheduler(
            "cosine_with_min_lr",
            optimizer,
            num_warmup_steps=1005,
            num_training_steps=33505,
            scheduler_specific_kwargs={"min_lr": 1.0e-6},
        )
        return optimizer, scheduler

    optimizer, scheduler = build()
    for _ in range(73):
        optimizer.step()
        scheduler.step()
    saved_state = scheduler.state_dict()

    resumed_optimizer, resumed_scheduler = build()
    resumed_scheduler.load_state_dict(saved_state)
    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_scheduler.last_epoch == scheduler.last_epoch == 74
    assert resumed_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
