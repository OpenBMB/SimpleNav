# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
from contextlib import contextmanager
import json
import numbers
import os
import random
import time
from pathlib import Path
from typing import Any, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.openloop_eval import (
    OpenLoopMetricAccumulator,
    build_openloop_eval_loaders,
    flatten_openloop_metrics,
    run_openloop_eval_loader,
    write_openloop_metrics,
)
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.resume_state import (
    TrainingProgressState,
    completed_training_state_step,
    find_latest_training_state_checkpoint,
    make_resume_dataloader_iterator,
)
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def build_accelerator(cfg) -> Accelerator:
    gradient_accumulation_steps = int(cfg.trainer.gradient_accumulation_steps)
    deepspeed_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=gradient_accumulation_steps,
        sync_each_batch=True,
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)
    return accelerator


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.vla_epoch_count = 0
        self.vla_batch_in_epoch = 0
        self.global_batches_consumed = 0
        self.training_progress = TrainingProgressState()
        self.resume_state_checkpoint_path = None
        self.openloop_eval_loaders = []
        self._last_openloop_eval_step = None
        self.tensorboard_writer = None
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        if self.resume_state_checkpoint_path is None:
            self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )
        self.accelerator.register_for_checkpointing(self.training_progress, self.lr_scheduler)
        if self.resume_state_checkpoint_path is not None:
            self._load_training_state_checkpoint(self.resume_state_checkpoint_path)

        self._prepare_openloop_eval()
        self._init_trackers()

    def _prepare_openloop_eval(self):
        trainer_cfg = self.config.trainer
        trainer_get = getattr(trainer_cfg, "get", None)
        openloop_cfg = (
            trainer_get("openloop_eval", None)
            if callable(trainer_get)
            else getattr(trainer_cfg, "openloop_eval", None)
        )
        openloop_get = getattr(openloop_cfg, "get", None)
        enabled = (
            openloop_get("enabled", False)
            if callable(openloop_get)
            else getattr(openloop_cfg, "enabled", False)
        )
        if openloop_cfg is None or not bool(enabled):
            self.openloop_eval_loaders = []
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.openloop_eval_loaders = build_openloop_eval_loaders(
            self.config.datasets.vla_data,
            openloop_cfg,
            rank=rank,
            world_size=world_size,
        )

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _configured_trackers(self) -> frozenset[str]:
        configured = getattr(self.config, "trackers", None)
        if configured is None:
            configured = ["wandb"]
        elif isinstance(configured, str):
            configured = [configured]

        trackers = frozenset(str(name).strip().lower() for name in configured if str(name).strip())
        unsupported = trackers - {"jsonl", "wandb", "tensorboard"}
        if unsupported:
            raise ValueError(f"Unsupported trackers: {', '.join(sorted(unsupported))}")
        return trackers

    def _init_trackers(self):
        """Initialize configured metric trackers on the global main process."""
        trackers = self._configured_trackers()
        if not self.accelerator.is_main_process:
            return

        if "wandb" in trackers:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

        if "tensorboard" in trackers:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard_writer = SummaryWriter(
                log_dir=os.path.join(self.config.output_dir, "tensorboard")
            )

    def _finish_trackers(self):
        """Close configured metric trackers on the global main process."""
        if not self.accelerator.is_main_process:
            return

        trackers = self._configured_trackers()
        if "wandb" in trackers:
            wandb.finish()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_training_state = bool(getattr(self.config.trainer, "resume_training_state", is_resume))
        self.resume_from_checkpoint = pretrained_checkpoint

        if resume_training_state:
            resume_state_checkpoint, completed_steps = self._resolve_resume_state_checkpoint()
            if resume_state_checkpoint is not None:
                self.resume_state_checkpoint_path = resume_state_checkpoint
                self.completed_steps = completed_steps
                logger.info(
                    f"Will resume full training state from checkpoint: {resume_state_checkpoint}, "
                    f"steps: {completed_steps}"
                )
                return
            if is_resume:
                logger.warning(
                    f"No valid full training-state checkpoint found in {self.checkpoint_dir}. "
                    "Falling back to legacy model-only resume."
                )

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _resolve_resume_state_checkpoint(self) -> tuple[str | None, int]:
        resume_checkpoint = getattr(self.config.trainer, "resume_checkpoint", "auto")
        if resume_checkpoint in (None, "", "auto"):
            checkpoint_path, step = find_latest_training_state_checkpoint(self.checkpoint_dir)
            return (str(checkpoint_path), step) if checkpoint_path is not None else (None, 0)

        checkpoint_path = Path(str(resume_checkpoint))
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(self.checkpoint_dir) / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {checkpoint_path}")
        completed_step = completed_training_state_step(checkpoint_path)
        if completed_step is None:
            raise ValueError(f"incomplete training-state checkpoint: {checkpoint_path}")
        return str(checkpoint_path), completed_step

    def _step_from_training_state_path(self, checkpoint_path: Path) -> int:
        trainer_state_path = checkpoint_path / "trainer_state.json"
        if trainer_state_path.exists():
            payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            return int(payload.get("completed_steps", 0))
        name = checkpoint_path.name
        if name.startswith("steps_") and name.endswith("_state"):
            return int(name[len("steps_") : -len("_state")])
        return 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _load_training_state_checkpoint(self, checkpoint_path):
        """Load full Accelerator/DeepSpeed state plus trainer progress."""
        self.accelerator.load_state(checkpoint_path)
        self._apply_training_progress()
        self.accelerator.print(f"Resumed full training state from checkpoint: {checkpoint_path}")

    def _capture_training_progress(self):
        self.training_progress.completed_steps = int(self.completed_steps)
        self.training_progress.vla_epoch_count = int(getattr(self, "vla_epoch_count", 0))
        self.training_progress.vla_batch_in_epoch = int(getattr(self, "vla_batch_in_epoch", 0))
        self.training_progress.global_batches_consumed = int(getattr(self, "global_batches_consumed", 0))

    def _apply_training_progress(self):
        self.completed_steps = int(self.training_progress.completed_steps)
        self.vla_epoch_count = int(self.training_progress.vla_epoch_count)
        self.vla_batch_in_epoch = int(self.training_progress.vla_batch_in_epoch)
        self.global_batches_consumed = int(self.training_progress.global_batches_consumed)

    def _save_training_progress_json(self, state_dir):
        trainer_state_path = Path(state_dir) / "trainer_state.json"
        trainer_state_path.write_text(
            json.dumps(self.training_progress.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _save_checkpoint(self, *, force_training_state: bool = False):
        """Save current training state."""
        self._capture_training_progress()
        save_format = getattr(self.config.trainer, "save_format", "pt")
        save_training_state = force_training_state or bool(
            getattr(self.config.trainer, "save_training_state", False)
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

        if save_training_state:
            state_dir = checkpoint_path + "_state"
            self.accelerator.save_state(state_dir, safe_serialization=(save_format == "safetensors"))
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_training_progress_json(state_dir)

        if self.accelerator.is_main_process:
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return
        if not self.accelerator.is_main_process:
            return

        metrics = dict(metrics)
        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        batches_per_epoch = len(self.vla_train_dataloader)
        batch_in_epoch = int(self.vla_batch_in_epoch)
        metrics["epoch"] = round(
            int(self.vla_epoch_count) + batch_in_epoch / batches_per_epoch,
            4,
        )
        metrics["batch_in_epoch"] = batch_in_epoch
        metrics["batches_per_epoch"] = batches_per_epoch
        trackers = self._configured_trackers()

        if "wandb" in trackers:
            wandb.log(metrics, step=self.completed_steps)
        if self.tensorboard_writer is not None:
            for name, value in metrics.items():
                if isinstance(value, numbers.Number) and not isinstance(value, bool):
                    self.tensorboard_writer.add_scalar(
                        f"train/{name}", float(value), self.completed_steps
                    )
            self.tensorboard_writer.flush()

        display_metrics = {
            name: value
            for name, value in metrics.items()
            if name not in {"epoch", "batch_in_epoch", "batches_per_epoch"}
        }
        logger.info(
            f"Step {self.completed_steps} | Epoch {metrics['epoch']:.4f} "
            f"| Batch {batch_in_epoch}/{batches_per_epoch} | Metrics: {display_metrics}"
        )

    def _create_data_iterators(self):
        """Create data iterators."""
        self._capture_training_progress()
        self.vla_iter = make_resume_dataloader_iterator(
            self.vla_train_dataloader,
            self.training_progress,
            skip_first_batches_fn=self.accelerator.skip_first_batches,
        )

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            self.vla_batch_in_epoch = 0
            batch_vla = next(self.vla_iter)

        self.vla_batch_in_epoch += 1
        self.global_batches_consumed += 1
        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        stop_at_step = self._configured_stop_at_step()
        save_on_stop = bool(getattr(self.config.trainer, "save_on_stop", False))
        if stop_at_step is not None and self.completed_steps >= stop_at_step:
            logger.info(
                f"Configured stop step {stop_at_step} has already been reached at "
                f"step {self.completed_steps}; exiting without another training step."
            )
            self._finish_trackers()
            self.accelerator.wait_for_everyone()
            return
        if self._should_run_openloop_eval(step=self.completed_steps, fresh_only=True):
            self.eval_openloop()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            did_sync_gradients = bool(self.accelerator.sync_gradients)
            if did_sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if did_sync_gradients:
                if self._should_run_openloop_eval(step=self.completed_steps):
                    step_metrics.update(self.eval_openloop())

                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                should_stop = stop_at_step is not None and self.completed_steps >= stop_at_step
                periodic_save = (
                    self.completed_steps % self.config.trainer.save_interval == 0
                    and self.completed_steps > 0
                )
                if periodic_save or (should_stop and save_on_stop):
                    self._save_checkpoint(force_training_state=should_stop and save_on_stop)
                if should_stop:
                    progress_bar.close()
                    logger.info(f"Reached configured stop_at_step={stop_at_step}; training stopped.")
                    self._finish_trackers()
                    self.accelerator.wait_for_everyone()
                    return

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _configured_stop_at_step(self) -> int | None:
        value = getattr(self.config.trainer, "stop_at_step", None)
        if value is None:
            return None
        stop_at_step = int(value)
        max_train_steps = int(self.config.trainer.max_train_steps)
        if stop_at_step <= 0:
            raise ValueError(f"stop_at_step must be positive, got {stop_at_step}")
        if stop_at_step > max_train_steps:
            raise ValueError(
                f"stop_at_step ({stop_at_step}) cannot exceed max_train_steps "
                f"({max_train_steps})"
            )
        return stop_at_step

    def _should_run_openloop_eval(self, *, step: int, fresh_only: bool = False) -> bool:
        if not self.openloop_eval_loaders:
            return False
        openloop_cfg = self.config.trainer.openloop_eval
        step = int(step)
        if step == 0:
            return bool(openloop_cfg.get("run_at_step_zero", True)) and (
                not fresh_only or self.resume_state_checkpoint_path is None
            ) and self._last_openloop_eval_step != 0
        if fresh_only:
            return False
        interval = int(self.config.trainer.eval_interval)
        if interval <= 0:
            raise ValueError(f"open-loop evaluation interval must be positive, got {interval}")
        return step % interval == 0 and self._last_openloop_eval_step != step

    def eval_openloop(self) -> dict[str, float]:
        if not self.openloop_eval_loaders:
            return {}
        local_results: list[dict[str, Any]] = []
        with self._model_evaluation_mode():
            with self._fixed_evaluation_rng(), torch.inference_mode():
                model = self.accelerator.unwrap_model(self.model)
                for loader in self.openloop_eval_loaders:
                    payload, duration = run_openloop_eval_loader(model=model, loader=loader)
                    local_results.append(
                        {
                            "dataset": loader.dataset_name,
                            "split": loader.split,
                            "target_count": loader.target_count,
                            "duration_seconds": duration,
                            "payload": payload,
                        }
                    )

        gathered = self._gather_openloop_results(local_results)
        self._last_openloop_eval_step = int(self.completed_steps)
        if not self.accelerator.is_main_process:
            return {}

        by_dataset: dict[str, dict[str, Any]] = {}
        for loader in self.openloop_eval_loaders:
            matching = [
                row
                for rank_rows in gathered
                for row in rank_rows
                if row["dataset"] == loader.dataset_name and row["split"] == loader.split
            ]
            accumulator = OpenLoopMetricAccumulator.merge([row["payload"] for row in matching])
            duration = max((float(row["duration_seconds"]) for row in matching), default=0.0)
            split_metrics = accumulator.finalize(duration_seconds=duration)
            if split_metrics["valid_frames"] != loader.target_count:
                raise ValueError(
                    f"open-loop {loader.dataset_name}/{loader.split} evaluated "
                    f"{split_metrics['valid_frames']} frames, expected {loader.target_count}"
                )
            by_dataset.setdefault(loader.dataset_name, {})[loader.split] = split_metrics

        normalized_mse = []
        for dataset_name, split_metrics in by_dataset.items():
            combined = self._combine_openloop_split_metrics(
                gathered,
                dataset_name=dataset_name,
            )
            split_metrics["combined"] = combined
            normalized_mse.append(float(combined["normalized_action_mse"]))
        report = {
            "datasets": by_dataset,
            "macro_normalized_action_mse": float(np.mean(normalized_mse)),
        }
        write_openloop_metrics(
            self.config.output_dir,
            step=self.completed_steps,
            metrics=report,
        )
        if "wandb" in self._configured_trackers():
            wandb.log(
                flatten_openloop_metrics(report),
                step=self.completed_steps,
            )
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(
                "openloop/macro_normalized_action_mse",
                report["macro_normalized_action_mse"],
                self.completed_steps,
            )
            for dataset_name, split_metrics in by_dataset.items():
                for split_name, metrics in split_metrics.items():
                    self.tensorboard_writer.add_scalar(
                        f"openloop/{dataset_name}/{split_name}/normalized_action_mse",
                        metrics["normalized_action_mse"],
                        self.completed_steps,
                    )
            self.tensorboard_writer.flush()
        return {
            "openloop_macro_normalized_mse": report["macro_normalized_action_mse"]
        }

    def _gather_openloop_results(self, local_results: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not dist.is_initialized():
            return [local_results]
        gathered: list[list[dict[str, Any]] | None] | None = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        dist.gather_object(local_results, gathered, dst=0)
        return [rows for rows in gathered if rows is not None] if gathered is not None else []

    @staticmethod
    def _combine_openloop_split_metrics(
        gathered: list[list[dict[str, Any]]],
        *,
        dataset_name: str,
    ) -> dict[str, Any]:
        matching = [
            row
            for rank_rows in gathered
            for row in rank_rows
            if row["dataset"] == dataset_name
        ]
        accumulator = OpenLoopMetricAccumulator.merge([row["payload"] for row in matching])
        duration = sum(
            max(
                (
                    float(row["duration_seconds"])
                    for row in matching
                    if row["split"] == split
                ),
                default=0.0,
            )
            for split in ("vln_val_seen", "vln_val_unseen")
        )
        return accumulator.finalize(duration_seconds=duration)

    @contextmanager
    def _model_evaluation_mode(self):
        """Temporarily enter eval mode while preserving mixed submodule modes."""
        training_modes = [(module, bool(module.training)) for module in self.model.modules()]
        self.model.eval()
        try:
            yield
        finally:
            for module, was_training in training_modes:
                module.training = was_training

    @contextmanager
    def _fixed_evaluation_rng(self):
        """Use a deterministic eval RNG stream and restore all training RNG states."""
        seed = int(getattr(self.config, "seed", 0))
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        cpu_state = torch.get_rng_state()
        cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        cuda_state = torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(seed)
            torch.set_rng_state(cpu_generator.get_state())
            if cuda_device is not None:
                cuda_generator = torch.Generator(device=torch.device("cuda", cuda_device))
                cuda_generator.manual_seed(seed)
                torch.cuda.set_rng_state(cuda_generator.get_state(), cuda_device)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(cpu_state)
            if cuda_device is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    training_step=self.completed_steps,
                    total_training_steps=self.config.trainer.max_train_steps,
                )
                total_loss = output_dict["action_loss"]
                action_loss = output_dict.get("action_dit_loss", total_loss)

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        log_dict = {
            "action_dit_loss": action_loss.item(),
        }
        if "stop_loss" in output_dict:
            log_dict["stop_loss"] = output_dict["stop_loss"].item()
        for metric_name in (
            "stop_logit_mean",
            "stop_prob_mean",
        ):
            if metric_name in output_dict:
                log_dict[metric_name] = output_dict[metric_name].item()
        return log_dict

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        self._finish_trackers()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    accelerator = build_accelerator(cfg)
    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if bool(cfg.get("is_debug", False)) and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
