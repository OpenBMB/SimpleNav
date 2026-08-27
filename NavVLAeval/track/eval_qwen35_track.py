#!/usr/bin/env python3
"""Independent closed-loop Track evaluation for navvla_qwen35_cpm checkpoints.

The policy input intentionally follows the training config saved beside the
checkpoint: one front camera, 256x256 Qwen3.5 post-merge cache, time_yaw TVI,
1024-token dynamic BATS, and an 8-waypoint action chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OPENTRACK_ROOT = Path("/path/track-lerobot")
for path in (ROOT, OPENTRACK_ROOT, OPENTRACK_ROOT / "habitat-lab"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import habitat
import numpy as np
import torch
from habitat.datasets import make_dataset
from habitat_sim.gfx import LightInfo, LightPositionModel
from PIL import Image, ImageDraw

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.bats import select_bats_history
from tool.navvla.compute_bats_k import (
    QWEN35_CACHED_HISTORY_WRAPPER_TOKENS,
    QWEN35_CURRENT_IMAGE_WRAPPER_TOKENS,
)
from tool.navvla.statistics import unnormalize_values


DEFAULT_RUN_DIR = ROOT / "results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/navvla_qwen35_cpm_track_at_dt_stt_bats_front_bs7_ga6_step7374_20260813_210046"
DEFAULT_CKPT = DEFAULT_RUN_DIR / "final_model/pytorch_model.pt"
TRACK_CONFIGS = {
    "at": OPENTRACK_ROOT / "habitat-lab/habitat/config/benchmark/nav/track/track_infer_at.yaml",
    "dt": OPENTRACK_ROOT / "habitat-lab/habitat/config/benchmark/nav/track/track_infer_dt.yaml",
    "stt": OPENTRACK_ROOT / "habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml",
}
STAT_KEYS = {
    "at": "evt-bench-at-teach-avoid",
    "dt": "evt-bench-dt-teach-avoid",
    "stt": "evt-bench-stt",
}
TRAIN_DATA_ROOT = Path("/path/data/four-view-evt-bench-v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TRACK_CONFIGS), required=True)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--exp-config", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--split-id", type=int, default=0)
    parser.add_argument("--split-num", type=int, default=1)
    parser.add_argument("--episode-ids", default="", help="Comma-separated IDs within the selected split.")
    parser.add_argument("--max-episodes", type=int, default=0, help="0 evaluates every selected episode.")
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=1,
        help="Number of control steps to execute from an action chunk before replanning (default: 1).",
    )
    parser.add_argument(
        "--platform-text",
        default="",
        help="Override platform_text for controlled prompt ablations; empty uses the training dataset value.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-front-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the robot front-camera view as <episode_id>.mp4 beside each result JSON (enabled by default).",
    )
    parser.add_argument("--front-video-fps", type=float, default=10.0, help="FPS for --save-front-video output.")
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Qwen attention backend. Defaults to sdpa because the Track Conda environment has no flash_attn.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra Habitat config overrides.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_episodes(dataset: Any, *, split_id: int, split_num: int, episode_ids: str, max_episodes: int) -> list[Any]:
    if split_num <= 0 or not 0 <= split_id < split_num:
        raise ValueError(f"split must satisfy 0 <= split-id < split-num, got {split_id}/{split_num}")
    episodes = list(dataset.get_splits(split_num)[split_id].episodes)
    requested = {value.strip() for value in episode_ids.split(",") if value.strip()}
    if requested:
        episodes = [episode for episode in episodes if str(episode.episode_id) in requested]
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    return episodes


def scene_key(episode: Any) -> str:
    return Path(str(episode.scene_id)).stem.split(".")[0]


def episode_result_path(save_path: Path, episode: Any) -> Path:
    return save_path / scene_key(episode) / f"{episode.episode_id}.json"


def done_result(path: Path) -> bool:
    try:
        return "success" in json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False


def save_front_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    imageio.mimsave(path, frames, fps=fps)


def render_trajectory_on_frame(rgb: np.ndarray, trajectory: np.ndarray | None) -> np.ndarray:
    """Overlay the predicted body-frame waypoint chunk on the front-camera frame."""
    try:
        if trajectory is None or not isinstance(trajectory, np.ndarray) or trajectory.size == 0:
            return rgb
        image = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        base_x = width // 2
        base_y = int(height * 0.86)
        scale = 120.0
        points = [
            (base_x - int(float(waypoint[1]) * scale), base_y - int(float(waypoint[0]) * scale))
            for waypoint in trajectory[:64]
        ]
        for start, end in zip(points, points[1:]):
            draw.line([start, end], fill=(0, 0, 0), width=8)
        for start, end in zip(points, points[1:]):
            draw.line([start, end], fill=(0, 255, 180), width=4)
        if points:
            radius = 4
            x, y = points[0]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(0, 255, 0))
        return np.asarray(image)
    except Exception:
        return rgb


def load_training_platform_text(task: str) -> str:
    import pandas as pd

    path = TRAIN_DATA_ROOT / task.upper() / "meta/tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing training task metadata: {path}")
    values = pd.read_parquet(path, columns=["platform_text"])["platform_text"].dropna().astype(str)
    texts = sorted({value.strip() for value in values if value.strip()})
    if len(texts) != 1:
        raise ValueError(f"expected one platform_text in {path}, found {len(texts)} distinct values")
    return texts[0]


class Qwen35TrackPolicy:
    """Online-cache policy adapter. It does not depend on any old Track evaluator."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        stats_key: str,
        platform_text: str,
        replan_steps: int = 1,
        attn_implementation: str = "",
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        overrides = {"framework": {"qwenvl": {"attn_implementation": attn_implementation}}} if attn_implementation else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[load] {self.checkpoint} on {self.device}", flush=True)
        self.model = baseframework.from_pretrained(str(self.checkpoint), config_overrides=overrides).to(self.device).eval()
        if str(self.model.config.framework.name) != "navvla_qwen35_cpm":
            raise ValueError(f"expected navvla_qwen35_cpm checkpoint, got {self.model.config.framework.name!r}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        config = self.model.config.framework
        nav_cfg = config.navvla
        action_cfg = config.action_model
        self.history_tokens = int(nav_cfg.history_visual_tokens)
        self.current_tokens = int(nav_cfg.current_visual_tokens)
        self.token_budget = 1024
        self.horizon = int(action_cfg.action_horizon)
        self.action_dim = int(action_cfg.action_dim)
        if (self.history_tokens, self.current_tokens, self.horizon, self.action_dim) != (4, 64, 8, 4):
            raise ValueError("this evaluator is for the saved Track contract: history/current/action = 4/64/8x4")
        if list(nav_cfg.visual_cache_input_resize) != [256, 256] or str(nav_cfg.tvi_mode) != "time_yaw":
            raise ValueError("checkpoint visual preprocessing is not the expected 256x256 time_yaw Track contract")
        if int(nav_cfg.long_memory_visual_tokens) != 0:
            raise ValueError("this evaluator intentionally supports the checkpoint's no-long-memory training setup only")

        stats_path = self.checkpoint.parents[1] / "dataset_statistics.json"
        statistics = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_key not in statistics:
            raise KeyError(f"statistics key {stats_key!r} absent from {stats_path}; keys={sorted(statistics)}")
        self.action_stats = statistics[stats_key]["action"]
        self.visual_profile = str(nav_cfg.visual_token_profile)
        self.encoder_ckpt = str(nav_cfg.visual_cache_encoder_ckpt)
        self.platform_text = str(platform_text)
        if replan_steps <= 0:
            raise ValueError(f"replan-steps must be positive, got {replan_steps}")
        self.replan_steps = int(replan_steps)
        self.reset()

    def reset(self) -> None:
        self.step_index = 0
        self.episode_id = ""
        self.history: list[dict[str, Any]] = []
        self._cached_trajectory: np.ndarray | None = None

    def start_episode(self, episode_id: str) -> None:
        self.reset()
        self.episode_id = str(episode_id)

    def _selected_history(self) -> list[dict[str, Any]]:
        return select_bats_history(
            candidates=[(record["frame_index"], record) for record in self.history],
            anchor_frame_index=self.step_index,
            episode_id=self.episode_id,
            dataset_name="opentrackvla",
            seed=42,
            epsilon=0.1,
            k=4.0,
            use_dynamic_bats_k=True,
            token_budget=self.token_budget,
            budget_num_cameras=1,
            current_visual_tokens=self.current_tokens,
            history_visual_tokens=self.history_tokens,
            tvi_tokens=1,
            current_wrapper_tokens=QWEN35_CURRENT_IMAGE_WRAPPER_TOKENS,
            history_wrapper_tokens=QWEN35_CACHED_HISTORY_WRAPPER_TOKENS,
            sampling_mode="priority_capped",
        ).selected

    def _sample(self, image: Image.Image, instruction: str) -> dict[str, Any]:
        selected = self._selected_history()
        if selected:
            cached_tokens = np.stack([record["tokens"] for record in selected], axis=0)
            cached_grid = np.stack([record["grid_thw"] for record in selected], axis=0)
        else:
            cached_tokens = np.zeros((0, self.history_tokens, self.model.hidden_size), dtype=np.uint16)
            cached_grid = np.zeros((0, 3), dtype=np.int64)
        timestamp = float(self.step_index) / 10.0
        return {
            "images": {"front": image},
            "current_tvi": np.asarray([[timestamp, 0.0]], dtype=np.float32),
            "history_tvi": np.asarray([[record["timestamp"], 0.0] for record in selected], dtype=np.float32).reshape(-1, 2),
            "history_mask": np.ones(len(selected), dtype=bool),
            "history_cached_embeds": cached_tokens,
            "history_cached_mask": np.ones(len(selected), dtype=bool),
            "history_cached_grid_thw": cached_grid,
            "history_cached_cache_stage": "vit_postmerge_pool4",
            "history_cached_storage_encoding": "bfloat16_bits_uint16",
            "history_cached_encoder_ckpt": self.encoder_ckpt,
            "lang": instruction or "follow the person",
            "platform_text": self.platform_text,
            "action": np.zeros((self.horizon, self.action_dim), dtype=np.float32),
            "action_padding_mask": np.zeros(self.horizon, dtype=bool),
            "metadata": {
                "required_cameras": ["front"],
                "history_steps": [{"timestamp": record["timestamp"]} for record in selected],
                "history_blocks": [
                    {"step_index": index, "camera_name": "front", "frame_index": record["frame_index"]}
                    for index, record in enumerate(selected)
                ],
                "long_memory_steps": [],
                "long_memory_blocks": [],
                "visual_token_profile": self.visual_profile,
                "timestamp": timestamp,
                "action_extra_dim_mode": "none",
            },
        }

    @torch.inference_mode()
    def act(self, rgb: np.ndarray, instruction: str) -> tuple[list[float], np.ndarray]:
        image = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB").resize((256, 256), Image.Resampling.BICUBIC)
        if self._cached_trajectory is None or self.step_index % self.replan_steps == 0:
            sample = self._sample(image, instruction)
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.no_grad()
            with autocast:
                normalized = np.asarray(self.model.predict_action([sample])["normalized_actions"], dtype=np.float32)
            self._cached_trajectory = unnormalize_values(normalized, self.action_stats)[0]
        trajectory = self._cached_trajectory
        # Track evaluation executes the second waypoint from each predicted chunk.
        waypoint = trajectory[1]
        action = [float(waypoint[0] * 10.0), float(-waypoint[1] * 10.0), float(-waypoint[3] * 10.0)]
        encoded = self.model.encode_history_images([image])[0]
        self.history.append(
            {
                "frame_index": self.step_index,
                "timestamp": float(self.step_index) / 10.0,
                "tokens": encoded["tokens"],
                "grid_thw": encoded["grid_thw"],
            }
        )
        self.step_index += 1
        return action, trajectory


def light_setup() -> list[LightInfo]:
    return [
        LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
    ]


def evaluate(args: argparse.Namespace) -> None:
    # Registers OpenTrackVLA's custom sensor config nodes before Hydra composes
    # track_infer_{at,dt,stt}.yaml.
    import evt_bench  # noqa: F401

    # Track dataset YAMLs intentionally use paths relative to OpenTrackVLA.
    os.chdir(OPENTRACK_ROOT)
    if args.front_video_fps <= 0:
        raise ValueError(f"front-video-fps must be positive, got {args.front_video_fps}")
    if args.replan_steps <= 0:
        raise ValueError(f"replan-steps must be positive, got {args.replan_steps}")
    seed_everything(args.seed)
    ckpt = args.ckpt.resolve()
    save_path = (args.save_path or (ckpt.parents[1] / "Eval" / f"track_qwen35_{args.task}")).resolve()
    config_path = args.exp_config or TRACK_CONFIGS[args.task]
    print(f"[config] {config_path}", flush=True)
    config = habitat.get_config(str(config_path), args.opts)
    print("[dataset] loading", flush=True)
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    print(f"[dataset] loaded {len(dataset.episodes)} episodes", flush=True)
    episodes = split_episodes(dataset, split_id=args.split_id, split_num=args.split_num, episode_ids=args.episode_ids, max_episodes=args.max_episodes)
    print(f"[eval] task={args.task} episodes={len(episodes)} split={args.split_id}/{args.split_num} save={save_path}", flush=True)
    if not episodes:
        return
    dataset.episodes = episodes
    platform_text = str(args.platform_text).strip() or load_training_platform_text(args.task)
    print(f"[platform_text] {platform_text}", flush=True)
    summaries: list[dict[str, Any]] = []
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        # On headless nodes, Habitat must claim the EGL device before Torch
        # creates its CUDA context; otherwise EGL cannot map CUDA device 0.
        policy = Qwen35TrackPolicy(
            ckpt,
            stats_key=STAT_KEYS[args.task],
            platform_text=platform_text,
            replan_steps=args.replan_steps,
            attn_implementation=args.attn_implementation,
        )
        for _ in range(len(env.episodes)):
            env.reset()
            episode = env.current_episode
            output = episode_result_path(save_path, episode)
            if args.resume and done_result(output):
                summaries.append(json.loads(output.read_text(encoding="utf-8")))
                print(f"[skip] {output}", flush=True)
                continue
            env.sim.set_light_setup(light_setup())
            policy.start_episode(str(episode.episode_id))
            instruction = str(getattr(episode, "info", {}).get("instruction", "") or "follow the person")
            robot = env.sim.agents_mgr[1].articulated_agent
            human = env.sim.agents_mgr[0].articulated_agent
            records: list[dict[str, Any]] = []
            front_frames: list[np.ndarray] = []
            followed = 0
            lost_steps = 0
            status = "Normal"
            while not env.episode_over:
                observations = env.sim.get_sensor_observations()
                rgb = observations["agent_1_articulated_agent_jaw_rgb"]
                action, trajectory = policy.act(rgb, instruction)
                if args.save_front_video:
                    front_frames.append(
                        np.ascontiguousarray(render_trajectory_on_frame(rgb, trajectory), dtype=np.uint8)
                    )
                env.step(
                    {
                        "action": (
                            "agent_0_humanoid_navigate_action", "agent_1_base_velocity",
                            "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle",
                            "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle",
                        ),
                        "action_args": {"agent_1_base_vel": action},
                    }
                )
                metrics = env.get_metrics()
                distance = float(np.linalg.norm(robot.base_pos - human.base_pos))
                followed += int(metrics["human_following"] == 1.0)
                lost_steps = lost_steps + 1 if distance > 4.0 else 0
                records.append({"step": len(records) + 1, "base_velocity": action, "distance": distance, "trajectory": trajectory.tolist()})
                if metrics["human_collision"] == 1.0:
                    status = "Collision"
                    break
                if lost_steps > 40:
                    status = "Lost"
                    break
            metrics = env.get_metrics()
            steps = len(records)
            collision = bool(metrics["human_collision"])
            success_signal = (
                bool(metrics["human_following_success"] or metrics["human_following"])
                if steps < 300
                else bool(metrics["human_following"])
            )
            result = {
                "finish": bool(env.episode_over),
                "status": status,
                "scene_id": scene_key(episode),
                "episode_id": str(episode.episode_id),
                "success": success_signal and not collision,
                "following_rate": followed / steps if steps else 0.0,
                "following_step": followed,
                "total_step": steps,
                "collision": collision,
                "instruction": instruction,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            output.with_name(f"{episode.episode_id}_info.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            if args.save_front_video:
                video_path = output.with_suffix(".mp4")
                save_front_video(front_frames, video_path, args.front_video_fps)
                print(f"[video] {video_path}", flush=True)
            summaries.append(result)
            print(f"[episode] {episode.episode_id}: success={result['success']} steps={steps} status={status}", flush=True)
    summary = {
        "task": args.task, "checkpoint": str(ckpt), "num_episodes": len(summaries),
        "success_rate": sum(bool(row["success"]) for row in summaries) / len(summaries),
        "mean_following_rate": sum(float(row["following_rate"]) for row in summaries) / len(summaries),
        "episodes": summaries,
    }
    save_path.mkdir(parents=True, exist_ok=True)
    summary_path = save_path / f"summary_split_{args.split_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path} SR={summary['success_rate']:.3f}", flush=True)


if __name__ == "__main__":
    evaluate(parse_args())
