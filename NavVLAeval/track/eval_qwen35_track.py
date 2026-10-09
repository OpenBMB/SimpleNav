#!/usr/bin/env python3
"""Track simulator client using the shared model session protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from contextlib import closing

import numpy as np
from PIL import Image, ImageDraw

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from NavVLAeval.common.config import apply_driver_environment, load_driver_environment

TRACK_CONFIGS = {
    task: Path(__file__).parent / "habitat" / "config/benchmark/nav/track" / f"track_infer_{task}.yaml"
    for task in ("at", "dt", "stt")
}
STAT_KEYS = {
    "at": "evt-bench-at-teach-avoid",
    "dt": "evt-bench-dt-teach-avoid",
    "stt": "evt-bench-stt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TRACK_CONFIGS), required=True)
    parser.add_argument("--model-uri", default="ws://127.0.0.1:10093")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--driver-paths-file", type=Path, default=ROOT / "NavVLAeval/driver_paths.yaml")
    parser.add_argument("--data-root", type=Path, default=ROOT / "local/simulators/track")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-front-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the robot front-camera view as <episode_id>.mp4 beside each result JSON (enabled by default).",
    )
    parser.add_argument("--front-video-fps", type=float, default=10.0, help="FPS for --save-front-video output.")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Extra Habitat config overrides.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def light_setup():
    from habitat_sim.gfx import LightInfo, LightPositionModel

    return [
        LightInfo(vector=[10.0, -2.0, 0.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[-10.0, -2.0, 0.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[0.0, -2.0, 10.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
        LightInfo(vector=[0.0, -2.0, -10.0, 0.0], color=[1.0] * 3, model=LightPositionModel.Global),
    ]


def evaluate(args: argparse.Namespace) -> None:
    apply_driver_environment(load_driver_environment(args.driver_paths_file), os.environ)
    import importlib

    import habitat

    importlib.import_module("NavVLAeval.track.habitat")
    from habitat.datasets import make_dataset

    # Track dataset YAMLs intentionally use paths relative to OpenTrackVLA.
    os.chdir(args.data_root)
    if args.front_video_fps <= 0:
        raise ValueError(f"front-video-fps must be positive, got {args.front_video_fps}")
    if args.replan_steps <= 0:
        raise ValueError(f"replan-steps must be positive, got {args.replan_steps}")
    seed_everything(args.seed)
    save_path = (args.save_path or (ROOT / "local/eval_results/evt_bench" / args.task)).resolve()
    config_path = args.exp_config or TRACK_CONFIGS[args.task]
    print(f"[config] {config_path}", flush=True)
    config = habitat.get_config(str(config_path), args.opts)
    print("[dataset] loading", flush=True)
    dataset = make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
    print(f"[dataset] loaded {len(dataset.episodes)} episodes", flush=True)
    episode_ordinals = {str(episode.episode_id): i for i, episode in enumerate(dataset.episodes)}
    episodes = split_episodes(
        dataset,
        split_id=args.split_id,
        split_num=args.split_num,
        episode_ids=args.episode_ids,
        max_episodes=args.max_episodes,
    )
    print(
        f"[eval] task={args.task} episodes={len(episodes)} split={args.split_id}/{args.split_num} save={save_path}",
        flush=True,
    )
    if not episodes:
        return
    dataset.episodes = episodes
    summaries: list[dict[str, Any]] = []
    with (
        closing(WebsocketClientPolicy(uri=args.model_uri, timeout=args.request_timeout)) as policy,
        habitat.Env(config=config, dataset=dataset) as env,
    ):
        for _ in range(len(env.episodes)):
            env.reset()
            episode = env.current_episode
            output = episode_result_path(save_path, episode)
            if args.resume and done_result(output):
                summaries.append(json.loads(output.read_text(encoding="utf-8")))
                print(f"[skip] {output}", flush=True)
                continue
            env.sim.set_light_setup(light_setup())
            session_id = str(episode.episode_id)
            policy.reset(
                session_id=session_id,
                episode_id=session_id,
                statistics_key=STAT_KEYS[args.task],
                seed=args.seed + episode_ordinals[session_id],
            )
            pending_frames, trajectory = [], None
            control_tick = 0
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
                from NavVLAeval.common.simulators.habitat.vlnce031_runtime import camera_pose_from_transform

                body = camera_pose_from_transform(robot.base_transformation)
                sensor = env.sim._sensors["agent_1_articulated_agent_jaw_rgb"]._sensor_object
                camera = camera_pose_from_transform(sensor.node.absolute_transformation())
                pending_frames.append(
                    {
                        "frame_id": control_tick,
                        "timestamp_s": control_tick / 10.0,
                        "images": {"front": np.asarray(rgb[:, :, :3], dtype=np.uint8)},
                        "body_pose": np.asarray(body[:4], dtype=np.float32),
                        "body_rotation": body[3:],
                        "camera_poses": {"front": camera},
                    }
                )
                if trajectory is None or control_tick % args.replan_steps == 0:
                    prediction = policy.predict(frames=pending_frames, instruction=instruction)
                    pending_frames = []
                    if prediction.get("stop", False):
                        status = "ModelStop"
                        break
                    if prediction["action_spec"] != "anchor_relative_body_frame_xyz_yaw":
                        raise ValueError("Track requires body-frame waypoint actions")
                    trajectory = np.asarray(prediction["actions"], dtype=np.float32)
                # Preserve Track's existing second-waypoint velocity controller.
                waypoint = trajectory[min(1, len(trajectory) - 1)]
                action = [float(waypoint[0] * 10), float(-waypoint[1] * 10), float(-waypoint[3] * 10)]
                control_tick += 1
                if args.save_front_video:
                    front_frames.append(
                        np.ascontiguousarray(render_trajectory_on_frame(rgb, trajectory), dtype=np.uint8)
                    )
                env.step(
                    {
                        "action": (
                            "agent_0_humanoid_navigate_action",
                            "agent_1_base_velocity",
                            "agent_2_oracle_nav_randcoord_action_obstacle",
                            "agent_3_oracle_nav_randcoord_action_obstacle",
                            "agent_4_oracle_nav_randcoord_action_obstacle",
                            "agent_5_oracle_nav_randcoord_action_obstacle",
                        ),
                        "action_args": {"agent_1_base_vel": action},
                    }
                )
                metrics = env.get_metrics()
                distance = float(np.linalg.norm(robot.base_pos - human.base_pos))
                followed += int(metrics["human_following"] == 1.0)
                lost_steps = lost_steps + 1 if distance > 4.0 else 0
                records.append(
                    {
                        "step": len(records) + 1,
                        "base_velocity": action,
                        "distance": distance,
                        "trajectory": trajectory.tolist(),
                    }
                )
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
            policy.close_session(session_id)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            output.with_name(f"{episode.episode_id}_info.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            if args.save_front_video:
                video_path = output.with_suffix(".mp4")
                save_front_video(front_frames, video_path, args.front_video_fps)
                print(f"[video] {video_path}", flush=True)
            summaries.append(result)
            print(
                f"[episode] {episode.episode_id}: success={result['success']} steps={steps} status={status}", flush=True
            )
    summary = {
        "task": args.task,
        "model_uri": args.model_uri,
        "num_episodes": len(summaries),
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
