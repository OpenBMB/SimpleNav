from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from tool.navvla.statistics import body_frame_action_from_pose, build_repeated_state_statistics, normalize_values


TRAVELUAV_CAMERA_INDEX_BY_NAVVLA_NAME = {
    "front": 0,
    "left": 1,
    "right": 2,
    "rear": 3,
    "down": 4,
}
TRAVELUAV_NAVVLA_PLATFORM_TEXT = "Platform: UAV. Task: urban navigation. Action: local 3D waypoints (dx, dy, dz, dyaw)."
TRAVELUAV_STAGE_PREFIX = "Stage: "
TRAVELUAV_INSTRUCTION_SEPARATOR = "\n\nInstruction: "
TRAVELUAV_FLY_PREFIX_RE = re.compile(r"^Fly\s+.+?\s+and find the target\.\s*", re.IGNORECASE | re.DOTALL)
TRAVELUAV_TARGET_ANGLE_PREFIX_RE = re.compile(r"^.*?degrees from you\.\s*", re.IGNORECASE | re.DOTALL)
TRAVELUAV_FINAL_COMMAND_RE = re.compile(r"\s*Please control the drone.*$", re.IGNORECASE | re.DOTALL)


def build_navvla_instruction(*, episodes: list[dict[str, Any]]) -> str:
    instruction = str(episodes[-1].get("instruction") or "").strip()
    if not instruction:
        raise ValueError("latest TravelUAV episode observation does not contain instruction")
    return instruction


def build_traveluav_stage_instruction(*, episodes: list[dict[str, Any]], assist_notice: str | None = None) -> str:
    stage = resolve_traveluav_stage(episodes=episodes, assist_notice=assist_notice)
    object_description = _traveluav_object_description(episodes[-1])
    if not object_description:
        object_description = _strip_traveluav_instruction_to_description(build_navvla_instruction(episodes=episodes))
    instruction = f"Fly {stage} and find the target."
    if object_description:
        instruction = f"{instruction} {object_description}"
    return (
        f"{TRAVELUAV_STAGE_PREFIX}{stage}"
        f"{TRAVELUAV_INSTRUCTION_SEPARATOR}{instruction}"
    )


def _traveluav_object_description(episode: dict[str, Any]) -> str:
    for key in ("object_description", "object_desc"):
        for container in (episode, episode.get("benchmark_metadata"), episode.get("source_metadata")):
            if not isinstance(container, dict):
                continue
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                for item in value:
                    text = str(item).strip()
                    if text:
                        return text
    return ""


def _strip_traveluav_instruction_to_description(instruction: str) -> str:
    text = str(instruction).strip()
    if text.startswith(TRAVELUAV_STAGE_PREFIX) and TRAVELUAV_INSTRUCTION_SEPARATOR in text:
        text = text.split(TRAVELUAV_INSTRUCTION_SEPARATOR, 1)[1].strip()
    text = TRAVELUAV_FLY_PREFIX_RE.sub("", text, count=1).strip()
    text = TRAVELUAV_TARGET_ANGLE_PREFIX_RE.sub("", text, count=1).strip()
    text = TRAVELUAV_FINAL_COMMAND_RE.sub("", text).strip()
    return text


def resolve_traveluav_stage(*, episodes: list[dict[str, Any]], assist_notice: str | None = None) -> str:
    if assist_notice is not None and str(assist_notice).strip():
        return str(assist_notice).strip()
    return "cruise" if len(episodes) > 20 else "take off"


def build_navvla_history_state(
    *,
    episodes: list[dict[str, Any]],
    history_steps: int,
    action_stats: dict[str, Any] | None,
) -> np.ndarray:
    history_steps = int(history_steps)
    raw_chunks = np.zeros((history_steps, 4), dtype=np.float32)
    if history_steps <= 0:
        return raw_chunks.reshape(-1)

    padding_mask = np.ones((history_steps, 4), dtype=bool)
    pose_history = [extract_xyz_yaw_state(episode) for episode in episodes if "rgb" in episode]
    if len(pose_history) >= 2:
        chunks = np.stack(
            [
                body_frame_action_from_pose(pose_history[index - 1], pose_history[index])
                for index in range(1, len(pose_history))
            ],
            axis=0,
        )
        chunks = chunks[-history_steps:]
        raw_chunks[-chunks.shape[0] :] = chunks
        padding_mask[-chunks.shape[0] :] = False

    if action_stats is None:
        return raw_chunks.reshape(-1).astype(np.float32)
    state_stats = build_repeated_state_statistics(action_stats, history_steps)
    normalized = normalize_values(raw_chunks.reshape(-1), state_stats).astype(np.float32)
    normalized[padding_mask.reshape(-1)] = 0.0
    return normalized

def extract_xyz_yaw_state(episode: dict[str, Any]) -> np.ndarray:
    position = np.asarray(episode["sensors"]["state"]["position"], dtype=np.float32).reshape(3)
    rotation = np.asarray(episode["sensors"]["imu"]["rotation"], dtype=np.float32)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.asarray([position[0], position[1], position[2], yaw], dtype=np.float32)
