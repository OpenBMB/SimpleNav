from __future__ import annotations

from math import cos, pi, sin
from typing import Any

import numpy as np

from NavVLAeval.common.types import Pose4D


class BodyFrameContinuousActionAdapter:
    """Map NavVLA body-frame actions to Habitat MOVE_BY_POSE_DELTA actions.

    NavVLA action chunk rows are anchor-relative body-frame waypoints:
    [x_forward, y_right, z_down, yaw_right_positive].
    ``MoveByPoseDelta031`` applies ``[-dx, 0, -dy]`` in Habitat's body frame.
    Therefore the matching mapping is ``dx=-y_right``, ``dy=x_forward``, and
    ``dyaw=-yaw_right_positive``.  Habitat's positive rotation around its UP
    axis turns the front sensor left, while NavVLA uses right turn as positive.
    """

    def __init__(
        self,
        *,
        step_scale: float = 1.0,
        stop_probability_threshold: float = 0.5,
        min_delta: float = 1e-3,
        enable_lateral: bool = True,
        lateral_sign: float = -1.0,
        yaw_sign: float = -1.0,
        action_index: int = 0,
    ) -> None:
        self.step_scale = float(step_scale)
        self.stop_probability_threshold = float(stop_probability_threshold)
        self.min_delta = float(min_delta)
        self.enable_lateral = bool(enable_lateral)
        self.lateral_sign = float(lateral_sign)
        self.yaw_sign = float(yaw_sign)
        self.action_index = int(action_index)

    @classmethod
    def from_config(cls, cfg: Any) -> "BodyFrameContinuousActionAdapter":
        return cls(**_adapter_kwargs(cfg))

    def to_server_action(
        self,
        raw_actions: np.ndarray,
        stop_prob: float | None = None,
        action_index: int | None = None,
        anchor_pose: Pose4D | None = None,
        current_pose: Pose4D | None = None,
    ) -> dict[str, Any]:
        action_chunk = np.asarray(raw_actions, dtype=np.float32).reshape(-1, 4)
        sequential_chunk_step = action_index is not None
        selected_action_index = self.action_index if action_index is None else int(action_index)
        selected_action_index = min(max(selected_action_index, 0), action_chunk.shape[0] - 1)
        raw_action = action_chunk[selected_action_index]
        if not sequential_chunk_step:
            delta_action = np.asarray(raw_action, dtype=np.float32)
        elif anchor_pose is not None and current_pose is not None:
            delta_action = _anchor_relative_waypoint_delta_from_pose(action_chunk, selected_action_index, anchor_pose, current_pose)
        else:
            delta_action = _anchor_relative_waypoint_delta(action_chunk, selected_action_index)
        dx_body, dy_body, _dz, dyaw_in = [float(value) for value in delta_action]
        if stop_prob is not None and float(stop_prob) >= self.stop_probability_threshold:
            return self._stop_payload(raw_action, "stop_probability", stop_prob, selected_action_index, delta_action=delta_action)
        dy = _clean_float(dx_body * self.step_scale)
        dx = _clean_float(self.lateral_sign * dy_body * self.step_scale) if self.enable_lateral else 0.0
        dyaw = _clean_float(self.yaw_sign * dyaw_in)
        if abs(dx) + abs(dy) + abs(dyaw) < self.min_delta:
            return self._stop_payload(raw_action, "below_min_delta", stop_prob, selected_action_index, delta_action=delta_action)
        return {
            "server_payload": {"action": "MOVE_BY_POSE_DELTA", "action_args": {"dx": float(dx), "dy": float(dy), "dyaw": float(dyaw)}},
            "action_label": "MOVE_BY_POSE_DELTA",
            "log": {
                "raw_action": raw_action.tolist(),
                "delta_action": _action_log_values(delta_action),
                "action_index": int(selected_action_index),
                "action": "MOVE_BY_POSE_DELTA",
                "dx": float(dx),
                "dy": float(dy),
                "dyaw": float(dyaw),
                "decision_reason": "continuous_anchor_relative_delta" if sequential_chunk_step else "continuous",
                "stop_prob": None if stop_prob is None else float(stop_prob),
            },
        }

    def _stop_payload(
        self,
        raw_action: np.ndarray,
        reason: str,
        stop_prob: float | None,
        action_index: int | None = None,
        *,
        delta_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return {
            "server_payload": {"action": "STOP"},
            "action_label": "STOP",
            "log": {
                "raw_action": raw_action.tolist(),
                "delta_action": None if delta_action is None else _action_log_values(delta_action),
                "action_index": None if action_index is None else int(action_index),
                "action": "STOP",
                "dx": 0.0,
                "dy": 0.0,
                "dyaw": 0.0,
                "decision_reason": reason,
                "stop_prob": None if stop_prob is None else float(stop_prob),
            },
        }


def _clean_float(value: float) -> float:
    return float(round(float(value), 7))


def _action_log_values(action: np.ndarray) -> list[float]:
    return [_clean_float(float(value)) for value in np.asarray(action, dtype=np.float32).reshape(4)]


def _anchor_relative_waypoint_delta(action_chunk: np.ndarray, selected_action_index: int) -> np.ndarray:
    waypoint = np.asarray(action_chunk[selected_action_index], dtype=np.float32)
    if selected_action_index <= 0:
        return np.asarray(waypoint, dtype=np.float32)

    previous = np.asarray(action_chunk[selected_action_index - 1], dtype=np.float32)
    dx_anchor = float(waypoint[0] - previous[0])
    dy_anchor = float(waypoint[1] - previous[1])
    previous_yaw = float(previous[3])
    cos_yaw = cos(previous_yaw)
    sin_yaw = sin(previous_yaw)
    dx_previous_body = cos_yaw * dx_anchor + sin_yaw * dy_anchor
    dy_previous_body = -sin_yaw * dx_anchor + cos_yaw * dy_anchor
    dz_previous_body = float(waypoint[2] - previous[2])
    dyaw_previous_body = _wrap_to_pi(float(waypoint[3] - previous[3]))
    return np.asarray(
        [dx_previous_body, dy_previous_body, dz_previous_body, dyaw_previous_body],
        dtype=np.float32,
    )


def _anchor_relative_waypoint_delta_from_pose(
    action_chunk: np.ndarray,
    selected_action_index: int,
    anchor_pose: Pose4D,
    current_pose: Pose4D,
) -> np.ndarray:
    waypoint = np.asarray(action_chunk[selected_action_index], dtype=np.float32)
    current_anchor = _pose_relative_to_anchor(anchor_pose, current_pose)
    dx_anchor = float(waypoint[0] - current_anchor[0])
    dy_anchor = float(waypoint[1] - current_anchor[1])
    current_yaw = float(current_anchor[3])
    cos_yaw = cos(current_yaw)
    sin_yaw = sin(current_yaw)
    dx_current_body = cos_yaw * dx_anchor + sin_yaw * dy_anchor
    dy_current_body = -sin_yaw * dx_anchor + cos_yaw * dy_anchor
    dz_current_body = float(waypoint[2] - current_anchor[2])
    dyaw_current_body = _wrap_to_pi(float(waypoint[3] - current_anchor[3]))
    return np.asarray(
        [dx_current_body, dy_current_body, dz_current_body, dyaw_current_body],
        dtype=np.float32,
    )


def _pose_relative_to_anchor(anchor_pose: Pose4D, current_pose: Pose4D) -> np.ndarray:
    dx_world = float(current_pose.x - anchor_pose.x)
    dy_world = float(current_pose.y - anchor_pose.y)
    yaw = float(anchor_pose.yaw)
    cos_yaw = cos(yaw)
    sin_yaw = sin(yaw)
    # VLN-CE states use yaw=0 facing +world-y, with positive yaw turning right.
    forward = -sin_yaw * dx_world + cos_yaw * dy_world
    right = -cos_yaw * dx_world - sin_yaw * dy_world
    dz = float(current_pose.z - anchor_pose.z)
    dyaw = _wrap_to_pi(float(current_pose.yaw - anchor_pose.yaw))
    return np.asarray([forward, right, dz, dyaw], dtype=np.float32)


def _wrap_to_pi(value: float) -> float:
    return float((float(value) + pi) % (2.0 * pi) - pi)


def _adapter_kwargs(cfg: Any) -> dict[str, Any]:
    return {
        "step_scale": float(_cfg_get(cfg, "step_scale", 1.0)),
        "stop_probability_threshold": float(_cfg_get(cfg, "stop_probability_threshold", 0.5)),
        "min_delta": float(_cfg_get(cfg, "min_delta", 1e-3)),
        "enable_lateral": bool(_cfg_get(cfg, "enable_lateral", True)),
        "lateral_sign": float(_cfg_get(cfg, "lateral_sign", -1.0)),
        "yaw_sign": float(_cfg_get(cfg, "yaw_sign", -1.0)),
        "action_index": int(_cfg_get(cfg, "action_index", 0)),
    }


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)
