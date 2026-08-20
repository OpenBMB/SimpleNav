import hashlib

import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_action_chunk(control_poses: np.ndarray, anchor_index: int, horizon: int = 8) -> np.ndarray:
    poses = np.asarray(control_poses, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 4:
        raise ValueError("control_poses must have shape [N, 4]")
    if not 0 <= anchor_index < len(poses):
        raise IndexError("anchor_index outside trajectory")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    future_indices = np.minimum(anchor_index + np.arange(1, horizon + 1), len(poses) - 1)
    anchor = poses[anchor_index]
    delta_world = poses[future_indices, :3] - anchor[:3]
    cosine = np.cos(anchor[3])
    sine = np.sin(anchor[3])
    dx = cosine * delta_world[:, 0] + sine * delta_world[:, 1]
    dy = -sine * delta_world[:, 0] + cosine * delta_world[:, 1]
    dz = delta_world[:, 2]
    dyaw = wrap_angle(poses[future_indices, 3] - anchor[3])
    return np.column_stack((dx, dy, dz, dyaw)).astype(np.float32)


def observation_indices(control_count: int, render_stride: int = 5) -> np.ndarray:
    if control_count <= 0:
        raise ValueError("control_count must be positive")
    if render_stride <= 0:
        raise ValueError("render_stride must be positive")
    indices = np.arange(0, control_count, render_stride, dtype=np.int64)
    terminal_index = control_count - 1
    if indices[-1] != terminal_index:
        indices = np.r_[indices, terminal_index]
    return indices


def random_observation_indices(
    control_count: int,
    stride_choices=(5, 6, 7, 8),
    *,
    seed: int = 0,
) -> np.ndarray:
    """Select observations using reproducible per-interval random gaps.

    The first and real terminal waypoint are always selected. Each complete gap
    is drawn with replacement from ``stride_choices``; only the final gap may be
    shorter when the terminal is not aligned with the sampled gap.
    """
    if control_count <= 0:
        raise ValueError("control_count must be positive")
    choices = tuple(int(value) for value in stride_choices)
    if not choices:
        raise ValueError("stride_choices must not be empty")
    if any(value <= 0 for value in choices):
        raise ValueError("stride_choices must be positive")

    terminal_index = control_count - 1
    indices = [0]
    draw_index = 0
    while indices[-1] < terminal_index:
        payload = f"{int(seed)}:{draw_index}".encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        gap = choices[bucket % len(choices)]
        next_index = indices[-1] + gap
        if next_index >= terminal_index:
            if indices[-1] != terminal_index:
                indices.append(terminal_index)
            break
        indices.append(next_index)
        draw_index += 1
    return np.asarray(indices, dtype=np.int64)


def build_observation_actions(control_poses: np.ndarray, render_stride: int = 5, horizon: int = 8):
    indices = observation_indices(len(control_poses), render_stride)
    actions = np.stack([build_action_chunk(control_poses, int(i), horizon) for i in indices])
    return indices, actions
