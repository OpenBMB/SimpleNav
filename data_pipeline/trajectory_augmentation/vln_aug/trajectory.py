from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TrajectoryConfig:
    control_frequency_hz: float = 1.0
    speed_mean_mps: float = 1.0
    speed_std_mps: float = 0.05
    speed_min_mps: float = 0.9
    speed_max_mps: float = 1.1
    smoothing_strength: float = 0.1
    dense_samples_per_meter: int = 50
    max_deviation_m: float | None = None
    turn_slowdown_enabled: bool = True
    turn_speed_min_factor: float = 0.55
    turn_curvature_start_rad_per_m: float = 0.08
    turn_curvature_full_rad_per_m: float = 0.45
    turn_smoothing_multiplier: float = 2.0


@dataclass(frozen=True)
class RetimedTrajectory:
    source_poses: np.ndarray
    smoothed_dense_poses: np.ndarray
    control_poses: np.ndarray
    control_times: np.ndarray
    movement_steps: int
    total_steps: int
    path_length_m: float
    movement_speed_mps: float
    max_deviation_m: float
    cruise_speed_mps: float = 0.0
    minimum_local_speed_mps: float = 0.0
    control_turn_intensity: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )


def stable_trajectory_seed(dataset_key: str, episode_index: int) -> int:
    payload = f"{dataset_key}:{int(episode_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _deduplicate_position(poses: np.ndarray) -> np.ndarray:
    keep = np.r_[True, np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) > 1e-8]
    return poses[keep]


def _validate_config(config: TrajectoryConfig) -> None:
    if config.control_frequency_hz != 1.0:
        raise ValueError("control_frequency_hz must be 1.0")
    if not 0.0 < config.turn_speed_min_factor <= 1.0:
        raise ValueError("turn_speed_min_factor must be in (0, 1]")
    if config.turn_curvature_start_rad_per_m < 0.0:
        raise ValueError("turn_curvature_start_rad_per_m must be non-negative")
    if (
        config.turn_curvature_full_rad_per_m
        <= config.turn_curvature_start_rad_per_m
    ):
        raise ValueError(
            "turn_curvature_full_rad_per_m must exceed the start threshold"
        )
    if config.turn_smoothing_multiplier < 1.0:
        raise ValueError("turn_smoothing_multiplier must be at least 1.0")


def _turn_intensity_from_xyz(
    xyz: np.ndarray,
    arc: np.ndarray,
    config: TrajectoryConfig,
) -> np.ndarray:
    if len(xyz) < 3 or not config.turn_slowdown_enabled:
        return np.zeros(len(xyz), dtype=float)
    arc_steps = np.diff(arc)
    if np.any(arc_steps < -1e-10):
        raise ValueError("trajectory arc length must be non-decreasing")
    if np.any(arc_steps <= 1e-10):
        keep = np.r_[True, arc_steps > 1e-10]
        unique_arc = arc[keep]
        unique_intensity = _turn_intensity_from_xyz(xyz[keep], unique_arc, config)
        return np.interp(arc, unique_arc, unique_intensity)
    tangent = np.gradient(xyz, arc, axis=0, edge_order=1)
    tangent_norm = np.linalg.norm(tangent, axis=1)
    unit_tangent = np.divide(
        tangent,
        tangent_norm[:, None],
        out=np.zeros_like(tangent),
        where=tangent_norm[:, None] > 1e-9,
    )
    tangent_change = np.gradient(unit_tangent, arc, axis=0, edge_order=1)
    curvature = np.linalg.norm(tangent_change, axis=1)
    spacing = float(arc[-1] / max(1, len(arc) - 1))
    sigma_samples = 0.35 / max(spacing, 1e-8)
    curvature = gaussian_filter1d(
        curvature, sigma=max(0.5, sigma_samples), mode="nearest", truncate=3.0
    )
    start = config.turn_curvature_start_rad_per_m
    full = config.turn_curvature_full_rad_per_m
    return np.clip((curvature - start) / (full - start), 0.0, 1.0)


def _smooth_spatial_path(poses: np.ndarray, config: TrajectoryConfig):
    points = _deduplicate_position(poses)
    if len(points) < 2:
        raise ValueError("trajectory has no positional extent")
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1))]
    if chord[-1] <= 1e-8:
        raise ValueError("trajectory has no positional extent")
    u = chord / chord[-1]
    dense_count = max(200, int(np.ceil(chord[-1] * config.dense_samples_per_meter)))
    dense_u = np.linspace(0.0, 1.0, dense_count)

    max_deviation = config.max_deviation_m
    if max_deviation is None:
        positive_steps = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
        max_deviation = min(0.3, 0.25 * float(np.median(positive_steps)))
    positive_steps = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
    median_source_step = float(np.median(positive_steps[positive_steps > 1e-8]))
    base_sigma_m = min(
        max_deviation * 1.5,
        max(0.0, config.smoothing_strength) * median_source_step,
    )
    attempts = [base_sigma_m, base_sigma_m * 0.5, base_sigma_m * 0.25, 0.0]
    xyz = None
    measured_deviation = float("inf")
    source_reference_count = max(dense_count, len(points) * 50)
    source_u = np.linspace(0.0, 1.0, source_reference_count)
    source_xyz = np.column_stack([np.interp(source_u, u, points[:, dim]) for dim in range(3)])
    source_tree = cKDTree(source_xyz)
    source_dense = np.column_stack(
        [np.interp(dense_u, u, points[:, dim]) for dim in range(3)]
    )
    dense_spacing_m = chord[-1] / max(1, dense_count - 1)
    source_arc = np.linspace(0.0, chord[-1], dense_count)
    turn_weight = _turn_intensity_from_xyz(source_dense, source_arc, config)
    for sigma_m in attempts:
        if sigma_m > 0.0:
            sigma_samples = sigma_m / max(dense_spacing_m, 1e-8)
            base_candidate = np.column_stack(
                [
                    gaussian_filter1d(
                        source_dense[:, dim],
                        sigma=sigma_samples,
                        mode="nearest",
                        truncate=4.0,
                    )
                    for dim in range(3)
                ]
            )
            turn_candidate = np.column_stack(
                [
                    gaussian_filter1d(
                        source_dense[:, dim],
                        sigma=sigma_samples * config.turn_smoothing_multiplier,
                        mode="nearest",
                        truncate=4.0,
                    )
                    for dim in range(3)
                ]
            )
            candidate = (
                base_candidate * (1.0 - turn_weight[:, None])
                + turn_candidate * turn_weight[:, None]
            )
        else:
            candidate = source_dense.copy()
        candidate[0] = poses[0, :3]
        candidate[-1] = poses[-1, :3]
        measured_deviation = float(np.max(source_tree.query(candidate, k=1)[0]))
        if measured_deviation <= max_deviation + 1e-9:
            xyz = candidate
            break
    if xyz is None:
        raise ValueError(
            f"smoothed trajectory exceeds source corridor: {measured_deviation:.6f} > {max_deviation:.6f} m"
        )
    xyz[0] = poses[0, :3]
    xyz[-1] = poses[-1, :3]

    unwrapped_yaw = np.unwrap(points[:, 3])
    yaw = CubicSpline(u, unwrapped_yaw, bc_type="natural")(dense_u)
    yaw[0] = unwrapped_yaw[0]
    yaw[-1] = unwrapped_yaw[-1]
    return dense_u, np.column_stack((xyz, yaw)), measured_deviation


def _choose_movement_steps(path_length: float, requested_speed: float, config: TrajectoryConfig) -> int:
    if path_length < config.speed_min_mps:
        raise ValueError("trajectory is shorter than minimum one-second travel at configured speed")
    min_steps = max(1, int(np.ceil(path_length / config.speed_max_mps)))
    max_steps = max(min_steps, int(np.floor(path_length / config.speed_min_mps)))
    candidates = np.arange(min_steps, max_steps + 1, dtype=int)
    if candidates.size == 0:
        raise ValueError("no integer-duration speed satisfies configured range")
    speeds = path_length / candidates
    return int(candidates[np.argmin(np.abs(speeds - requested_speed))])


def smooth_and_retime(poses: np.ndarray, config: TrajectoryConfig, seed: int) -> RetimedTrajectory:
    source = np.asarray(poses, dtype=float)
    if source.ndim != 2 or source.shape[1] != 4 or len(source) < 2:
        raise ValueError("poses must have shape [N, 4] with N >= 2")
    if not np.all(np.isfinite(source)):
        raise ValueError("poses contain non-finite values")
    _validate_config(config)

    _, dense, max_deviation = _smooth_spatial_path(source, config)
    segment = np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(segment)]
    path_length = float(arc[-1])
    if path_length <= 1e-8:
        raise ValueError("trajectory has no positional extent")

    rng = np.random.default_rng(seed)
    requested_speed = float(
        np.clip(rng.normal(config.speed_mean_mps, config.speed_std_mps), config.speed_min_mps, config.speed_max_mps)
    )
    uniform_steps = _choose_movement_steps(path_length, requested_speed, config)
    dense_turn_intensity = _turn_intensity_from_xyz(dense[:, :3], arc, config)
    if config.turn_slowdown_enabled:
        local_speed = requested_speed * (
            1.0
            - dense_turn_intensity * (1.0 - config.turn_speed_min_factor)
        )
        segment_speed = 0.5 * (local_speed[:-1] + local_speed[1:])
        raw_time = np.r_[0.0, np.cumsum(segment / np.maximum(segment_speed, 1e-8))]
        movement_steps = max(
            uniform_steps,
            int(np.ceil(float(raw_time[-1]) - 1e-9)),
        )
        scaled_time = raw_time * (movement_steps / float(raw_time[-1]))
        movement_distances = np.interp(
            np.arange(movement_steps + 1, dtype=float), scaled_time, arc
        )
    else:
        local_speed = np.full(len(dense), requested_speed, dtype=float)
        movement_steps = uniform_steps
        movement_distances = np.linspace(0.0, path_length, movement_steps + 1)
    control_xyz = np.column_stack([np.interp(movement_distances, arc, dense[:, dim]) for dim in range(3)])
    control_yaw = np.interp(movement_distances, arc, dense[:, 3])
    moving = np.column_stack((control_xyz, control_yaw))
    moving[0] = source[0]
    moving[-1, :3] = source[-1, :3]
    moving[-1, 3] = np.unwrap(source[:, 3])[-1]

    total_steps = movement_steps
    controls = moving
    step_midpoint_distance = 0.5 * (
        movement_distances[:-1] + movement_distances[1:]
    )
    control_turn_intensity = np.interp(
        step_midpoint_distance, arc, dense_turn_intensity
    )
    arc_step_distance = np.diff(movement_distances)

    return RetimedTrajectory(
        source_poses=source.copy(),
        smoothed_dense_poses=dense,
        control_poses=controls,
        control_times=np.arange(total_steps + 1, dtype=float),
        movement_steps=movement_steps,
        total_steps=total_steps,
        path_length_m=path_length,
        movement_speed_mps=path_length / movement_steps,
        max_deviation_m=max_deviation,
        cruise_speed_mps=(
            float(np.max(arc_step_distance)) if len(arc_step_distance) else 0.0
        ),
        minimum_local_speed_mps=(
            float(np.min(arc_step_distance)) if len(arc_step_distance) else 0.0
        ),
        control_turn_intensity=control_turn_intensity,
    )
