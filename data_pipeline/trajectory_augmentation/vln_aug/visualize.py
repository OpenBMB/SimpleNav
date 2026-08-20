from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from vln_aug.actions import observation_indices
from vln_aug.trajectory import RetimedTrajectory


SOURCE_COLOR = "#4D4D4D"
SMOOTH_COLOR = "#0072B2"
CONTROL_COLOR = "#E69F00"
RENDER_COLOR = "#D55E00"
DEVIATION_COLOR = "#CC79A7"
ZERO_DEVIATION_TOLERANCE_M = 1e-9


@dataclass(frozen=True)
class DeviationProfile:
    progress_percent: np.ndarray
    distance_m: np.ndarray
    nearest_source_index: np.ndarray
    nearest_source_xyz: np.ndarray
    max_index: int


def compute_trajectory_metrics(
    result: RetimedTrajectory,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
) -> dict[str, float | int]:
    controls = result.control_poses
    resolved_image_indices = (
        observation_indices(len(controls), image_stride)
        if image_indices is None
        else np.asarray(image_indices, dtype=np.int64)
    )
    step_distance = np.linalg.norm(np.diff(controls[: result.movement_steps + 1, :3], axis=0), axis=1)
    return {
        "source_point_count": int(len(result.source_poses)),
        "control_point_count": int(len(controls)),
        "render_frame_count": int(
            len(resolved_image_indices)
        ),
        "path_length_m": float(result.path_length_m),
        "movement_speed_mps": float(result.movement_speed_mps),
        "max_deviation_m": float(result.max_deviation_m),
        "movement_duration_s": int(result.movement_steps),
        "total_duration_s": int(result.total_steps),
        "terminal_hover_s": int(result.total_steps - result.movement_steps),
        "control_frequency_hz": 1.0,
        "image_stride_waypoints": int(image_stride),
        "mean_control_step_m": float(np.mean(step_distance)) if len(step_distance) else 0.0,
        "min_control_step_m": float(np.min(step_distance)) if len(step_distance) else 0.0,
        "max_control_step_m": float(np.max(step_distance)) if len(step_distance) else 0.0,
        "cruise_speed_mps": float(result.cruise_speed_mps),
        "minimum_local_speed_mps": float(result.minimum_local_speed_mps),
        "turn_slow_step_fraction": float(
            np.mean(result.control_turn_intensity > 0.05)
        ) if len(result.control_turn_intensity) else 0.0,
    }


def compute_sampling_audit(
    result: RetimedTrajectory,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
    image_stride_choices: tuple[int, ...] | None = None,
) -> dict:
    controls = np.asarray(result.control_poses, dtype=float)
    indices = (
        observation_indices(len(controls), image_stride)
        if image_indices is None
        else np.asarray(image_indices, dtype=np.int64)
    )
    chord_steps = np.linalg.norm(np.diff(controls[:, :3], axis=0), axis=1)
    turn_intensity = np.asarray(result.control_turn_intensity, dtype=float)
    if len(turn_intensity) != len(chord_steps):
        turn_intensity = np.zeros(len(chord_steps), dtype=float)
    index_gaps = np.diff(indices)
    allowed_gaps = tuple(image_stride_choices or (image_stride,))
    regular_gaps = index_gaps[:-1] if len(index_gaps) else index_gaps
    return {
        "control_waypoint_count": int(len(controls)),
        "image_waypoint_count": int(len(indices)),
        "image_waypoint_indices": indices.astype(int).tolist(),
        "image_waypoint_index_gaps": index_gaps.astype(int).tolist(),
        "image_stride_waypoints": int(image_stride),
        "image_stride_choices_waypoints": [int(value) for value in allowed_gaps],
        "regular_image_gaps_match_stride": bool(
            len(regular_gaps) == 0 or all(int(gap) in allowed_gaps for gap in regular_gaps)
        ),
        "regular_image_gaps_are_five": bool(
            image_stride == 5
            and (len(regular_gaps) == 0 or np.all(regular_gaps == 5))
        ),
        "real_terminal_included": bool(indices[-1] == len(controls) - 1),
        "target_arc_step_m": float(
            result.cruise_speed_mps or result.movement_speed_mps
        ),
        "control_chord_step_min_m": float(np.min(chord_steps)) if len(chord_steps) else 0.0,
        "control_chord_step_mean_m": float(np.mean(chord_steps)) if len(chord_steps) else 0.0,
        "control_chord_step_max_m": float(np.max(chord_steps)) if len(chord_steps) else 0.0,
        "tightest_chord_start_index": int(np.argmin(chord_steps)) if len(chord_steps) else 0,
        "strongest_turn_step_index": int(np.argmax(turn_intensity))
        if len(turn_intensity)
        else 0,
        "control_turn_intensity": turn_intensity.tolist(),
    }


def display_indices(count: int, max_points: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if max_points < 2 and count > 1:
        raise ValueError("max_points must be at least two")
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, count - 1, max_points)).astype(np.int64))


def compute_deviation_profile(source: np.ndarray, smoothed: np.ndarray) -> DeviationProfile:
    source = np.asarray(source, dtype=float)
    dense = np.asarray(smoothed, dtype=float)
    if source.ndim != 2 or dense.ndim != 2 or source.shape[1] < 3 or dense.shape[1] < 3:
        raise ValueError("source and smoothed trajectories must contain xyz")
    if len(source) == 0 or len(dense) == 0:
        raise ValueError("trajectories must be non-empty")
    if len(source) == 1:
        nearest_xyz = np.repeat(source[:, :3], len(dense), axis=0)
        nearest = np.zeros(len(dense), dtype=np.int64)
    else:
        segment_start = source[:-1, :3]
        segment_vector = source[1:, :3] - segment_start
        segment_norm_sq = np.sum(segment_vector * segment_vector, axis=1)
        nearest_xyz = np.empty((len(dense), 3), dtype=float)
        nearest = np.empty(len(dense), dtype=np.int64)
        for point_index, point in enumerate(dense[:, :3]):
            relative = point - segment_start
            fraction = np.divide(
                np.sum(relative * segment_vector, axis=1),
                segment_norm_sq,
                out=np.zeros_like(segment_norm_sq),
                where=segment_norm_sq > 0,
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            projected = segment_start + fraction[:, None] * segment_vector
            distance_sq = np.sum((projected - point) ** 2, axis=1)
            segment_index = int(np.argmin(distance_sq))
            nearest_xyz[point_index] = projected[segment_index]
            nearest[point_index] = segment_index
    distance = np.linalg.norm(nearest_xyz - dense[:, :3], axis=1)
    distance[distance <= ZERO_DEVIATION_TOLERANCE_M] = 0.0
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense[:, :3], axis=0), axis=1))]
    progress = np.zeros(len(dense), dtype=float) if arc[-1] == 0 else arc / arc[-1] * 100.0
    return DeviationProfile(
        progress_percent=progress,
        distance_m=np.asarray(distance, dtype=float),
        nearest_source_index=np.asarray(nearest, dtype=np.int64),
        nearest_source_xyz=nearest_xyz,
        max_index=int(np.argmax(distance)),
    )


def _style_xy_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=11, fontweight="bold")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.22, linewidth=0.7)


def _plot_source(axis, source: np.ndarray, label: str = "source") -> None:
    line = axis.plot(
        source[:, 0], source[:, 1], color=SOURCE_COLOR, linewidth=2.2,
        linestyle=(0, (4, 2)), label=label, zorder=3,
    )[0]
    line.set_path_effects([path_effects.Stroke(linewidth=4.5, foreground="white"), path_effects.Normal()])
    point_indices = display_indices(len(source), 90)
    axis.scatter(
        source[point_indices, 0], source[point_indices, 1], s=22,
        facecolors="white", edgecolors=SOURCE_COLOR, linewidths=1.0, zorder=4,
    )
    axis.scatter(source[0, 0], source[0, 1], marker="s", s=72, color="#009E73", label="start", zorder=7)
    axis.scatter(source[-1, 0], source[-1, 1], marker="*", s=110, color="#CC3311", label="goal", zorder=7)


def _plot_enhanced(
    axis,
    dense: np.ndarray,
    control: np.ndarray,
    render: np.ndarray,
    sampling_label: str,
) -> None:
    line = axis.plot(dense[:, 0], dense[:, 1], color=SMOOTH_COLOR, linewidth=2.7, label="smoothed", zorder=3)[0]
    line.set_path_effects([path_effects.Stroke(linewidth=5.2, foreground="white"), path_effects.Normal()])
    control_idx = display_indices(len(control), 100)
    axis.scatter(
        control[control_idx, 0], control[control_idx, 1], color=CONTROL_COLOR,
        edgecolors="white", linewidths=0.65, s=34, label="1 Hz control (display-thinned)", zorder=5,
    )
    axis.scatter(
        render[:, 0], render[:, 1], facecolors="none",
        edgecolors=RENDER_COLOR, linewidths=1.8, s=72, marker="o",
        label=f"image points: {sampling_label} + terminal", zorder=6,
    )
    axis.scatter(dense[0, 0], dense[0, 1], marker="s", s=72, color="#009E73", zorder=7)
    axis.scatter(dense[-1, 0], dense[-1, 1], marker="*", s=110, color="#CC3311", zorder=7)


def _add_max_deviation_inset(axis, source, dense, profile) -> None:
    maximum = profile.max_index
    center = dense[maximum, :2]
    all_xy = np.vstack((source[:, :2], dense[:, :2]))
    span = np.ptp(all_xy, axis=0)
    radius = max(float(np.max(span)) * 0.08, float(profile.distance_m[maximum]) * 4.0, 0.2)
    inset = inset_axes(axis, width="38%", height="38%", loc="lower right", borderpad=1.0)
    inset.plot(source[:, 0], source[:, 1], color=SOURCE_COLOR, linewidth=1.5, linestyle="--")
    inset.plot(dense[:, 0], dense[:, 1], color=SMOOTH_COLOR, linewidth=2.0)
    inset.plot(
        [profile.nearest_source_xyz[maximum, 0], dense[maximum, 0]],
        [profile.nearest_source_xyz[maximum, 1], dense[maximum, 1]],
        color=DEVIATION_COLOR, linewidth=2.0,
    )
    inset.scatter(dense[maximum, 0], dense[maximum, 1], marker="X", s=55, color=RENDER_COLOR)
    inset.set_xlim(center[0] - radius, center[0] + radius)
    inset.set_ylim(center[1] - radius, center[1] + radius)
    inset.grid(True, alpha=0.2)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"max Δ={profile.distance_m[maximum]:.3f} m", fontsize=8)
    mark_inset(axis, inset, loc1=2, loc2=4, fc="none", ec="0.45", linewidth=0.8)


def build_trajectory_comparison_figure(
    result: RetimedTrajectory,
    title: str,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
    image_stride_choices: tuple[int, ...] | None = None,
):
    source = result.source_poses
    dense = result.smoothed_dense_poses
    control = result.control_poses
    resolved_image_indices = (
        observation_indices(len(control), image_stride)
        if image_indices is None
        else np.asarray(image_indices, dtype=np.int64)
    )
    render = control[resolved_image_indices]
    sampling_label = (
        f"random gaps in {list(image_stride_choices)} waypoints"
        if image_stride_choices is not None
        else f"every {image_stride} waypoints"
    )
    profile = compute_deviation_profile(source, dense)

    figure, grid = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = {
        "source": grid[0, 0],
        "enhanced": grid[0, 1],
        "overlay": grid[1, 0],
        "deviation": grid[1, 1],
    }

    _plot_source(axes["source"], source)
    _style_xy_axis(axes["source"], "Original trajectory")
    axes["source"].legend(fontsize=8, loc="best")

    _plot_enhanced(
        axes["enhanced"], dense, control, render, sampling_label=sampling_label
    )
    _style_xy_axis(
        axes["enhanced"],
        f"Enhanced sampling: turn-aware 1 Hz controls + {sampling_label}",
    )
    axes["enhanced"].legend(fontsize=8, loc="best")

    _plot_source(axes["overlay"], source)
    _plot_enhanced(
        axes["overlay"], dense, control, render, sampling_label=sampling_label
    )
    maximum = profile.max_index
    has_visible_deviation = profile.distance_m[maximum] > 0.0
    if has_visible_deviation:
        connector_idx = display_indices(len(dense), 24)
        for dense_index in connector_idx:
            axes["overlay"].plot(
                [profile.nearest_source_xyz[dense_index, 0], dense[dense_index, 0]],
                [profile.nearest_source_xyz[dense_index, 1], dense[dense_index, 1]],
                color=DEVIATION_COLOR, alpha=0.65, linewidth=0.8, zorder=2,
            )
        axes["overlay"].scatter(
            dense[maximum, 0], dense[maximum, 1], marker="X", s=90,
            color=RENDER_COLOR, edgecolors="white", linewidths=0.9,
            label="maximum deviation", zorder=8,
        )
    else:
        axes["overlay"].text(
            0.02, 0.04, "source and smoothed path coincide within numerical precision",
            transform=axes["overlay"].transAxes, fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
        )
    _style_xy_axis(
        axes["overlay"],
        "True-coordinate XY overlay and 3D source-to-smoothed deviation",
    )
    axes["overlay"].legend(fontsize=7.5, loc="best", ncol=2)
    if has_visible_deviation:
        _add_max_deviation_inset(axes["overlay"], source, dense, profile)

    axes["deviation"].plot(
        profile.progress_percent, profile.distance_m,
        color=DEVIATION_COLOR, linewidth=2.0,
    )
    axes["deviation"].fill_between(
        profile.progress_percent, 0.0, profile.distance_m,
        color=DEVIATION_COLOR, alpha=0.16,
    )
    if has_visible_deviation:
        axes["deviation"].scatter(
            profile.progress_percent[maximum], profile.distance_m[maximum],
            marker="X", s=85, color=RENDER_COLOR, zorder=5,
        )
        axes["deviation"].annotate(
            f"max {profile.distance_m[maximum]:.3f} m",
            xy=(profile.progress_percent[maximum], profile.distance_m[maximum]),
            xytext=(8, 10), textcoords="offset points", fontsize=9,
        )
    else:
        axes["deviation"].text(
            0.5, 0.5, "0 m: coincident within numerical precision",
            transform=axes["deviation"].transAxes, ha="center", va="center",
            fontsize=10, color=SOURCE_COLOR,
        )
        axes["deviation"].set_ylim(0.0, 1.0)
    axes["deviation"].set_title("Deviation from source versus smoothed-path progress", fontsize=11, fontweight="bold")
    axes["deviation"].set_xlabel("smoothed-path progress (%)")
    axes["deviation"].set_ylabel("nearest-source 3D distance (m)")
    axes["deviation"].grid(True, alpha=0.24)
    if has_visible_deviation:
        axes["deviation"].set_ylim(bottom=0.0)

    metrics = compute_trajectory_metrics(
        result, image_stride=image_stride, image_indices=resolved_image_indices
    )
    figure.suptitle(
        f"{title}\n"
        f"L={metrics['path_length_m']:.2f} m | v={metrics['movement_speed_mps']:.3f} m/s | "
        f"move={metrics['movement_duration_s']} s | total={metrics['total_duration_s']} s | "
        f"max deviation={profile.distance_m[maximum]:.3f} m",
        fontsize=14,
    )
    return figure, axes, profile


def plot_trajectory_comparison(
    result: RetimedTrajectory,
    output_path: Path,
    title: str,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
    image_stride_choices: tuple[int, ...] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, _, _ = build_trajectory_comparison_figure(
        result,
        title,
        image_stride=image_stride,
        image_indices=image_indices,
        image_stride_choices=image_stride_choices,
    )
    figure.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _projection_axes(points: np.ndarray) -> tuple[int, int, str, str]:
    spans = np.ptp(points[:, :3], axis=0)
    order = np.argsort(spans)[::-1]
    first, second = int(order[0]), int(order[1])
    names = ("x", "y", "z")
    return first, second, names[first], names[second]


def _plot_sampling_window(axis, controls, image_indices, center, title, radius=12):
    lo = max(0, int(center) - radius)
    hi = min(len(controls), int(center) + radius + 1)
    window_indices = np.arange(lo, hi, dtype=int)
    window = controls[window_indices]
    axis_a, axis_b, label_a, label_b = _projection_axes(window)
    axis.plot(
        window[:, axis_a], window[:, axis_b], color=SMOOTH_COLOR, linewidth=2.0, zorder=2
    )
    axis.scatter(
        window[:, axis_a], window[:, axis_b], s=30, color=CONTROL_COLOR,
        edgecolors="white", linewidths=0.5, zorder=3, label="every 1 Hz waypoint",
    )
    image_set = set(int(i) for i in image_indices)
    local_images = [int(i) for i in window_indices if int(i) in image_set]
    if local_images:
        local = controls[local_images]
        axis.scatter(
            local[:, axis_a], local[:, axis_b], s=88, facecolors="none",
            edgecolors=RENDER_COLOR, linewidths=1.8, zorder=4,
            label="image waypoint",
        )
    for index in window_indices:
        if int(index) in image_set:
            label = f"I@W{int(index)}"
        else:
            label = f"W{int(index)}"
        axis.annotate(
            label,
            (controls[index, axis_a], controls[index, axis_b]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7 if int(index) in image_set else 6,
            color=RENDER_COLOR if int(index) in image_set else SOURCE_COLOR,
        )
    axis.set_title(
        f"{title} ({label_a.upper()}{label_b.upper()} projection)",
        fontsize=10,
        fontweight="bold",
    )
    axis.set_xlabel(label_a)
    axis.set_ylabel(label_b)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=7, loc="best")


def build_sampling_audit_figure(
    result: RetimedTrajectory,
    title: str,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
    image_stride_choices: tuple[int, ...] | None = None,
):
    controls = np.asarray(result.control_poses, dtype=float)
    audit = compute_sampling_audit(
        result,
        image_stride=image_stride,
        image_indices=image_indices,
        image_stride_choices=image_stride_choices,
    )
    image_indices = np.asarray(audit["image_waypoint_indices"], dtype=int)
    figure, grid = plt.subplots(2, 2, figsize=(17, 12), constrained_layout=True)
    axes = {
        "index": grid[0, 0],
        "speed": grid[0, 1],
        "tightest": grid[1, 0],
        "terminal": grid[1, 1],
    }

    gap_numbers = np.arange(1, len(image_indices), dtype=int)
    gaps = np.diff(image_indices)
    axes["index"].step(
        gap_numbers,
        gaps,
        where="mid",
        color=RENDER_COLOR,
        linewidth=1.8,
        label="next image waypoint index - previous image waypoint index",
    )
    axes["index"].scatter(gap_numbers, gaps, color=RENDER_COLOR, s=18)
    allowed_gaps = tuple(image_stride_choices or (image_stride,))
    for allowed_gap in allowed_gaps:
        axes["index"].axhline(
            allowed_gap,
            color=SMOOTH_COLOR,
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
        )
    axes["index"].plot([], [], color=SMOOTH_COLOR, linestyle="--", label=f"allowed regular gaps = {list(allowed_gaps)}")
    if len(gaps) and gaps[-1] not in allowed_gaps:
        axes["index"].scatter(
            [gap_numbers[-1]], [gaps[-1]], marker="*", s=120, color="#CC3311",
            label="extra true-terminal image",
        )
    axes["index"].set_title(
        f"Image waypoint index gaps: regular gaps in {list(allowed_gaps)}",
        fontweight="bold",
    )
    axes["index"].set_xlabel("image transition number")
    axes["index"].set_ylabel("waypoint index gap")
    axes["index"].set_ylim(bottom=0)
    axes["index"].grid(True, alpha=0.2)
    axes["index"].legend(fontsize=8, loc="best")

    step_distance = np.linalg.norm(np.diff(controls[:, :3], axis=0), axis=1)
    step_numbers = np.arange(1, len(controls), dtype=int)
    axes["speed"].plot(
        step_numbers,
        step_distance,
        color=CONTROL_COLOR,
        linewidth=1.8,
        label="1 Hz waypoint distance",
    )
    axes["speed"].set_title(
        "Local 1 Hz step distance and turn slowdown", fontweight="bold"
    )
    axes["speed"].set_xlabel("control step index")
    axes["speed"].set_ylabel("waypoint distance (m)", color=CONTROL_COLOR)
    axes["speed"].tick_params(axis="y", labelcolor=CONTROL_COLOR)
    axes["speed"].grid(True, alpha=0.2)
    turn_axis = axes["speed"].twinx()
    turn_intensity = np.asarray(audit["control_turn_intensity"], dtype=float)
    turn_axis.plot(
        step_numbers,
        turn_intensity,
        color=DEVIATION_COLOR,
        linewidth=1.4,
        alpha=0.9,
        label="turn intensity",
    )
    turn_axis.fill_between(
        step_numbers, 0.0, turn_intensity, color=DEVIATION_COLOR, alpha=0.12
    )
    turn_axis.set_ylabel("turn intensity", color=DEVIATION_COLOR)
    turn_axis.tick_params(axis="y", labelcolor=DEVIATION_COLOR)
    turn_axis.set_ylim(0.0, 1.05)
    _plot_sampling_window(
        axes["tightest"],
        controls,
        image_indices,
        audit["strongest_turn_step_index"],
        "Strongest-turn window: reduced waypoint spacing",
    )
    _plot_sampling_window(
        axes["terminal"],
        controls,
        image_indices,
        len(controls) - 1,
        "Terminal window: true terminal always included",
    )
    figure.suptitle(
        f"{title}\n"
        f"1 Hz cruise arc step={audit['target_arc_step_m']:.3f} m | "
        f"waypoints={audit['control_waypoint_count']} | images={audit['image_waypoint_count']} | "
        f"image index gaps={audit['image_waypoint_index_gaps'][:8]}"
        f"{'...' if len(audit['image_waypoint_index_gaps']) > 8 else ''}",
        fontsize=14,
    )
    return figure, axes, audit


def plot_sampling_audit(
    result: RetimedTrajectory,
    output_path: Path,
    title: str,
    image_stride: int = 5,
    image_indices: np.ndarray | list[int] | None = None,
    image_stride_choices: tuple[int, ...] | None = None,
) -> dict:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, _, audit = build_sampling_audit_figure(
        result,
        title,
        image_stride=image_stride,
        image_indices=image_indices,
        image_stride_choices=image_stride_choices,
    )
    figure.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return audit
