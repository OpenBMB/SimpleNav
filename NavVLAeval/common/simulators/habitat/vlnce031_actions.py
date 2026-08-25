from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import habitat_sim
import numpy as np
from gym import spaces
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry


def structured_action_configs(*, control_mode: str) -> dict[str, Any]:
    from habitat.config.default_structured_configs import ActionConfig, StopActionConfig

    @dataclass
    class MoveByPoseDelta031ActionConfig(ActionConfig):
        type: str = "MoveByPoseDelta031"
        control_mode: str = "filtered_pose_delta"

    return {
        "STOP": StopActionConfig(),
        "MOVE_BY_POSE_DELTA": MoveByPoseDelta031ActionConfig(control_mode=str(control_mode)),
    }


@registry.register_task_action
class MoveByPoseDelta031(SimulatorTaskAction):
    """Body-frame continuous pose-delta action for Habitat-Lab/Sim 0.3.1.

    ``collision_slide_pose_delta`` follows the requested straight-line
    displacement in short collision-filtered increments. It never invokes a
    navmesh route or target projection; Habitat's collision filter may apply
    its native tangential wall-slide response at each increment.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.control_mode = str(getattr(self._config, "control_mode", "filtered_pose_delta"))

    def step(
        self,
        *args: Any,
        dx: float,
        dy: float,
        dyaw: float,
        control_mode: str | None = None,
        **kwargs: Any,
    ):
        del args, kwargs
        agent_state = self._sim.get_agent_state()
        agent_pos = np.asarray(agent_state.position, dtype=np.float64)
        current_rot = agent_state.rotation
        mode = str(control_mode or self.control_mode)

        local_delta = _body_delta_to_habitat_local(dx, dy)
        if np.linalg.norm(local_delta) > 1e-6:
            world_delta = np.asarray(habitat_sim.utils.quat_rotate_vector(current_rot, local_delta), dtype=np.float64)
            target = agent_pos + world_delta
            if mode == "direct_pose_delta":
                new_pos = target
            elif mode == "filtered_pose_delta":
                new_pos = np.asarray(self._sim.step_filter(agent_pos, target), dtype=np.float64)
                if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
                    new_pos = agent_pos
                else:
                    new_pos = np.asarray(self._sim.pathfinder.snap_point(new_pos), dtype=np.float64)
                    if np.any(np.isnan(new_pos)) or not self._sim.is_navigable(new_pos):
                        new_pos = agent_pos
            elif mode == "collision_stop_pose_delta":
                new_pos = _collision_stop_pose_delta_position(self._sim, agent_pos, target)
            elif mode == "collision_slide_pose_delta":
                new_pos = _collision_slide_pose_delta_position(self._sim, agent_pos, target)
            elif mode == "geodesic_pose_delta":
                new_pos = _geodesic_pose_delta_position(self._sim, agent_pos, target)
            else:
                raise ValueError(f"Unsupported MoveByPoseDelta031 control_mode={mode!r}")
        else:
            new_pos = agent_pos

        if abs(float(dyaw)) > 1e-6:
            dyaw_quat = habitat_sim.utils.quat_from_angle_axis(float(dyaw), habitat_sim.geo.UP)
            new_rot = current_rot * dyaw_quat
        else:
            new_rot = current_rot

        return self._sim.get_observations_at(
            position=new_pos,
            rotation=new_rot,
            keep_agent_at_new_pose=True,
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "dx": spaces.Box(low=np.asarray([-100.0]), high=np.asarray([100.0]), dtype=np.float32),
                "dy": spaces.Box(low=np.asarray([-100.0]), high=np.asarray([100.0]), dtype=np.float32),
                "dyaw": spaces.Box(low=np.asarray([-np.pi]), high=np.asarray([np.pi]), dtype=np.float32),
            }
        )


def _body_delta_to_habitat_local(dx: float, dy: float) -> np.ndarray:
    """Convert VLN-CE ``(left, forward)`` to Habitat's ``(right, up, back)``."""
    # Keep the original VLN-CE MoveByPoseDelta convention: ``dx`` is
    # lateral-left and ``dy`` is forward. Habitat's local +x is right, so the
    # lateral component must be negated before applying it directly.
    return np.asarray([-float(dx), 0.0, -float(dy)], dtype=np.float64)


def _geodesic_pose_delta_position(sim: Any, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Advance toward ``target`` on the navmesh without exceeding its action budget.

    The action budget is the direct body-frame displacement norm.  A reachable
    target therefore remains unchanged.  If a wall blocks the direct segment,
    the same distance is spent along the shortest navmesh path instead.  For an
    off-navmesh target, its nearest navigable projection is used only when it
    belongs to the current connected component.  Falling back to ``step_filter``
    keeps collision behavior conservative when no valid route exists.
    """
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    distance_budget = float(np.linalg.norm(target - start))
    if distance_budget <= 1e-6:
        return start

    candidate = target
    if not sim.is_navigable(candidate):
        candidate = np.asarray(sim.pathfinder.snap_point(candidate), dtype=np.float64)
        if np.any(np.isnan(candidate)) or not sim.is_navigable(candidate):
            return _filtered_navmesh_position(sim, start, target)

    shortest_path = habitat_sim.ShortestPath()
    shortest_path.requested_start = start.astype(np.float32)
    shortest_path.requested_end = candidate.astype(np.float32)
    if sim.pathfinder.find_path(shortest_path):
        path_points = [np.asarray(point, dtype=np.float64) for point in shortest_path.points]
        routed_target = _advance_along_navmesh_path(start, path_points, distance_budget)
        guarded = np.asarray(sim.step_filter(start, routed_target), dtype=np.float64)
        if np.all(np.isfinite(guarded)) and sim.is_navigable(guarded):
            return guarded

    return _filtered_navmesh_position(sim, start, target)


def _collision_stop_pose_delta_position(
    sim: Any,
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_substep_m: float = 0.05,
) -> np.ndarray:
    """Move straight toward ``target`` and stop at the first blocked substep.

    A single long ``step_filter`` call can jump over a narrow obstacle when its
    endpoint is navigable. Incremental filtering gives a velocity-like swept
    path while retaining Habitat's navmesh collision behavior. A filter can
    also slide tangentially along a wall; that changes the model's command, so
    this controller stops before such a lateral deviation. No path finding or
    target projection is involved.
    """
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return start

    substeps = max(1, int(np.ceil(distance / float(max_substep_m))))
    current = start.copy()
    for index in range(1, substeps + 1):
        desired = start + delta * (float(index) / float(substeps))
        filtered = np.asarray(sim.step_filter(current, desired), dtype=np.float64)
        if np.any(~np.isfinite(filtered)) or not sim.is_navigable(filtered):
            return current
        requested_step = desired - current
        desired_distance = float(np.linalg.norm(requested_step))
        actual_step = filtered - current
        if desired_distance <= 1e-8:
            continue
        requested_direction = requested_step / desired_distance
        forward_progress = float(np.dot(actual_step, requested_direction))
        lateral_deviation = float(np.linalg.norm(actual_step - forward_progress * requested_direction))
        # Habitat's navmesh filter may return an equally long tangential slide
        # rather than a shorter blocked step. Treat either outcome as a stop.
        if lateral_deviation > 1e-3:
            return current
        if forward_progress < desired_distance * (1.0 - 1e-3):
            return filtered
        current = filtered
    return current


def _collision_slide_pose_delta_position(
    sim: Any,
    start: np.ndarray,
    target: np.ndarray,
    *,
    max_substep_m: float = 0.05,
) -> np.ndarray:
    """Execute the entire direct displacement through collision-filtered substeps.

    Every substep applies the same fraction of the model direct waypoint
    command from the agent's actual current position. There is no shortest path,
    target snap, or early return on collision: a zero-displacement block is
    retried by later command increments, while a tangential displacement from
    Habitat step_filter is retained as wall sliding.
    """
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    delta = target - start
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return start

    substeps = max(1, int(np.ceil(distance / float(max_substep_m))))
    command_delta = delta / float(substeps)
    current = start.copy()
    for _ in range(substeps):
        desired = current + command_delta
        filtered = np.asarray(sim.step_filter(current, desired), dtype=np.float64)
        if np.all(np.isfinite(filtered)) and sim.is_navigable(filtered):
            current = filtered
    return current


def _filtered_navmesh_position(sim: Any, start: np.ndarray, target: np.ndarray) -> np.ndarray:
    filtered = np.asarray(sim.step_filter(start, target), dtype=np.float64)
    if np.any(np.isnan(filtered)) or not sim.is_navigable(filtered):
        return np.asarray(start, dtype=np.float64)
    return filtered


def _advance_along_navmesh_path(start: np.ndarray, path_points: list[np.ndarray], distance_budget: float) -> np.ndarray:
    """Return the point reached after walking at most ``distance_budget`` on a polyline."""
    current = np.asarray(start, dtype=np.float64)
    remaining = max(0.0, float(distance_budget))
    for point in path_points:
        point = np.asarray(point, dtype=np.float64)
        segment = point - current
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-8:
            current = point
            continue
        if remaining < segment_length:
            return current + segment * (remaining / segment_length)
        current = point
        remaining -= segment_length
    return current
