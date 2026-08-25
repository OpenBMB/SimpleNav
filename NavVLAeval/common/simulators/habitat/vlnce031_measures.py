from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from typing import Any

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success


def structured_measurement_configs(*, split: str, gt_path: str, success_distance: float = 3.0) -> dict[str, Any]:
    from habitat.config.default_structured_configs import (
        DistanceToGoalMeasurementConfig,
        MeasurementConfig,
        SPLMeasurementConfig,
        SuccessMeasurementConfig,
    )

    @dataclass
    class PathLength031MeasurementConfig(MeasurementConfig):
        type: str = "PathLength031"

    @dataclass
    class OracleSuccess031MeasurementConfig(MeasurementConfig):
        type: str = "OracleSuccess031"
        success_distance: float = 3.0

    @dataclass
    class StepsTaken031MeasurementConfig(MeasurementConfig):
        type: str = "StepsTaken031"

    @dataclass
    class NDTW031MeasurementConfig(MeasurementConfig):
        type: str = "NDTW031"
        split: str = "val_seen"
        gt_path: str = ""
        success_distance: float = 3.0
        fdtw: bool = True

    return {
        "distance_to_goal": DistanceToGoalMeasurementConfig(distance_to="POINT"),
        "success": SuccessMeasurementConfig(success_distance=float(success_distance)),
        "spl": SPLMeasurementConfig(),
        "path_length": PathLength031MeasurementConfig(),
        "oracle_success": OracleSuccess031MeasurementConfig(success_distance=float(success_distance)),
        "steps_taken": StepsTaken031MeasurementConfig(),
        "ndtw": NDTW031MeasurementConfig(
            split=str(split),
            gt_path=str(gt_path),
            success_distance=float(success_distance),
            fdtw=True,
        ),
    }


def euclidean_distance(pos_a: Any, pos_b: Any) -> float:
    return float(np.linalg.norm(np.asarray(pos_b, dtype=np.float64) - np.asarray(pos_a, dtype=np.float64), ord=2))


def _dtw(path_a: list[Any], path_b: list[Any]) -> float:
    rows = len(path_a)
    cols = len(path_b)
    table = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    table[0, 0] = 0.0
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            cost = euclidean_distance(path_a[row - 1], path_b[col - 1])
            table[row, col] = cost + min(table[row - 1, col], table[row, col - 1], table[row - 1, col - 1])
    return float(table[rows, cols])


@registry.register_measure
class PathLength031(Measure):
    cls_uuid: str = "path_length"

    def __init__(self, *args: Any, sim: Simulator, **kwargs: Any) -> None:
        self._sim = sim
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._previous_position = np.asarray(self._sim.get_agent_state().position, dtype=np.float64)
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        current_position = np.asarray(self._sim.get_agent_state().position, dtype=np.float64)
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class OracleSuccess031(Measure):
    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any) -> None:
        self._success_distance = float(getattr(config, "success_distance", 3.0))
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        distance = float(task.measurements.measures[DistanceToGoal.cls_uuid].get_metric())
        self._metric = float(bool(self._metric) or distance < self._success_distance)


@registry.register_measure
class StepsTaken031(Measure):
    cls_uuid: str = "steps_taken"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        self._metric += 1.0


@registry.register_measure
class NDTW031(Measure):
    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._success_distance = float(getattr(config, "success_distance", 3.0))
        gt_path = str(getattr(config, "gt_path"))
        split = str(getattr(config, "split", "val_seen"))
        with gzip.open(gt_path.format(split=split), "rt", encoding="utf-8") as f:
            self.gt_json = json.load(f)
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, **kwargs: Any) -> None:
        self.locations: list[list[float]] = []
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        current_position = np.asarray(self._sim.get_agent_state().position, dtype=np.float64).tolist()
        if self.locations and current_position == self.locations[-1]:
            return
        self.locations.append(current_position)
        dtw_distance = _dtw(self.locations, self.gt_locations)
        self._metric = float(np.exp(-dtw_distance / (len(self.gt_locations) * self._success_distance)))


@registry.register_measure
class SDTW031(Measure):
    cls_uuid: str = "sdtw"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        task.measurements.check_measure_dependencies(self.uuid, [NDTW031.cls_uuid, Success.cls_uuid])
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any) -> None:
        success = float(task.measurements.measures[Success.cls_uuid].get_metric())
        ndtw = float(task.measurements.measures[NDTW031.cls_uuid].get_metric())
        self._metric = success * ndtw
