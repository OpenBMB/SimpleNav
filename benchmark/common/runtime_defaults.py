from benchmark.common.types import TerminationStatus


class BaseBenchmarkRuntime:
    """Small task hooks; simulator execution and model state have separate owners."""

    success_on_timeout = False

    def observe(self, episode, observation):
        pass

    def prepare_observation_for_model(self, *, episode, history, step, observation, instruction):
        return observation

    def log_step_artifacts(self, state, artifacts):
        return {}

    def finalize(self, episode, pose, *, reason, previous, oracle_success):
        if previous.done:
            return previous
        stopped = reason in {"model_stop", "waypoint_motion_stop"}
        success = int(self.is_success(pose, episode) and (stopped or self.success_on_timeout))
        return TerminationStatus(True, success, int(oracle_success), reason, None, None)
