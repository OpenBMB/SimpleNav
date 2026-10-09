"""Check optional simulator packages without importing their native runtimes."""

from importlib.util import find_spec


SIMULATOR_MODULES = {
    "airsim": ("airsim", "msgpackrpc"),
    "habitat": ("habitat", "habitat_sim", "magnum"),
    "unrealcv": ("unrealcv", "gym"),
}


def require_simulator(simulator: str) -> None:
    """Fail before starting a simulator when its extra has not been installed."""
    modules = SIMULATOR_MODULES.get(simulator)
    if modules is None:
        return  # Offline and externally configured backends have their own checks.
    missing = [module for module in modules if find_spec(module) is None]
    if missing:
        raise ImportError(
            f"Missing {simulator} dependencies: {', '.join(missing)}. "
            f"From the repository root run: uv sync --frozen --extra {simulator}. "
            "Include every simulator extra you want to keep installed. "
            "See README.md for simulator wheels and system requirements."
        )
