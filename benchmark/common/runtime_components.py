from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from benchmark.common.config import load_class


def build_benchmark_runtime(cfg):
    return load_class(cfg.benchmark.class_path)(**cfg.benchmark.kwargs)


def build_model(cfg, worker_index=0):
    return WebsocketClientPolicy(
        uri=cfg.model.uri.format(worker_index=worker_index),
        timeout=cfg.model.request_timeout_sec,
    )
