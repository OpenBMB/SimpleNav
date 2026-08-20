from __future__ import annotations

from typing import Any

import numpy as np


def normalize_msgpack_payload(value: Any) -> Any:
    if hasattr(value, "to_msgpack"):
        return normalize_msgpack_payload(value.to_msgpack())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [normalize_msgpack_payload(item) for item in value.tolist()]
    if isinstance(value, list):
        return [normalize_msgpack_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_msgpack_payload(item) for item in value)
    if isinstance(value, dict):
        return {normalize_msgpack_payload(key): normalize_msgpack_payload(item) for key, item in value.items()}
    return value


def patch_msgpackrpc_transport() -> None:
    import msgpack
    import msgpackrpc

    base_socket = msgpackrpc.transport.tcp.BaseSocket
    if getattr(base_socket, "_airsim_eval_msgpack_compat_patch", False):
        return

    def patched_init(self, stream, encodings):
        self._stream = stream
        self._packer = msgpack.Packer(default=normalize_msgpack_payload, use_bin_type=True)
        self._unpacker = msgpack.Unpacker(raw=False, max_buffer_size=64 * 1024 * 1024)

    base_socket.__init__ = patched_init
    base_socket._airsim_eval_msgpack_compat_patch = True
