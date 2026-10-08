"""Lightweight policy RPC client; no model or Torch dependency."""

from __future__ import annotations

import time

import websockets.sync.client

from . import msgpack_numpy


class WebsocketClientPolicy:
    def __init__(self, *, uri="ws://127.0.0.1:10093", timeout=120.0):
        self.timeout = float(timeout)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._ws = websockets.sync.client.connect(
                    uri, compression=None, max_size=None, open_timeout=max(0.1, deadline - time.monotonic()), proxy=None
                )
                self.metadata = msgpack_numpy.unpackb(self._ws.recv(timeout=self.timeout))
                break
            except ConnectionRefusedError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Policy connection deadline exceeded: {uri}")
                time.sleep(min(0.5, remaining))
        self.session_id = None

    def _request(self, method, payload):
        self._ws.send(msgpack_numpy.packb({"method": method, "session_id": self.session_id, "payload": payload}))
        try:
            response = msgpack_numpy.unpackb(self._ws.recv(timeout=self.timeout))
        except TimeoutError:
            # A stateful prediction may have committed. Close instead of replaying it.
            self._ws.close()
            raise
        if not response["ok"]:
            raise RuntimeError(f"Policy {method} failed: {response['error']}")
        return response["data"]

    def reset(self, *, session_id, episode_id, statistics_key=None, seed=42):
        self.session_id = str(session_id)
        return self._request("reset", dict(episode_id=str(episode_id), statistics_key=statistics_key, seed=int(seed)))

    def predict(self, *, frames, instruction):
        return self._request("predict", dict(frames=frames, instruction=instruction))

    def close_session(self, session_id):
        if self.session_id == session_id:
            self._request("close", {})
            self.session_id = None

    def close(self):
        self._ws.close()
