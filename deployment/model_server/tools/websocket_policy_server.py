"""Serial model inference with connection-owned, isolated episode sessions."""

import asyncio
import logging
import time

import websockets.asyncio.server

from . import msgpack_numpy


class WebsocketPolicyServer:
    def __init__(self, policy, host="127.0.0.1", port=10093, idle_timeout=-1, metadata=None):
        self._policy, self._host, self._port = policy, host, port
        self._idle_timeout = idle_timeout
        self._last_active = time.monotonic()
        self._metadata = {"protocol": "simplenav.session.v1", **(metadata or {})}

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            if self._idle_timeout <= 0:
                await server.serve_forever()
            else:
                while time.monotonic() - self._last_active < self._idle_timeout:
                    await asyncio.sleep(min(5, self._idle_timeout))

    async def _handler(self, websocket):
        sessions = {}
        await websocket.send(msgpack_numpy.packb(self._metadata))
        try:
            async for raw in websocket:
                self._last_active = time.monotonic()
                response = self._route_message(msgpack_numpy.unpackb(raw), sessions)
                await websocket.send(msgpack_numpy.packb(response))
        finally:
            sessions.clear()

    def _route_message(self, message, sessions):
        session_id = message["session_id"]
        method, payload = message["method"], message["payload"]
        try:
            if method == "reset":
                sessions[session_id] = self._policy.new_session(**payload)
                result = None
            elif method == "predict":
                result = self._policy.predict_session(sessions[session_id], **payload)
            elif method == "close":
                sessions.pop(session_id, None)
                result = None
            else:
                raise ValueError(f"Unknown policy method: {method}")
            return {"ok": True, "data": result}
        except Exception as exc:
            logging.exception("Policy %s failed for %s", method, session_id)
            # Failed state is invalid until reset; a partially committed step is not retried.
            sessions.pop(session_id, None)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
