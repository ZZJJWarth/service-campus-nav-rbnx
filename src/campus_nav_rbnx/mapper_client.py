# SPDX-License-Identifier: MulanPSL-2.0
"""Loopback JSON-lines client for dedicated-server manual mapping mode (25577).

Used ONLY while manual survey/mapping mode is active in the Web UI, or to
display placed RTK base station locations on the campus map. Never used for
runtime autonomous robot localization or obstacle sensing.
"""
from __future__ import annotations

import json
import socket
import threading
from typing import Any


class ServerMapperClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 25577, timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._seq = 0

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.port <= 0:
            raise RuntimeError("server mapper port is disabled")
        with self._lock:
            self._seq += 1
            req_id = f"cn_{self._seq}"
        msg = {"id": req_id, **payload}
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as conn:
            conn.settimeout(self.timeout_s)
            conn.sendall(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
            line = conn.makefile("r", encoding="utf-8").readline()
        if not line:
            raise RuntimeError("empty response from server RTK endpoint")
        resp = json.loads(line)
        if not resp.get("ok"):
            err = resp.get("error") or {}
            raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))
        return resp.get("result") or {}

    def list_stations(self, dimension: str = "minecraft:overworld") -> list[dict[str, Any]]:
        result = self._request({"cmd": "rtk_stations_list", "dimension": dimension})
        return list(result.get("stations") or [])

    def poll_mapper(
        self,
        dimension: str = "minecraft:overworld",
        max_range_m: float = 256.0,
        noise_enabled: bool = True,
    ) -> dict[str, Any]:
        return self._request({
            "cmd": "rtk_mapper_poll",
            "dimension": dimension,
            "max_range_m": float(max_range_m),
            "noise_enabled": bool(noise_enabled),
        })
