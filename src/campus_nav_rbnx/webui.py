# SPDX-License-Identifier: MulanPSL-2.0
"""HTTP server for the Google Maps-style campus map editor and global navigator."""
from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

log = logging.getLogger("campus_nav_rbnx.webui")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"
MAX_POST_BYTES = 2 * 1024 * 1024  # 2 MiB for rich campus vector maps


class CampusWebUI:
    def __init__(
        self,
        host: str,
        port: int,
        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        self._host = host
        self._port = int(port)
        self._handlers = dict(handlers)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        if self._server is not None:
            return int(self._server.server_address[1])
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return
        handlers = self._handlers

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_bytes(self, status: HTTPStatus, ctype: str, body: bytes) -> None:
                self.send_response(int(status))
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send_bytes(status, "application/json; charset=utf-8", body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_POST_BYTES:
                    raise ValueError("request body out of range")
                if length == 0:
                    return {}
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON body must be an object")
                return data

            def do_HEAD(self) -> None:  # noqa: N802
                self.do_GET()

            def do_GET(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                if route in ("", "index.html"):
                    self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
                    return
                fn = handlers.get(f"GET:{route}")
                if fn is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                    return
                try:
                    self._send_json(HTTPStatus.OK, fn({}))
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                route = urlsplit(self.path).path.strip("/")
                fn = handlers.get(f"POST:{route}")
                if fn is None:
                    self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"ok": False, "error": "unsupported endpoint"})
                    return
                try:
                    body = self._read_json()
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "detail": str(exc)})
                    return
                try:
                    res = fn(body)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "detail": str(exc)})
                    return
                status = HTTPStatus.OK if res.get("ok", True) else HTTPStatus.CONFLICT
                self._send_json(status, res)

        class _ReusableServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = _ReusableServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.25},
            name="campus-nav-webui",
            daemon=True,
        )
        self._thread.start()
        log.info("campus map web UI listening on %s", self.url)

    def stop(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            try:
                server.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                server.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=2.0)
