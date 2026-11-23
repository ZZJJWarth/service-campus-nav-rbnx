# SPDX-License-Identifier: MulanPSL-2.0
"""Pure-sensor doorway inspection, automatic door opening, and narrow 1-block doorway helpers.

Design boundary:
- Uses ONLY the rolling local costmap (`LocalCostGridSnapshot`), onboard `minecraft_odom` pose,
  optional RGB-D visual door detections (`8094 /api/objects`), and the physical crosshair
  right-click primitive (`cmd: "interact_point"` on 127.0.0.1:25576).
- NEVER queries or receives Minecraft block IDs, door open/closed NBT states, or ground-truth poses.
"""
from __future__ import annotations

import json
import logging
import math
import socket
import threading
import time
import uuid
from typing import Any, Callable
from urllib.request import ProxyHandler, Request, build_opener

log = logging.getLogger("campus_nav_rbnx.door")


class DoorInteractor:
    """Executes pure-physical right-click interactions via the Forge client loopback (`25576`)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25576,
        timeout_s: float = 8.0,
        transport: Any = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._transport = transport

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport.request(payload)
        if self.port <= 0:
            raise RuntimeError("Minecraft client command port is disabled")
        req_id = str(uuid.uuid4())
        msg = {"id": req_id, **payload}
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as conn:
            conn.settimeout(self.timeout_s)
            conn.sendall(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
            with conn.makefile("rb") as stream:
                line = stream.readline(1024 * 1024 + 1)
        if not line.endswith(b"\n"):
            raise RuntimeError("incomplete response from Minecraft client command port")
        resp = json.loads(line.decode("utf-8"))
        if resp.get("id") != req_id:
            raise RuntimeError("Minecraft client response id mismatch")
        if resp.get("ok") is not True:
            err = resp.get("error") or {}
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(detail or "interact_point command failed")
        return dict(resp.get("result") or {})

    def interact_odom_point(
        self,
        odom_x: float,
        odom_y: float,
        odom_z: float,
        *,
        max_reach: float = 4.2,
    ) -> dict[str, Any]:
        """Convert a pure-sensor minecraft_odom ENU point (E, N, U) to MC coords and right-click."""
        return self._request({
            "cmd": "interact_point",
            "x": float(odom_x),
            "y": float(odom_z),
            "z": float(-odom_y),
            "max_reach": float(max_reach),
        })

    def interact_relative_point(
        self,
        forward_m: float,
        left_m: float,
        up_m: float,
        *,
        max_reach: float = 4.2,
    ) -> dict[str, Any]:
        """Pure-physical body-relative right-click (forward, left, up from feet)."""
        return self._request({
            "cmd": "interact_point",
            "relative": True,
            "forward": float(forward_m),
            "left": float(left_m),
            "up": float(up_m),
            "max_reach": float(max_reach),
        })


def is_doorway_blocked(
    grid: Any,
    door_ox: float,
    door_oy: float,
    heading_yaw: float,
    *,
    span_m: float = 0.65,
    lethal_cost: int = 80,
) -> bool:
    """Pure-sensor costmap check: returns True if a closed door/barrier blocks the doorway center."""
    if grid is None:
        return False
    ux, uy = math.cos(heading_yaw), math.sin(heading_yaw)
    px, py = -uy, ux
    # Sample along the doorway crossing line; if at any longitudinal slice ALL three center-band
    # samples (-0.15, 0, +0.15 m) are lethal (>= lethal_cost), the doorway is closed/blocked!
    for s in (-span_m, -0.35, -0.15, 0.0, 0.15, 0.35, span_m):
        bx = door_ox + ux * s
        by = door_oy + uy * s
        blocked_all = True
        saw_known = False
        for lat in (-0.15, 0.0, 0.15):
            c = grid.cost_at(bx + px * lat, by + py * lat)
            if c is None or c < 0:
                blocked_all = False
                continue
            saw_known = True
            if c < lethal_cost:
                blocked_all = False
                break
        if saw_known and blocked_all:
            return True
    return False


def is_doorway_confirmed_open(
    grid: Any,
    door_ox: float,
    door_oy: float,
    heading_yaw: float,
    *,
    span_m: float = 0.65,
    lethal_cost: int = 80,
    min_known_free: int = 2,
) -> bool:
    """Strict pure-sensor open verification: requires known free cells AND no blocked slice.

    Returns False when the cost grid is missing (None) or all doorway samples are UNKNOWN (-1).
    """
    if grid is None:
        return False
    if is_doorway_blocked(
        grid, door_ox, door_oy, heading_yaw, span_m=span_m, lethal_cost=lethal_cost
    ):
        return False
    ux, uy = math.cos(heading_yaw), math.sin(heading_yaw)
    px, py = -uy, ux
    known_free = 0
    for s in (-0.35, 0.0, 0.35):
        bx = door_ox + ux * s
        by = door_oy + uy * s
        for lat in (-0.15, 0.0, 0.15):
            c = grid.cost_at(bx + px * lat, by + py * lat)
            if c is not None and 0 <= c < lethal_cost:
                known_free += 1
    return known_free >= min_known_free


def doorway_clear_width_m(
    grid: Any,
    door_ox: float,
    door_oy: float,
    heading_yaw: float,
    *,
    max_half_m: float = 1.6,
    lethal_cost: int = 85,
) -> float | None:
    """Estimate passable lateral width (m) at a doorway from the local costmap."""
    if grid is None:
        return None
    px, py = -math.sin(heading_yaw), math.cos(heading_yaw)
    step = max(0.1, min(0.2, float(getattr(grid, "resolution", 0.2))))
    left_clear = 0.0
    d = step
    while d <= max_half_m:
        c = grid.cost_at(door_ox + px * d, door_oy + py * d)
        if c is None or c >= lethal_cost:
            break
        left_clear = d
        d += step
    right_clear = 0.0
    d = step
    while d <= max_half_m:
        c = grid.cost_at(door_ox - px * d, door_oy - py * d)
        if c is None or c >= lethal_cost:
            break
        right_clear = d
        d += step
    return round(left_clear + right_clear + step, 3)


def query_visual_doors(visual_object_url: str, *, timeout_s: float = 1.2) -> list[dict[str, Any]]:
    """Query visually detected RGB-D closed doors from service-visual-object-rbnx (pure sensor)."""
    base = (visual_object_url or "").strip().rstrip("/")
    if not base:
        return []
    try:
        opener = build_opener(ProxyHandler({}))
        req = Request(f"{base}/api/objects", method="GET")
        with opener.open(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read(65536).decode("utf-8"))
        items = data.get("objects") or []
        return [obj for obj in items if str(obj.get("label") or "").lower() == "door"]
    except Exception:  # noqa: BLE001
        return []


def ensure_door_open_and_verified(
    interactor: DoorInteractor,
    cost_grid_provider: Callable[[], Any],
    door_ox: float,
    door_oy: float,
    robot_z: float,
    heading_yaw: float,
    *,
    robot_odom_pose: tuple[float, float, float] | None = None,
    reset_local_map_fn: Callable[[], bool] | None = None,
    cancel_event: threading.Event | None = None,
    settle_s: float = 0.45,
) -> tuple[bool, str]:
    """Check if a doorway is blocked; if so, right-click to open and verify via LiDAR/costmap.

    Returns (is_passable, status_code):
    - (True, "already_open"): doorway center was already clear -> did NOT click (avoids closing open door).
    - (True, "opened_center" | "opened_side"): clicked and confirmed doorway cleared in costmap.
    - (False, "still_blocked"): clicked center and side offsets (e.g. button/double door), still blocked.
    """
    def _wait(sec: float) -> bool:
        if cancel_event is not None:
            return cancel_event.wait(sec)
        time.sleep(sec)
        return False

    grid = cost_grid_provider()
    if is_doorway_confirmed_open(grid, door_ox, door_oy, heading_yaw):
        return True, "already_open"

    px, py = -math.sin(heading_yaw), math.cos(heading_yaw)
    # Candidate interaction points:
    # 1) Door panel center (+1.05 m above feet)
    # 2) Left/right side (+0.55 m laterally, +1.15 m height) for double doors or adjacent wall buttons
    candidates = [
        ("opened_center", door_ox, door_oy, robot_z + 1.05),
        ("opened_side", door_ox + px * 0.55, door_oy + py * 0.55, robot_z + 1.15),
        ("opened_side", door_ox - px * 0.55, door_oy - py * 0.55, robot_z + 1.15),
    ]

    last_err = "still_blocked"
    for tag, tx, ty, tz in candidates:
        if cancel_event is not None and cancel_event.is_set():
            return False, "canceled"
        try:
            if robot_odom_pose is not None and hasattr(interactor, "interact_relative_point"):
                rox, roy, royaw = robot_odom_pose
                dx, dy = tx - rox, ty - roy
                cy, sy = math.cos(royaw), math.sin(royaw)
                fwd = cy * dx + sy * dy
                left = -sy * dx + cy * dy
                up = tz - robot_z
                resp = interactor.interact_relative_point(fwd, left, up, max_reach=4.2)
            else:
                resp = interactor.interact_odom_point(tx, ty, tz, max_reach=4.2)
            if isinstance(resp, dict) and resp.get("interacted") is False:
                last_err = "not_interacted"
        except Exception as exc:  # noqa: BLE001
            log.debug("door interact_odom_point(%s) failed: %s", tag, exc)
            last_err = "interact_error"
            continue
        if _wait(min(0.25, settle_s)):
            return False, "canceled"
        if reset_local_map_fn is not None:
            try:
                reset_local_map_fn()
            except Exception:  # noqa: BLE001
                pass
        if _wait(settle_s):
            return False, "canceled"
        grid_after = cost_grid_provider()
        if grid_after is None:
            last_err = "missing_cost_grid"
            continue
        if is_doorway_confirmed_open(grid_after, door_ox, door_oy, heading_yaw):
            log.info("doorway at (%.2f, %.2f) confirmed open after %s", door_ox, door_oy, tag)
            return True, tag
        if is_doorway_blocked(grid_after, door_ox, door_oy, heading_yaw):
            last_err = "still_blocked"
        else:
            last_err = "unknown_cost_grid"

    return False, last_err
