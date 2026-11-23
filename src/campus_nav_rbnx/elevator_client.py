# SPDX-License-Identifier: MulanPSL-2.0
"""Lightweight smart-building elevator controller client for campus_nav.

Talks to the Minecraft dedicated-server elevator endpoint (JSON-lines over TCP,
default 127.0.0.1:25577) using the physical controller & door-region bindings
from `elevator.local.json`.

Design boundary:
- Operates strictly as an external building-elevator IoT device (cabin location
  via floor/ceiling shaft planes, door open/close pulses, target-floor dispatch).
- NEVER queries or returns robot/player ground-truth coordinates for navigation.
- Zero dependency on the legacy RTAB-Map / Scene stack.
"""
from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger("campus_nav_rbnx.elevator")

_ADDRESS_RE = re.compile(r"^([^@]+)@(-?\d+),(-?\d+),(-?\d+)$")


def parse_controller_address(raw: str) -> dict[str, Any]:
    """Parse 'minecraft:overworld@x,y,z' into {dimension, x, y, z}."""
    m = _ADDRESS_RE.fullmatch(str(raw or "").strip())
    if not m:
        raise ValueError(f"invalid controller address: {raw!r}")
    return {
        "dimension": m.group(1),
        "x": int(m.group(2)),
        "y": int(m.group(3)),
        "z": int(m.group(4)),
    }


def _norm_floor_token(token: str) -> str:
    return re.sub(r"[\s_\-·]+", "", str(token or "").strip().lower())


def _canonical_floor_label(key: str, cfg: dict[str, Any]) -> str:
    """Return a clean short floor label like '1F', '2F', '3F', '4F'."""
    for alias in cfg.get("aliases") or []:
        s = str(alias).strip().upper()
        if re.fullmatch(r"B?\d+F", s):
            return s
    m = re.search(r"(\d+)f", str(key).lower())
    if m:
        return f"{m.group(1)}F"
    return str(cfg.get("name") or key)


class _TcpJsonClient:
    """Thread-safe one-shot JSON-lines client for the 25577 elevator server port."""

    def __init__(self, host: str = "127.0.0.1", port: int = 25577, timeout_s: float = 10.0) -> None:
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.port <= 0:
            raise RuntimeError("elevator mod port is disabled")
        req_id = str(uuid.uuid4())
        msg = {"id": req_id, **payload}
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as conn:
            conn.settimeout(self.timeout_s)
            conn.sendall(json.dumps(msg, separators=(",", ":")).encode("utf-8") + b"\n")
            with conn.makefile("rb") as stream:
                line = stream.readline(4 * 1024 * 1024 + 1)
        if not line.endswith(b"\n"):
            raise RuntimeError("incomplete response from elevator server endpoint")
        resp = json.loads(line.decode("utf-8"))
        if resp.get("id") != req_id:
            raise RuntimeError("elevator response id mismatch")
        if resp.get("ok") is not True:
            err = resp.get("error") or {}
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(detail or "elevator controller command failed")
        return dict(resp.get("result") or {})


class ElevatorClient:
    """Controls a moving-cabin multi-floor elevator from `elevator.local.json`."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        host: str | None = None,
        port: int | None = None,
        transport: Any = None,
        pulse_ticks: int = 4,
        settle_ticks: int = 21,
        poll_interval_s: float = 0.1,
    ) -> None:
        if not isinstance(config, dict) or not isinstance(config.get("floors"), dict) or not config["floors"]:
            raise ValueError("elevator config must define a non-empty 'floors' mapping")
        self.config = config
        self.moving_cabin = bool(config.get("moving_cabin", True))
        self.wait_timeout_s = float(config.get("wait_timeout", 90.0))
        self.pulse_ticks = max(2, int(pulse_ticks))
        self.settle_ticks = max(1, int(settle_ticks))
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        eff_host = host or str(config.get("elevator_mod_host") or "127.0.0.1")
        eff_port = int(port if port is not None else config.get("elevator_mod_port", 25577))
        self._client = transport if transport is not None else _TcpJsonClient(eff_host, eff_port)
        self._lock = threading.RLock()

        self.floors: dict[str, dict[str, Any]] = {}
        self._alias_to_key: dict[str, str] = {}
        self._key_to_label: dict[str, str] = {}
        for fkey, fcfg in config["floors"].items():
            if not isinstance(fcfg, dict):
                continue
            self.floors[fkey] = fcfg
            label = _canonical_floor_label(fkey, fcfg)
            self._key_to_label[fkey] = label
            tokens = [fkey, label, str(fcfg.get("name") or ""), *(fcfg.get("aliases") or [])]
            for tok in tokens:
                nt = _norm_floor_token(tok)
                if nt:
                    self._alias_to_key[nt] = fkey

        self._last_floor: str = ""
        self._last_door_open: bool = False
        self._last_error: str = ""

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> "ElevatorClient":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data, host=host, port=port)

    def resolve_floor(self, floor_label: str) -> str | None:
        return self._alias_to_key.get(_norm_floor_token(floor_label))

    def floor_label(self, floor_key: str) -> str:
        return self._key_to_label.get(floor_key, floor_key)

    def supports_floor(self, floor_label: str) -> bool:
        return self.resolve_floor(floor_label) is not None

    def floor_elevation_y(self, floor_label: str) -> float | None:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            return None
        bounds = self.floors[fkey].get("cabin_bounds") or {}
        mn = bounds.get("min")
        if isinstance(mn, list) and len(mn) >= 2:
            return float(mn[1])
        return None

    def _usable_cabin_enu_box(self, floor_label: str) -> tuple[float, float, float, float] | None:
        """Return usable cabin interior (xmin, xmax, ymin, ymax) in ENU, excluding cabin_door_region.

        Derived purely from static building config (`cabin_bounds` & `cabin_door_region`).
        Minecraft block coords [min..max] inclusive span [min, max+1] in continuous meters,
        and ENU y = -mc_z so [z_min, z_max+1] maps to [-(z_max+1), -z_min].
        """
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            return None
        fcfg = self.floors[fkey]
        cb = fcfg.get("cabin_bounds")
        if not isinstance(cb, dict):
            return None
        mn, mx = cb.get("min"), cb.get("max")
        if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) >= 3 and len(mx) >= 3):
            return None
        xmin = float(mn[0])
        xmax = float(mx[0]) + 1.0
        ymin = -(float(mx[2]) + 1.0)
        ymax = -float(mn[2])

        # Exclude the glass cabin door slice so standing targets & checks stay fully clear of the door.
        cd = fcfg.get("cabin_door_region")
        if isinstance(cd, dict) and isinstance(cd.get("min"), list) and isinstance(cd.get("max"), list):
            d_xmin, d_xmax = float(cd["min"][0]), float(cd["max"][0]) + 1.0
            d_ymin, d_ymax = -(float(cd["max"][2]) + 1.0), -float(cd["min"][2])
            if d_xmin == xmin and d_xmax < xmax:
                xmin = d_xmax
            elif d_xmax == xmax and d_xmin > xmin:
                xmax = d_xmin
            elif d_ymin == ymin and d_ymax < ymax:
                ymin = d_ymax
            elif d_ymax == ymax and d_ymin > ymin:
                ymax = d_ymin
        return xmin, xmax, ymin, ymax

    def safe_cabin_interior_xy(self, floor_label: str) -> tuple[float, float] | None:
        """Return a safe ENU target point deep inside the cabin, centered and clear of the door."""
        box = self._usable_cabin_enu_box(floor_label)
        if box is None:
            return None
        xmin, xmax, ymin, ymax = box
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        fkey = self.resolve_floor(floor_label)
        fcfg = self.floors[fkey] if fkey else {}
        cd = fcfg.get("cabin_door_region")
        if isinstance(cd, dict) and isinstance(cd.get("min"), list) and isinstance(cd.get("max"), list):
            door_x = 0.5 * (float(cd["min"][0]) + float(cd["max"][0]) + 1.0)
            door_y = -0.5 * (float(cd["min"][2]) + float(cd["max"][2]) + 1.0)
            dx, dy = cx - door_x, cy - door_y
            norm = (dx * dx + dy * dy) ** 0.5
            if norm > 1e-6:
                # Bias slightly deeper into the cabin away from the door, staying >= 0.75 m from walls.
                cand_x = cx + 0.15 * dx / norm
                cand_y = cy + 0.15 * dy / norm
                cx = max(xmin + 0.75, min(xmax - 0.75, cand_x))
                cy = max(ymin + 0.75, min(ymax - 0.75, cand_y))
        return round(cx, 3), round(cy, 3)

    def shaft_center_enu_xy(self) -> tuple[float, float] | None:
        for fkey in self.floors:
            pt = self.safe_cabin_interior_xy(fkey)
            if pt is not None:
                return pt
        return None

    def is_xy_safely_inside_cabin(
        self, floor_label: str, cx: float, cy: float, *, margin_m: float = 0.45
    ) -> bool:
        """Check whether sensor-fused robot pose (cx, cy) is fully inside the usable cabin interior."""
        box = self._usable_cabin_enu_box(floor_label)
        if box is None:
            return True
        xmin, xmax, ymin, ymax = box
        m = max(0.0, float(margin_m))
        return (xmin + m) <= cx <= (xmax - m) and (ymin + m) <= cy <= (ymax - m)

    def is_xy_safely_outside_cabin(
        self, floor_label: str, cx: float, cy: float, *, margin_m: float = 0.5
    ) -> bool:
        """Check whether sensor-fused robot pose (cx, cy) is fully clear of both cabin and landing door."""
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            return True
        fcfg = self.floors[fkey]
        cb = fcfg.get("cabin_bounds")
        if not isinstance(cb, dict):
            return True
        xmin = float(cb["min"][0])
        xmax = float(cb["max"][0]) + 1.0
        ymin = -(float(cb["max"][2]) + 1.0)
        ymax = -float(cb["min"][2])
        # Expand exclusion box to include landing door_region as well
        dr = fcfg.get("door_region")
        if isinstance(dr, dict) and isinstance(dr.get("min"), list) and isinstance(dr.get("max"), list):
            xmin = min(xmin, float(dr["min"][0]))
            xmax = max(xmax, float(dr["max"][0]) + 1.0)
            ymin = min(ymin, -(float(dr["max"][2]) + 1.0))
            ymax = max(ymax, -float(dr["min"][2]))
        m = max(0.0, float(margin_m))
        return cx < (xmin - m) or cx > (xmax + m) or cy < (ymin - m) or cy > (ymax + m)

    def configured_floors(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fkey, fcfg in self.floors.items():
            bounds = fcfg.get("cabin_bounds") or {}
            mn = bounds.get("min")
            out.append({
                "key": fkey,
                "label": self._key_to_label.get(fkey, fkey),
                "name": str(fcfg.get("name") or fkey),
                "elevation_y": float(mn[1]) if isinstance(mn, list) and len(mn) >= 2 else None,
            })
        return out

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "moving_cabin": self.moving_cabin,
            "floors": self.configured_floors(),
            "last_floor": self._key_to_label.get(self._last_floor, self._last_floor),
            "last_door_open": self._last_door_open,
            "last_error": self._last_error,
        }

    # ── low-level controller helpers ─────────────────────────────────────────
    def _command_block_pos(self, fkey: str, action: str) -> dict[str, int] | None:
        cmd_key = action.replace("_controller", "_command_block")
        raw = self.floors[fkey].get(cmd_key)
        if not raw:
            return None
        addr = parse_controller_address(raw)
        return {"x": addr["x"], "y": addr["y"], "z": addr["z"]}

    def _controller_payload(
        self,
        fkey: str,
        action: str,
        *,
        door_region: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fcfg = self.floors[fkey]
        addr = parse_controller_address(fcfg[action])
        payload: dict[str, Any] = dict(addr)
        if door_region is not None:
            payload["door_region"] = door_region
        cmd_pos = self._command_block_pos(fkey, action)
        if cmd_pos is not None:
            payload["command_block"] = cmd_pos
        return payload

    def _get(self, base_payload: dict[str, Any]) -> dict[str, Any]:
        return self._client.request({"cmd": "elevator_controller_get", **base_payload})

    def _set(self, base_payload: dict[str, Any], powered: bool) -> dict[str, Any]:
        return self._client.request(
            {"cmd": "elevator_controller_set", **base_payload, "powered": bool(powered)}
        )

    def _wait_ticks(
        self,
        base_payload: dict[str, Any],
        start_tick: int,
        ticks: int,
        timeout_s: float,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("elevator operation canceled")
            state = self._get(base_payload)
            if int(state.get("game_time", 0)) >= start_tick + ticks:
                return state
            if cancel_event is not None:
                cancel_event.wait(min(0.05, self.poll_interval_s))
            else:
                time.sleep(min(0.05, self.poll_interval_s))
        raise TimeoutError("server game_time did not advance during elevator pulse")

    def _pulse(
        self,
        base_payload: dict[str, Any],
        timeout_s: float,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        # Rising-edge pulse measured in game ticks (matches scripts/elevator_controller.py).
        state = self._set(base_payload, False)
        self._wait_ticks(base_payload, int(state.get("game_time", 0)), 2, timeout_s, cancel_event)
        try:
            state = self._set(base_payload, True)
            return self._wait_ticks(
                base_payload,
                int(state.get("game_time", 0)),
                self.pulse_ticks,
                timeout_s,
                cancel_event,
            )
        finally:
            try:
                self._set(base_payload, False)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _door_state(result: dict[str, Any]) -> str:
        region = result.get("door_region")
        if not isinstance(region, dict):
            raise RuntimeError("missing door_region observation from elevator endpoint")
        total = int(region.get("total_blocks", 0))
        air = int(region.get("air_blocks", 0))
        closed = int(region.get("closed_blocks", 0))
        if total <= 0:
            raise RuntimeError("empty door_region observation")
        if air == total:
            return "open"
        if closed == total:
            return "closed"
        return "mixed"

    def _landing_door_state(self, fkey: str, action: str = "door_open_controller") -> tuple[str, dict[str, Any]]:
        payload = self._controller_payload(fkey, action, door_region=self.floors[fkey]["door_region"])
        res = self._get(payload)
        return self._door_state(res), res

    def _cabin_door_state(self, fkey: str, action: str = "door_open_controller") -> str | None:
        cabin_region = self.floors[fkey].get("cabin_door_region")
        if not isinstance(cabin_region, dict):
            return None
        payload = self._controller_payload(fkey, action, door_region=cabin_region)
        return self._door_state(self._get(payload))

    # ── high-level elevator device API ───────────────────────────────────────
    def current_floor(self) -> str:
        """Observe which floor currently holds the physical moving cabin (returns floor key or '')."""
        observed: list[str] = []
        with self._lock:
            for fkey, fcfg in self.floors.items():
                cabin = fcfg.get("cabin_bounds")
                if not isinstance(cabin, dict):
                    continue
                for height in (int(cabin["min"][1]) - 1, int(cabin["max"][1]) + 1):
                    plane = {
                        "dimension": cabin["dimension"],
                        "min": [int(cabin["min"][0]), height, int(cabin["min"][2])],
                        "max": [int(cabin["max"][0]), height, int(cabin["max"][2])],
                        "closed_block": fcfg["door_region"]["closed_block"],
                    }
                    payload = self._controller_payload(fkey, "door_open_controller", door_region=plane)
                    st = self._get(payload).get("door_region") or {}
                    if not st.get("total_blocks") or st.get("air_blocks"):
                        break
                else:
                    observed.append(fkey)
        if len(observed) > 1:
            raise RuntimeError(f"multiple cabin locations observed: {observed}")
        found = observed[0] if observed else ""
        if found:
            self._last_floor = found
        return found

    def is_door_open(self, floor_label: str) -> bool:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        with self._lock:
            hall, _ = self._landing_door_state(fkey, "door_open_controller")
            if self.current_floor() != fkey:
                return False
            cabin = self._cabin_door_state(fkey, "door_open_controller")
            return hall == "open" and (cabin in (None, "open"))

    def set_door(
        self,
        floor_label: str,
        opened: bool,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        action = "door_open_controller" if opened else "door_close_controller"
        desired = "open" if opened else "closed"

        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("door operation canceled")
            cabin_here = self.current_floor() == fkey
            if opened and not cabin_here:
                raise RuntimeError(f"cannot open {fkey} door while cabin is not at {fkey}")
            hall, before = self._landing_door_state(fkey, action)
            cabin = self._cabin_door_state(fkey, action) if cabin_here else None
            if hall == desired and cabin in (None, desired):
                self._last_door_open = opened
                return
            if "command_tick" not in before:
                raise RuntimeError(f"door command_tick feedback unavailable for {fkey}.{action}")
            base = self._controller_payload(fkey, action, door_region=self.floors[fkey]["door_region"])

        self._pulse(base, eff_timeout, cancel_event)
        deadline = time.monotonic() + eff_timeout
        before_tick = int(before["command_tick"])
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("door operation canceled")
            with self._lock:
                after_hall, after = self._landing_door_state(fkey, action)
                after_cabin = self._cabin_door_state(fkey, action) if cabin_here else None
            if (
                int(after.get("command_tick", -1)) > before_tick
                and (int(after.get("command_success", 0)) > 0 or hall == desired)
                and after_hall == desired
                and after_cabin in (None, desired)
            ):
                self._last_door_open = opened
                log.info("elevator door %s confirmed %s", fkey, desired)
                return
            if cancel_event is not None:
                cancel_event.wait(self.poll_interval_s)
            else:
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"elevator door at {fkey} did not reach {desired}")

    def close_door(
        self,
        floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.set_door(floor_label, False, cancel_event=cancel_event, timeout_s=timeout_s)

    def ensure_all_doors_closed(
        self,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        for fkey in self.floors:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("elevator operation canceled")
            hall, _ = self._landing_door_state(fkey, "door_close_controller")
            cabin_here = self.current_floor() == fkey
            cabin = self._cabin_door_state(fkey, "door_close_controller") if cabin_here else None
            if hall != "closed" or (cabin is not None and cabin != "closed"):
                self.set_door(fkey, False, cancel_event=cancel_event, timeout_s=timeout_s)

    def move_cabin(
        self,
        target_floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Close all doors and dispatch the moving cabin to `target_floor_label`."""
        target_key = self.resolve_floor(target_floor_label)
        if not target_key:
            raise ValueError(f"unknown target elevator floor: {target_floor_label!r}")
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("elevator move canceled")

        cur = self.current_floor()
        if cur == target_key:
            return target_key

        self.ensure_all_doors_closed(cancel_event=cancel_event, timeout_s=eff_timeout)
        base = self._controller_payload(target_key, "target_controller")
        before = self._get(base)
        if "command_tick" not in before:
            raise RuntimeError(f"target command_tick feedback unavailable for {target_key}")
        before_tick = int(before["command_tick"])

        log.info("dispatching elevator cabin: %s -> %s", cur or "unknown", target_key)
        self._pulse(base, eff_timeout, cancel_event)

        deadline = time.monotonic() + eff_timeout
        stable_since: int | None = None
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("elevator move canceled while cabin in transit")
            after = self._get(base)
            observed = self.current_floor()
            if (
                int(after.get("command_tick", -1)) > before_tick
                and int(after.get("command_success", 0)) > 0
                and observed == target_key
            ):
                gt = int(after.get("game_time", 0))
                if stable_since is None:
                    stable_since = gt
                if gt - stable_since >= self.settle_ticks:
                    self._last_floor = target_key
                    log.info("elevator cabin settled at %s", target_key)
                    return target_key
            else:
                stable_since = None
            if cancel_event is not None:
                cancel_event.wait(self.poll_interval_s)
            else:
                time.sleep(self.poll_interval_s)
        raise TimeoutError(f"moving elevator cabin did not settle at {target_key}")

    def call_and_open(
        self,
        floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Bring the cabin to `floor_label` (if elsewhere) and open both doors."""
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        if self.current_floor() != fkey:
            self.move_cabin(fkey, cancel_event=cancel_event, timeout_s=timeout_s)
        self.set_door(fkey, True, cancel_event=cancel_event, timeout_s=timeout_s)
        return fkey


class NativeElevatorClient:
    """Controls one zero-command-block native Forge Mod elevator via pure IoT commands.

    Design boundary (Zero-Coordinate Runtime):
    - NEVER auto-discovers elevators or receives any world coordinates, cabin_bounds,
      door_region, or elevation_y from 25577 at runtime.
    - All geometric information (`cabin_bounds_enu`, `door_bounds_enu`, `cabin_xy`,
      `door_xy`, `elevation_y`) comes STRICTLY from the offline `IndoorFloorMap`
      recorded during offline mapping (`indoor_mapper` right-click bind).
    - Respects doorway light-curtain safety (`door_blocked: true`): waits & retries
      cleanly instead of forcing doors shut on obstacles.
    """

    def __init__(
        self,
        elevator_id: str,
        *,
        offline_floors: dict[str, dict[str, Any]] | None = None,
        dimension: str = "minecraft:overworld",
        host: str = "127.0.0.1",
        port: int = 25577,
        transport: Any = None,
        wait_timeout_s: float = 90.0,
        poll_interval_s: float = 0.15,
    ) -> None:
        self.elevator_id = str(elevator_id or "").strip()
        if not self.elevator_id:
            raise ValueError("native elevator missing 'elevator_id'")
        self.dimension = str(dimension or "minecraft:overworld")
        self.moving_cabin = True
        self.wait_timeout_s = float(wait_timeout_s)
        self.poll_interval_s = max(0.02, float(poll_interval_s))
        self._client = transport if transport is not None else _TcpJsonClient(host, port)
        self._lock = threading.RLock()

        self.floors: dict[str, dict[str, Any]] = {}
        self._alias_to_key: dict[str, str] = {}
        self._iot_floors: list[str] = []
        self._last_floor: str = ""
        self._last_door_open: bool = False
        self._last_door_blocked: bool = False
        self._last_error: str = ""
        for fl, geom in (offline_floors or {}).items():
            if isinstance(geom, dict):
                self.update_offline_floor(fl, **geom)

    def _register_floor_aliases(self, label: str) -> None:
        nt = _norm_floor_token(label)
        if nt:
            self._alias_to_key[nt] = label
        m = re.fullmatch(r"(\d+)f", nt)
        if m:
            num = m.group(1)
            for alias in (num, f"{num}楼", f"{num}层", f"floor{num}"):
                self._alias_to_key[_norm_floor_token(alias)] = label

    def update_offline_floor(
        self,
        floor_label: str,
        *,
        cabin_bounds_enu: tuple[float, float, float, float] | list[float] | None = None,
        door_bounds_enu: tuple[float, float, float, float] | list[float] | None = None,
        cabin_xy: tuple[float, float] | list[float] | None = None,
        door_xy: tuple[float, float] | list[float] | None = None,
        elevation_y: float | None = None,
    ) -> None:
        label = str(floor_label or "").strip()
        if not label:
            return
        with self._lock:
            cur = dict(self.floors.get(label) or {"label": label})
            if cabin_bounds_enu is not None and len(cabin_bounds_enu) >= 4:
                cur["cabin_bounds_enu"] = tuple(float(v) for v in cabin_bounds_enu[:4])
            if door_bounds_enu is not None and len(door_bounds_enu) >= 4:
                cur["door_bounds_enu"] = tuple(float(v) for v in door_bounds_enu[:4])
            if cabin_xy is not None and len(cabin_xy) >= 2:
                cur["cabin_xy"] = (float(cabin_xy[0]), float(cabin_xy[1]))
            if door_xy is not None and len(door_xy) >= 2:
                cur["door_xy"] = (float(door_xy[0]), float(door_xy[1]))
            if elevation_y is not None:
                cur["elevation_y"] = float(elevation_y)
            self.floors[label] = cur
            self._register_floor_aliases(label)

    def _update_iot_state(self, res: dict[str, Any]) -> None:
        with self._lock:
            if res.get("current_floor") is not None:
                self._last_floor = str(res.get("current_floor") or "")
            if "door_open" in res:
                self._last_door_open = bool(res.get("door_open"))
            if "door_blocked" in res:
                self._last_door_blocked = bool(res.get("door_blocked"))
            floors_raw = res.get("floors")
            if isinstance(floors_raw, list):
                labels = [str(f).strip() for f in floors_raw if isinstance(f, str) and str(f).strip()]
                if labels:
                    self._iot_floors = labels
                    for lbl in labels:
                        self._register_floor_aliases(lbl)

    def resolve_floor(self, floor_label: str) -> str | None:
        nt = _norm_floor_token(floor_label)
        hit = self._alias_to_key.get(nt)
        if hit:
            return hit
        m = re.search(r"(\d+)f$", nt)
        if m:
            hit = self._alias_to_key.get(f"{m.group(1)}f")
            if hit:
                return hit
        clean = str(floor_label or "").strip()
        return clean or None

    def floor_label(self, floor_key: str) -> str:
        return str(floor_key or "")

    def supports_floor(self, floor_label: str) -> bool:
        nt = _norm_floor_token(floor_label)
        return nt in self._alias_to_key

    def floor_elevation_y(self, floor_label: str) -> float | None:
        fkey = self.resolve_floor(floor_label)
        if not fkey or fkey not in self.floors:
            return None
        val = self.floors[fkey].get("elevation_y")
        return float(val) if val is not None else None

    def _usable_cabin_enu_box(self, floor_label: str) -> tuple[float, float, float, float] | None:
        fkey = self.resolve_floor(floor_label)
        if not fkey or fkey not in self.floors:
            return None
        box = self.floors[fkey].get("cabin_bounds_enu")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        return None

    def safe_cabin_interior_xy(self, floor_label: str) -> tuple[float, float] | None:
        box = self._usable_cabin_enu_box(floor_label)
        if box is not None:
            xmin, xmax, ymin, ymax = box
            return round(0.5 * (xmin + xmax), 3), round(0.5 * (ymin + ymax), 3)
        fkey = self.resolve_floor(floor_label)
        if fkey and fkey in self.floors and self.floors[fkey].get("cabin_xy") is not None:
            cx, cy = self.floors[fkey]["cabin_xy"]
            return round(float(cx), 3), round(float(cy), 3)
        return None

    def is_xy_safely_inside_cabin(
        self, floor_label: str, cx: float, cy: float, *, margin_m: float = 0.45
    ) -> bool:
        box = self._usable_cabin_enu_box(floor_label)
        if box is None:
            # Fallback if user manually mapped without right-click bounds: within 0.65 m of cabin_xy
            pt = self.safe_cabin_interior_xy(floor_label)
            if pt is None:
                return True
            return ((cx - pt[0]) ** 2 + (cy - pt[1]) ** 2) ** 0.5 <= 0.65
        xmin, xmax, ymin, ymax = box
        m = max(0.0, float(margin_m))
        return (xmin + m) <= cx <= (xmax - m) and (ymin + m) <= cy <= (ymax - m)

    def is_xy_safely_outside_cabin(
        self, floor_label: str, cx: float, cy: float, *, margin_m: float = 0.5
    ) -> bool:
        fkey = self.resolve_floor(floor_label)
        if not fkey or fkey not in self.floors:
            return True
        fcfg = self.floors[fkey]
        cbox = fcfg.get("cabin_bounds_enu")
        dbox = fcfg.get("door_bounds_enu")
        if not cbox and not dbox:
            return True
        boxes = [b for b in (cbox, dbox) if isinstance(b, (list, tuple)) and len(b) >= 4]
        xmin = min(float(b[0]) for b in boxes)
        xmax = max(float(b[1]) for b in boxes)
        ymin = min(float(b[2]) for b in boxes)
        ymax = max(float(b[3]) for b in boxes)
        m = max(0.0, float(margin_m))
        return cx < (xmin - m) or cx > (xmax + m) or cy < (ymin - m) or cy > (ymax + m)

    def _status(self) -> dict[str, Any]:
        res = self._client.request({
            "cmd": "elevator_status",
            "dimension": self.dimension,
            "elevator_id": self.elevator_id,
        })
        self._update_iot_state(res)
        return res

    def _sleep(self, sec: float, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None:
            cancel_event.wait(sec)
        else:
            time.sleep(sec)

    def current_floor(self) -> str:
        st = self._status()
        return str(st.get("current_floor") or "")

    def is_door_open(self, floor_label: str) -> bool:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        st = self._status()
        return str(st.get("state") or "") == "IDLE" and str(st.get("current_floor") or "") == fkey and bool(st.get("door_open"))

    def set_door(
        self,
        floor_label: str,
        opened: bool,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        deadline = time.monotonic() + eff_timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("door operation canceled")
            res = self._client.request({
                "cmd": "elevator_door",
                "dimension": self.dimension,
                "elevator_id": self.elevator_id,
                "floor": fkey,
                "open": bool(opened),
            })
            self._update_iot_state(res)
            if opened:
                log.info("native elevator %s door %s -> open", self.elevator_id, fkey)
                return
            blocked = bool(res.get("door_blocked", False)) or (res.get("action_accepted") is False)
            if not blocked and not bool(res.get("door_open", False)):
                log.info("native elevator %s door %s -> closed", self.elevator_id, fkey)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"电梯 {self.elevator_id} 门口防夹传感器持续检测到障碍物，等待关门超时")
            log.warning("native elevator %s doorway blocked by obstacle, waiting before retrying close...", self.elevator_id)
            self._sleep(min(0.4, max(0.05, self.poll_interval_s * 2.0)), cancel_event)

    def close_door(
        self,
        floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.set_door(floor_label, False, cancel_event=cancel_event, timeout_s=timeout_s)

    def ensure_all_doors_closed(
        self,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> None:
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        deadline = time.monotonic() + eff_timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("elevator operation canceled")
            res = self._client.request({
                "cmd": "elevator_door",
                "dimension": self.dimension,
                "elevator_id": self.elevator_id,
                "open": False,
            })
            self._update_iot_state(res)
            blocked = bool(res.get("door_blocked", False)) or (res.get("action_accepted") is False)
            if not blocked and not bool(res.get("door_open", False)):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"电梯 {self.elevator_id} 门口防夹传感器持续检测到障碍物，无法关门")
            self._sleep(min(0.4, max(0.05, self.poll_interval_s * 2.0)), cancel_event)

    def _dispatch_with_light_curtain_wait(
        self,
        cmd: str,
        payload: dict[str, Any],
        target_key: str,
        *,
        require_door_open: bool,
        timeout_s: float,
        cancel_event: threading.Event | None,
    ) -> str:
        deadline = time.monotonic() + timeout_s
        dispatched = False
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("elevator operation canceled")
            if not dispatched:
                res = self._client.request(payload)
                self._update_iot_state(res)
                blocked = bool(res.get("door_blocked", False)) or (res.get("action_accepted") is False)
                if blocked and str(res.get("state") or "") == "IDLE" and str(res.get("current_floor") or "") != target_key:
                    log.warning("native elevator %s dispatch delayed by doorway obstacle; retrying...", self.elevator_id)
                    self._sleep(min(0.4, max(0.05, self.poll_interval_s * 2.0)), cancel_event)
                    continue
                dispatched = True
                st = res
            else:
                st = self._status()
            if (
                str(st.get("state") or "") == "IDLE"
                and str(st.get("current_floor") or "") == target_key
                and (not require_door_open or bool(st.get("door_open")))
            ):
                return target_key
            # If elevator aborted departure due to late doorway obstruction, re-dispatch once clear.
            if str(st.get("state") or "") == "IDLE" and str(st.get("current_floor") or "") != target_key:
                dispatched = False
                self._sleep(min(0.4, max(0.05, self.poll_interval_s * 2.0)), cancel_event)
                continue
            self._sleep(self.poll_interval_s, cancel_event)
        raise TimeoutError(f"native elevator {self.elevator_id} did not arrive at {target_key} ({cmd})")

    def move_cabin(
        self,
        target_floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> str:
        fkey = self.resolve_floor(target_floor_label)
        if not fkey:
            raise ValueError(f"unknown target elevator floor: {target_floor_label!r}")
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        log.info("dispatching native elevator %s -> %s", self.elevator_id, fkey)
        return self._dispatch_with_light_curtain_wait(
            "elevator_move",
            {
                "cmd": "elevator_move",
                "dimension": self.dimension,
                "elevator_id": self.elevator_id,
                "floor": fkey,
            },
            fkey,
            require_door_open=False,
            timeout_s=eff_timeout,
            cancel_event=cancel_event,
        )

    def call_and_open(
        self,
        floor_label: str,
        cancel_event: threading.Event | None = None,
        timeout_s: float | None = None,
    ) -> str:
        fkey = self.resolve_floor(floor_label)
        if not fkey:
            raise ValueError(f"unknown elevator floor: {floor_label!r}")
        eff_timeout = float(timeout_s if timeout_s is not None else self.wait_timeout_s)
        log.info("calling native elevator %s -> %s (open_door=True)", self.elevator_id, fkey)
        return self._dispatch_with_light_curtain_wait(
            "elevator_call",
            {
                "cmd": "elevator_call",
                "dimension": self.dimension,
                "elevator_id": self.elevator_id,
                "floor": fkey,
                "open_door": True,
            },
            fkey,
            require_door_open=True,
            timeout_s=eff_timeout,
            cancel_event=cancel_event,
        )

    def configured_floors(self) -> list[dict[str, Any]]:
        return [
            {
                "key": fkey,
                "label": fkey,
                "name": fkey,
                "elevation_y": fcfg.get("elevation_y"),
            }
            for fkey, fcfg in self.floors.items()
        ]

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "id": self.elevator_id,
            "kind": "native",
            "enabled": True,
            "moving_cabin": True,
            "floors": self.configured_floors(),
            "last_floor": self._last_floor,
            "last_door_open": self._last_door_open,
            "last_door_blocked": self._last_door_blocked,
            "last_error": self._last_error,
        }


class MultiElevatorManager:
    """Offline-map-bound multi-elevator manager (ZERO runtime world-coordinate discovery).

    - Native reusable Mod elevators are resolved strictly by the `elevator_id` & offline
      geometry recorded on the `IndoorFloorMap` (via `indoor_mapper` right-click binding).
    - Legacy `elevator.local.json` fallback is preserved for existing Lab floor maps.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 25577,
        dimension: str = "minecraft:overworld",
        legacy: ElevatorClient | None = None,
        transport: Any = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self.dimension = str(dimension)
        self.legacy = legacy
        self._transport = transport
        self._lock = threading.RLock()
        self._native: dict[str, NativeElevatorClient] = {}
        self.wait_timeout_s = float(legacy.wait_timeout_s) if legacy is not None else 90.0

    def resolve_for_leg(self, leg: Any) -> NativeElevatorClient | ElevatorClient | None:
        """Return the elevator controller bound to a planned RouteLeg (zero coordinate query)."""
        elev_id = str(getattr(leg, "elevator_id", "") or "").strip()
        if elev_id:
            with self._lock:
                client = self._native.get(elev_id)
                if client is None:
                    client = NativeElevatorClient(
                        elev_id,
                        dimension=self.dimension,
                        host=self.host,
                        port=self.port,
                        transport=self._transport,
                        wait_timeout_s=self.wait_timeout_s,
                    )
                    self._native[elev_id] = client
            from_lbl = str(getattr(leg, "from_floor_label", "") or getattr(leg, "from_floor", "") or "")
            to_lbl = str(getattr(leg, "to_floor_label", "") or getattr(leg, "to_floor", "") or "")
            if from_lbl:
                client.update_offline_floor(
                    from_lbl,
                    cabin_bounds_enu=getattr(leg, "from_cabin_bounds_enu", None),
                    door_bounds_enu=getattr(leg, "from_door_bounds_enu", None),
                    cabin_xy=getattr(leg, "from_cabin_xy", None),
                    door_xy=getattr(leg, "from_door_xy", None),
                    elevation_y=getattr(leg, "from_elevation_y", None),
                )
            if to_lbl:
                client.update_offline_floor(
                    to_lbl,
                    cabin_bounds_enu=getattr(leg, "to_cabin_bounds_enu", None),
                    door_bounds_enu=getattr(leg, "to_door_bounds_enu", None),
                    cabin_xy=getattr(leg, "to_cabin_xy", None),
                    door_xy=getattr(leg, "to_door_xy", None),
                    elevation_y=getattr(leg, "to_elevation_y", None),
                )
            return client
        if self.legacy is not None:
            return self.legacy
        return None

    def supports_floor(self, floor_label: str) -> bool:
        with self._lock:
            if any(c.supports_floor(floor_label) for c in self._native.values()):
                return True
        return self.legacy is not None and self.legacy.supports_floor(floor_label)

    def status_snapshot(self, indoor_store: Any = None) -> dict[str, Any]:
        by_id: dict[str, dict[str, Any]] = {}
        if indoor_store is not None and hasattr(indoor_store, "all_elevators"):
            for el in indoor_store.all_elevators():
                eid = str(el.get("elevator_id") or "").strip()
                if not eid:
                    continue
                rec = by_id.setdefault(
                    eid,
                    {
                        "id": eid,
                        "kind": "native",
                        "enabled": True,
                        "moving_cabin": True,
                        "name": str(el.get("name") or "电梯"),
                        "building_name": str(el.get("building_name") or ""),
                        "floors": [],
                    },
                )
                rec["floors"].append({
                    "key": str(el.get("floor_label") or el.get("floor") or ""),
                    "label": str(el.get("floor") or ""),
                    "elevation_y": el.get("elevation_y"),
                })
        with self._lock:
            for eid, client in self._native.items():
                snap = client.status_snapshot()
                if eid in by_id:
                    by_id[eid].update({
                        "last_floor": snap["last_floor"],
                        "last_door_open": snap["last_door_open"],
                        "last_door_blocked": snap["last_door_blocked"],
                    })
                else:
                    by_id[eid] = snap
        items = list(by_id.values())
        if self.legacy is not None:
            leg_snap = dict(self.legacy.status_snapshot())
            leg_snap["id"] = str(self.legacy.config.get("elevator_id") or "legacy_elevator")
            leg_snap["kind"] = "legacy"
            items.append(leg_snap)
        return {
            "enabled": bool(items),
            "count": len(items),
            "elevators": items,
        }
