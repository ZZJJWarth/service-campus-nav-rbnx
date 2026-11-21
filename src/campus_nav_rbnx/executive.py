# SPDX-License-Identifier: MulanPSL-2.0
"""Receding-horizon piecewise campus & multi-floor navigation executive.

Turns a campus-scale or multi-floor global route into rolling local Nav2 subgoals
inside the 64x64 m local mapping window, smoothly advancing without full stops,
triggering global road-graph rerouting when a street is completely blocked, and
executing smart-elevator transitions (call -> open -> board -> ride -> reset local
map & switch floor -> exit -> close) across floors.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from campus_nav_rbnx.campus_map import CampusMap
from campus_nav_rbnx.global_planner import GlobalPlanner, GlobalRoute, RouteLeg
from campus_nav_rbnx.params import CampusNavConfig
from campus_nav_rbnx.rtk_fusion import RtkOdomFusion

log = logging.getLogger("campus_nav_rbnx.executive")

NAV_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"})

_NAVIGATION_RPC = {
    "navigate": (
        "robonix/service/navigation/navigate",
        "RobonixServiceNavigationNavigateStub",
        "Navigate",
        "Navigate_Request",
    ),
    "status": (
        "robonix/service/navigation/navigate/status",
        "RobonixServiceNavigationNavigateStatusStub",
        "GetNavigationStatus",
        "GetNavigationStatus_Request",
    ),
    "cancel": (
        "robonix/service/navigation/navigate/cancel",
        "RobonixServiceNavigationNavigateCancelStub",
        "CancelNavigation",
        "CancelNavigation_Request",
    ),
}


def default_navigation_rpc(
    operation: str, payload: dict[str, Any], timeout_s: float = 10.0
) -> dict[str, Any]:
    """Call standard robonix/service/navigation/* gRPC capabilities via Atlas."""
    import grpc
    import navigation_pb2  # type: ignore
    import robonix_contracts_pb2_grpc as contracts_grpc  # type: ignore
    from robonix_api import ATLAS, Transport

    contract_id, stub_name, method_name, request_name = _NAVIGATION_RPC[operation]
    channel_ref = None
    try:
        caps = ATLAS.find_capability(contract_id=contract_id, transport=Transport.GRPC)
        if not caps:
            return {"ok": False, "detail": f"no provider for {contract_id}"}
        provider = caps[0]
        channel_ref = ATLAS.connect_capability(
            consumer_id=os.environ.get("ROBONIX_CAPABILITY_ID", "campus_nav"),
            provider_id=provider.provider_id,
            contract_id=contract_id,
            transport=Transport.GRPC,
        )
        endpoint = (channel_ref.endpoint or "").strip()
        if not endpoint:
            return {"ok": False, "detail": f"empty endpoint for {contract_id}"}
        with grpc.insecure_channel(endpoint) as ch:
            stub = getattr(contracts_grpc, stub_name)(ch)
            req = getattr(navigation_pb2, request_name)(**payload)
            resp = getattr(stub, method_name)(req, timeout=timeout_s)
        return {f.name: v for f, v in resp.ListFields()}
    except grpc.RpcError as exc:
        return {"ok": False, "detail": f"{contract_id}: {exc.code().name}: {exc.details()}"}
    except Exception as exc:  # noqa: BLE001
        log.exception("campus_nav navigation rpc %s failed", contract_id)
        return {"ok": False, "detail": str(exc)}
    finally:
        if channel_ref is not None:
            try:
                channel_ref.close()
            except Exception:  # noqa: BLE001
                pass


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class LocalCostGridSnapshot:
    origin_x: float
    origin_y: float
    resolution: float
    width: int
    height: int
    data: np.ndarray  # shape (height, width), int8

    def cost_at(self, wx: float, wy: float) -> int | None:
        col = int(math.floor((wx - self.origin_x) / self.resolution))
        row = int(math.floor((wy - self.origin_y) / self.resolution))
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return None
        return int(self.data[row, col])

    def nudge_to_passable(
        self, wx: float, wy: float, search_radius_m: float = 3.0
    ) -> tuple[float, float]:
        if search_radius_m <= 0.0:
            return wx, wy
        cur = self.cost_at(wx, wy)
        if cur is None or cur < 85:
            return wx, wy
        rad_cells = max(1, int(math.ceil(search_radius_m / self.resolution)))
        c0 = int(math.floor((wx - self.origin_x) / self.resolution))
        r0 = int(math.floor((wy - self.origin_y) / self.resolution))
        best_xy = (wx, wy)
        best_score = float("inf")
        for dr in range(-rad_cells, rad_cells + 1):
            rr = r0 + dr
            if rr < 0 or rr >= self.height:
                continue
            for dc in range(-rad_cells, rad_cells + 1):
                cc = c0 + dc
                if cc < 0 or cc >= self.width:
                    continue
                val = int(self.data[rr, cc])
                if val >= 85:
                    continue
                dist = math.hypot(dc, dr) * self.resolution
                if dist > search_radius_m:
                    continue
                penalty = 0.5 if val < 0 else (val / 50.0)
                score = dist + penalty
                if score < best_score:
                    best_score = score
                    best_xy = (
                        self.origin_x + (cc + 0.5) * self.resolution,
                        self.origin_y + (rr + 0.5) * self.resolution,
                    )
        return best_xy

    def pick_safe_explore_target(
        self,
        robot_ox: float,
        robot_oy: float,
        visited_xy: list[tuple[float, float]],
        *,
        min_dist_m: float = 3.0,
        max_dist_m: float = 5.5,
    ) -> tuple[float, float, float] | None:
        """Pick an open, collision-free corridor spot in local odom for active marker search.

        Uses strictly the local costmap (zero global map or game-truth coordinates).
        Returns (target_ox, target_oy, heading_rad) or None if no safe open spot is visible.
        """
        best: tuple[float, float, float] | None = None
        best_score = -1e9
        step_m = max(0.25, self.resolution)
        for deg in range(0, 360, 15):
            rad = math.radians(deg)
            ux, uy = math.cos(rad), math.sin(rad)
            # Ray-march outward until we hit an obstacle/unknown/wall-aversion cell
            clear_d = 0.0
            d = step_m
            while d <= max_dist_m:
                c = self.cost_at(robot_ox + ux * d, robot_oy + uy * d)
                if c is None or c < 0 or c >= 30:
                    break
                clear_d = d
                d += step_m
            if clear_d < min_dist_m:
                continue
            # Pull back slightly from the ray tip for extra wall clearance
            target_d = max(min_dist_m, min(max_dist_m - 0.5, clear_d - 0.6))
            tx = robot_ox + ux * target_d
            ty = robot_oy + uy * target_d
            # Verify 3x3 neighborhood around target is completely open
            clear_nb = True
            nb_cost = 0
            for dx in (-0.4, 0.0, 0.4):
                for dy in (-0.4, 0.0, 0.4):
                    vc = self.cost_at(tx + dx, ty + dy)
                    if vc is None or vc < 0 or vc >= 35:
                        clear_nb = False
                        break
                    nb_cost = max(nb_cost, vc)
                if not clear_nb:
                    break
            if not clear_nb:
                continue
            min_visit_d = min(
                (math.hypot(tx - vx, ty - vy) for vx, vy in visited_xy),
                default=10.0,
            )
            if min_visit_d < 2.0:
                continue
            score = target_d + 1.5 * min(min_visit_d, 8.0) - nb_cost * 0.08
            if score > best_score:
                best_score = score
                best = (tx, ty, rad)
        return best

    def mark_camera_fov_seen(
        self,
        robot_ox: float,
        robot_oy: float,
        robot_yaw: float,
        seen_cells: set[tuple[int, int]],
        *,
        cell_m: float = 0.5,
        fov_half_deg: float = 36.0,
        max_range_m: float = 8.5,
        num_rays: int = 15,
    ) -> int:
        """Ray-cast camera FOV on the local costmap and record visually observed 0.5m cells.

        Rays stop at walls/obstacles (cost >= 85) so rooms cannot mark hidden hallways behind walls as seen.
        """
        added = 0
        step_m = max(0.25, self.resolution)
        # Mark robot's immediate surroundings as seen
        rcx, rcy = int(math.floor(robot_ox / cell_m)), int(math.floor(robot_oy / cell_m))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (rcx + dx, rcy + dy)
                if key not in seen_cells:
                    seen_cells.add(key)
                    added += 1

        if num_rays <= 1:
            angles = [robot_yaw]
        else:
            angles = [
                robot_yaw + math.radians(-fov_half_deg + (2.0 * fov_half_deg) * i / (num_rays - 1))
                for i in range(num_rays)
            ]
        for ang in angles:
            ux, uy = math.cos(ang), math.sin(ang)
            d = step_m
            while d <= max_range_m:
                wx = robot_ox + ux * d
                wy = robot_oy + uy * d
                c = self.cost_at(wx, wy)
                if c is None or c < 0:
                    break
                key = (int(math.floor(wx / cell_m)), int(math.floor(wy / cell_m)))
                if key not in seen_cells:
                    seen_cells.add(key)
                    added += 1
                if c >= 85:
                    # Wall/door surface observed; cannot see behind it
                    break
                d += step_m
        return added

    def pick_frontier_explore_target(
        self,
        robot_ox: float,
        robot_oy: float,
        seen_cells: set[tuple[int, int]],
        visited_xy: list[tuple[float, float]],
        *,
        failed_xy: list[tuple[float, float]] | None = None,
        cell_m: float = 0.5,
        min_dist_m: float = 2.0,
        max_dist_m: float = 16.0,
    ) -> tuple[float, float, float, str] | None:
        """BFS-based frontier & unseen-area explorer on the local costmap (100% pure sensor).

        Flows through narrow open doorways (allowing cost < 75 along transit) to escape enclosed
        rooms where every interior wall has already been scanned, prioritizing:
        1) Unexplored frontiers (reachable free cells adjacent to unknown `-1` space)
        2) Large patches of visually UNSEEN space outside the already-scanned room.
        Returns (target_ox, target_oy, heading_rad, reason) or None.
        """
        from collections import deque

        # Downsample stride so BFS runs in a few ms (~0.4m grid)
        stride = max(1, int(round(0.4 / max(0.05, self.resolution))))
        step_m = stride * self.resolution
        gw = self.width // stride
        gh = self.height // stride
        if gw < 3 or gh < 3:
            return None

        # Coarse cost grid: max cost in each stride x stride block (but preserve unknown=-1 if all unknown)
        trimmed = self.data[: gh * stride, : gw * stride]
        blocks = trimmed.reshape(gh, stride, gw, stride)
        has_known = np.any(blocks >= 0, axis=(1, 3))
        max_known = np.max(np.where(blocks >= 0, blocks, -1), axis=(1, 3))
        coarse = np.where(has_known, max_known, -1)

        sr = int(math.floor((robot_oy - self.origin_y) / step_m))
        sc = int(math.floor((robot_ox - self.origin_x) / step_m))
        if sr < 0 or sc < 0 or sr >= gh or sc >= gw:
            return None

        # BFS through passable cells (cost < 75 lets path squeeze through 2-block doorways with wall aversion)
        max_steps = int(math.ceil(max_dist_m / step_m))
        dist_grid = np.full((gh, gw), -1, dtype=np.int32)
        dist_grid[sr, sc] = 0
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        reachable: list[tuple[int, int, int]] = []

        while q:
            r, c = q.popleft()
            d = int(dist_grid[r, c])
            if d >= max_steps:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= gh or nc >= gw or dist_grid[nr, nc] >= 0:
                    continue
                val = int(coarse[nr, nc])
                if val < 0 or val >= 75:
                    continue
                dist_grid[nr, nc] = d + 1
                q.append((nr, nc))
                reachable.append((nr, nc, d + 1))

        if not reachable:
            return None

        failed_list = failed_xy or []
        best: tuple[float, float, float, str] | None = None
        best_score = -1e9

        # Evaluate candidate cells (stride 2 on coarse grid ~= every 0.8m for speed)
        for r, c, steps in reachable:
            if (r & 1) and (c & 1):
                continue
            bfs_dist_m = steps * step_m
            if bfs_dist_m < min_dist_m:
                continue
            val = int(coarse[r, c])
            # Target itself should be comfortably clear of walls/doorframes
            if val >= 40:
                continue
            wx = self.origin_x + (c + 0.5) * step_m
            wy = self.origin_y + (r + 0.5) * step_m
            if any(math.hypot(wx - vx, wy - vy) < 2.2 for vx, vy in visited_xy):
                continue
            if any(math.hypot(wx - fx, wy - fy) < 2.5 for fx, fy in failed_list):
                continue

            # Count nearby unknown (frontier) cells and visually UNSEEN cells within ~3.5 m
            unknown_near = 0
            unseen_free = 0
            rad_c = max(2, int(round(3.2 / step_m)))
            r0, r1 = max(0, r - rad_c), min(gh - 1, r + rad_c)
            c0, c1 = max(0, c - rad_c), min(gw - 1, c + rad_c)
            for rr in range(r0, r1 + 1, 2):
                for cc in range(c0, c1 + 1, 2):
                    cv = int(coarse[rr, cc])
                    if cv < 0:
                        unknown_near += 1
                    elif cv < 75:
                        ux = self.origin_x + (cc + 0.5) * step_m
                        uy = self.origin_y + (rr + 0.5) * step_m
                        key = (int(math.floor(ux / cell_m)), int(math.floor(uy / cell_m)))
                        if key not in seen_cells:
                            unseen_free += 1

            self_key = (int(math.floor(wx / cell_m)), int(math.floor(wy / cell_m)))
            self_unseen = 1 if self_key not in seen_cells else 0
            if unknown_near == 0 and unseen_free == 0 and not self_unseen:
                continue

            # Prioritize large unexplored frontiers (outside open doorways!) and unseen open areas
            score = (
                3.5 * min(unknown_near, 18)
                + 2.2 * min(unseen_free, 20)
                + 4.0 * self_unseen
                - 0.45 * bfs_dist_m
                - 0.05 * val
            )
            if score > best_score:
                best_score = score
                heading = math.atan2(wy - robot_oy, wx - robot_ox)
                reason = "未探索区域边界" if unknown_near >= 2 else "未扫描过的新空间"
                best = (wx, wy, heading, reason)
        return best


INDOOR_SEG_CLASSES = frozenset({"corridor", "room_entry"})
CLAMP_DOOR_KINDS = frozenset({"entrance", "room_door"})


def _vertex_turn_deg(p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> float:
    ax, ay = p1[0] - p0[0], p1[1] - p0[1]
    bx, by = p2[0] - p1[0], p2[1] - p1[1]
    la, lb = math.hypot(ax, ay), math.hypot(bx, by)
    if la <= 1e-6 or lb <= 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
    return math.degrees(math.acos(cos_a))


def polyline_lookahead(
    points: list[tuple[float, float]],
    robot_xy: tuple[float, float],
    lookahead_m: float,
    *,
    point_kinds: list[str] | None = None,
    seg_classes: list[str] | None = None,
    indoor_lookahead_m: float = 5.0,
    corner_clamp_deg: float = 35.0,
) -> tuple[tuple[float, float], float, bool, int, float]:
    """Return (target_xy, target_yaw, is_final, seg_index, remaining_m).

    On indoor corridor/room_entry segments, narrows lookahead to indoor_lookahead_m
    and clamps subgoals at doorway nodes (entrance / room_door) and sharp corridor
    corners (>= corner_clamp_deg) so the robot never tries to cut through walls.
    """
    if len(points) < 2:
        pt = points[0] if points else robot_xy
        return pt, 0.0, True, 0, math.hypot(pt[0] - robot_xy[0], pt[1] - robot_xy[1])

    rx, ry = robot_xy
    best_seg = 0
    best_t = 0.0
    best_dist = float("inf")
    cum = [0.0]
    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        seg_len = math.hypot(bx - ax, by - ay)
        cum.append(cum[-1] + seg_len)
        if seg_len <= 1e-9:
            t = 0.0
            qx, qy = ax, ay
        else:
            t = max(0.0, min(1.0, ((rx - ax) * (bx - ax) + (ry - ay) * (by - ay)) / (seg_len * seg_len)))
            qx, qy = ax + t * (bx - ax), ay + t * (by - ay)
        d = math.hypot(rx - qx, ry - qy)
        if d < best_dist:
            best_dist = d
            best_seg = i
            best_t = t

    seg_len = cum[best_seg + 1] - cum[best_seg]
    s_cur = cum[best_seg] + best_t * seg_len
    total = cum[-1]
    remaining = max(0.0, total - s_cur)

    pkinds = point_kinds or []
    sclasses = seg_classes or []
    cur_indoor = best_seg < len(sclasses) and sclasses[best_seg] in INDOOR_SEG_CLASSES
    eff_lookahead = min(lookahead_m, indoor_lookahead_m) if cur_indoor else lookahead_m
    s_target = min(total, s_cur + eff_lookahead)

    # Doorway & indoor-corner clamping ahead of the robot.
    for v_idx in range(best_seg + 1, len(points) - 1):
        sv = cum[v_idx]
        if sv <= s_cur + 0.15:
            continue
        if sv >= s_target:
            break
        vx, vy = points[v_idx]
        if math.hypot(vx - rx, vy - ry) <= 1.35:
            continue
        vkind = pkinds[v_idx] if v_idx < len(pkinds) else ""
        is_door = vkind in CLAMP_DOOR_KINDS
        seg_before_in = (v_idx - 1) < len(sclasses) and sclasses[v_idx - 1] in INDOOR_SEG_CLASSES
        seg_after_in = v_idx < len(sclasses) and sclasses[v_idx] in INDOOR_SEG_CLASSES
        is_indoor_corner = (seg_before_in or seg_after_in) and (
            _vertex_turn_deg(points[v_idx - 1], points[v_idx], points[v_idx + 1]) >= corner_clamp_deg
        )
        if is_door or is_indoor_corner:
            nx, ny = points[v_idx + 1]
            yaw = math.atan2(ny - vy, nx - vx)
            return (vx, vy), yaw, False, best_seg, remaining

    is_final = s_target >= total - 0.5

    for i in range(len(points) - 1):
        if cum[i + 1] >= s_target - 1e-9 or i == len(points) - 2:
            ax, ay = points[i]
            bx, by = points[i + 1]
            sl = max(1e-9, cum[i + 1] - cum[i])
            frac = max(0.0, min(1.0, (s_target - cum[i]) / sl))
            tx = ax + frac * (bx - ax)
            ty = ay + frac * (by - ay)
            yaw = math.atan2(by - ay, bx - ax)
            return (tx, ty), yaw, is_final, best_seg, remaining

    end = points[-1]
    prev = points[-2]
    return end, math.atan2(end[1] - prev[1], end[0] - prev[0]), True, len(points) - 2, remaining


class HierarchicalNavExecutive:
    """Runs campus-scale & multi-floor navigation as rolling local Nav2 subgoals + elevator legs."""

    def __init__(
        self,
        cfg: CampusNavConfig,
        map_provider: Callable[[], CampusMap],
        fusion: RtkOdomFusion,
        planner: GlobalPlanner,
        cost_grid_provider: Callable[[], LocalCostGridSnapshot | None],
        indoor_store_provider: Callable[[], Any] | None = None,
        rpc_fn: Callable[[str, dict[str, Any], float], dict[str, Any]] | None = None,
        elevator_client: Any = None,
        reset_local_map_fn: Callable[[], bool] | None = None,
        on_floor_changed: Callable[[str, str, str], None] | None = None,
        tick_s: float = 0.25,
    ) -> None:
        self.cfg = cfg
        self._map_provider = map_provider
        self._indoor_store_provider = indoor_store_provider
        self._fusion = fusion
        self._planner = planner
        self._cost_grid_provider = cost_grid_provider
        self._rpc = rpc_fn or default_navigation_rpc
        self._elevator = elevator_client
        self._reset_local_map = reset_local_map_fn
        self._on_floor_changed = on_floor_changed
        self._tick_s = float(tick_s)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._replan_event = threading.Event()
        self._replan_reason: str = ""
        self._last_visual_replan_t: float = 0.0
        self._active_goal: dict[str, Any] | None = None
        self._allowed_leg_indoor_ids: set[str] = set()
        self._worker: threading.Thread | None = None
        self._run: dict[str, Any] = self._idle_state()

    @staticmethod
    def _idle_state() -> dict[str, Any]:
        return {
            "run_id": "",
            "state": "IDLE",
            "place": "",
            "goal_x": None,
            "goal_y": None,
            "goal_node_id": "",
            "route_points": [],
            "route_edge_ids": [],
            "subgoal_x": None,
            "subgoal_y": None,
            "subgoal_is_final": False,
            "local_run_id": "",
            "current_segment": 0,
            "total_segments": 0,
            "leg_index": 0,
            "total_legs": 0,
            "elevator_phase": "",
            "active_indoor_id": "",
            "remaining_m": 0.0,
            "reroutes": 0,
            "detail": "ready",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._run)
            out["blocked_edges"] = sorted(self._planner.active_blocked_edges())
            return out

    def start_run(
        self,
        *,
        place: str = "",
        x: float | None = None,
        y: float | None = None,
        node_id: str = "",
        indoor_id: str = "",
        start_indoor_id: str = "",
        timeout_s: float = 0.0,
    ) -> dict[str, Any]:
        campus_map = self._map_provider()
        indoor_store = self._indoor_store_provider() if self._indoor_store_provider else None
        goal_node_id = node_id.strip()
        goal_indoor_id = indoor_id.strip()
        place_label = place.strip()

        if goal_node_id and goal_indoor_id and indoor_store is not None and not place_label:
            fm = indoor_store.get(goal_indoor_id)
            n = fm.node_map().get(goal_node_id) if fm is not None else None
            if n is None:
                return {"ok": False, "accepted": False, "run_id": "", "detail": f"unknown indoor node: {goal_node_id}"}
            gx, gy = float(n["x"]), float(n["y"])
            place_label = str(n.get("label") or goal_node_id)
        elif goal_node_id and not place_label:
            n = campus_map.node_map().get(goal_node_id)
            if n is None:
                return {"ok": False, "accepted": False, "run_id": "", "detail": f"unknown node_id: {goal_node_id}"}
            gx, gy = float(n["x"]), float(n["y"])
            place_label = str(n.get("label") or goal_node_id)
        elif place_label:
            # If caller pinned an indoor floor, check indoor rooms on that floor first.
            poi = None if goal_indoor_id else campus_map.find_poi(place_label)
            if poi is not None:
                gx, gy = float(poi["x"]), float(poi["y"])
                if not goal_node_id:
                    goal_node_id = str(poi.get("node_id") or "")
                elif goal_node_id == "__door__" and poi.get("door_node_id"):
                    goal_node_id = str(poi["door_node_id"])
                    dn = campus_map.node_map().get(goal_node_id)
                    if dn is not None:
                        gx, gy = float(dn["x"]), float(dn["y"])
                place_label = poi["name"]
            elif indoor_store is not None:
                hit = indoor_store.find_room(place_label, indoor_id=goal_indoor_id)
                if hit is None:
                    return {"ok": False, "accepted": False, "run_id": "", "detail": f"unknown place: {place_label}"}
                fm, room = hit
                goal_indoor_id = fm.indoor_id
                gx, gy = float(room["x"]), float(room["y"])
                if goal_node_id == "__door__" and room.get("door_node_id"):
                    goal_node_id = str(room["door_node_id"])
                    dn = fm.node_map().get(goal_node_id)
                    if dn is not None:
                        gx, gy = float(dn["x"]), float(dn["y"])
                elif not goal_node_id:
                    goal_node_id = str(room.get("node_id") or "")
                place_label = f"{fm.building_name}·{fm.floor}·{room['name']}"
            else:
                return {"ok": False, "accepted": False, "run_id": "", "detail": f"unknown place: {place_label}"}
        elif x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            gx, gy = float(x), float(y)
            place_label = f"({gx:.1f}, {gy:.1f})"
        else:
            return {"ok": False, "accepted": False, "run_id": "", "detail": "provide place name or finite (x, y)"}

        pose = self._fusion.campus_pose()
        if pose is None:
            return {"ok": False, "accepted": False, "run_id": "", "detail": "no robot localization available yet"}

        if indoor_store is not None and (indoor_store.all_maps() or goal_indoor_id or start_indoor_id):
            route = self._planner.plan_unified(
                campus_map,
                indoor_store.all_maps(),
                (pose[0], pose[1]),
                (gx, gy),
                start_indoor_id=start_indoor_id,
                goal_indoor_id=goal_indoor_id,
                goal_node_id=goal_node_id,
            )
        else:
            route = self._planner.plan(campus_map, (pose[0], pose[1]), (gx, gy), goal_node_id=goal_node_id)
        if route is None:
            return {"ok": False, "accepted": False, "run_id": "", "detail": f"no reachable campus route to {place_label}"}

        legs = list(route.legs) if route.legs else [
            RouteLeg(
                kind="walk",
                description=place_label,
                route=route,
                indoor_id=goal_indoor_id,
                goal_xy=(gx, gy),
                goal_node_id=goal_node_id,
            )
        ]

        # Fail fast if the planned route requires an elevator that isn't configured.
        for leg in legs:
            if leg.kind == "elevator":
                if self._elevator is None:
                    return {
                        "ok": False,
                        "accepted": False,
                        "run_id": "",
                        "detail": f"路线需要乘坐电梯 ({leg.from_floor}→{leg.to_floor})，但未配置 elevator_config_file",
                    }
                if not self._elevator.supports_floor(leg.from_floor) or not self._elevator.supports_floor(leg.to_floor):
                    return {
                        "ok": False,
                        "accepted": False,
                        "run_id": "",
                        "detail": f"电梯配置不支持楼层 {leg.from_floor} → {leg.to_floor}",
                    }

        self.cancel()
        run_id = f"campus-{uuid.uuid4().hex[:8]}"
        deadline = (time.monotonic() + timeout_s) if timeout_s > 0 else 0.0
        first_walk = next((lg.route for lg in legs if lg.kind == "walk" and lg.route is not None), route)
        elev_count = sum(1 for lg in legs if lg.kind == "elevator")
        plan_note = (
            f"planned {route.total_length_m:.1f} m multi-floor route ({elev_count} elevator ride) to {place_label}"
            if elev_count > 0
            else f"planned {route.total_length_m:.1f} m route to {place_label}"
        )
        with self._lock:
            self._stop_event.clear()
            self._replan_event.clear()
            self._replan_reason = ""
            self._active_goal = {
                "place": place_label,
                "gx": gx,
                "gy": gy,
                "goal_node_id": goal_node_id,
                "goal_indoor_id": goal_indoor_id,
            }
            self._run = {
                "run_id": run_id,
                "state": "RUNNING",
                "place": place_label,
                "goal_x": round(gx, 3),
                "goal_y": round(gy, 3),
                "goal_node_id": goal_node_id,
                "route_points": [[round(p[0], 3), round(p[1], 3)] for p in first_walk.points],
                "route_edge_ids": list(first_walk.edge_ids),
                "subgoal_x": None,
                "subgoal_y": None,
                "subgoal_is_final": False,
                "local_run_id": "",
                "current_segment": 1,
                "total_segments": max(1, len(first_walk.points) - 1),
                "leg_index": 1,
                "total_legs": len(legs),
                "elevator_phase": "",
                "active_indoor_id": start_indoor_id,
                "remaining_m": round(route.total_length_m, 2),
                "reroutes": 0,
                "detail": plan_note,
            }
            worker = threading.Thread(
                target=self._run_legs,
                args=(run_id, legs, start_indoor_id, deadline),
                name=f"campus-nav-{run_id}",
                daemon=True,
            )
            self._worker = worker
            worker.start()
        log.info("started campus nav %s -> %s (%s)", run_id, place_label, plan_note)
        return {"ok": True, "accepted": True, "run_id": run_id, "detail": self._run["detail"], "navigation": self.snapshot()}

    def cancel(self, run_id: str = "") -> dict[str, Any]:
        with self._lock:
            cur_id = str(self._run.get("run_id") or "")
            if run_id and cur_id and run_id != cur_id:
                return {"ok": False, "accepted": False, "detail": f"unknown run_id {run_id}"}
            local_id = str(self._run.get("local_run_id") or "")
            was_running = self._run.get("state") in ("PENDING", "RUNNING")
            self._stop_event.set()
            worker = self._worker
            if was_running:
                self._run["state"] = "CANCELED"
                self._run["detail"] = "canceled"
        if local_id:
            self._rpc("cancel", {"run_id": local_id}, 5.0)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        return {"ok": True, "accepted": True, "detail": "canceled" if was_running else "idle"}

    def shutdown(self) -> None:
        self.cancel()

    def notify_visual_fix(
        self,
        confirmed_indoor_id: str,
        confirmed_floor: str,
        *,
        old_xy: tuple[float, float] | None,
        new_xy: tuple[float, float],
        pose_jump_threshold_m: float = 4.0,
    ) -> bool:
        """Validate an in-flight ArUco visual marker observation against the active navigation plan.

        - Normal minor drift on the expected floor: returns False (does NOT interrupt navigation).
        - Floor mismatch (e.g. planned Outdoor/1F but spotted 4F marker) or large pose jump (>= 4 m):
          signals the executive to cancel the stale leg and replan from the true floor/pose.
        """
        now = time.monotonic()
        with self._lock:
            if self._run.get("state") != "RUNNING":
                return False
            # Never abort mid-elevator physical door/cabin transport; wait until walk leg resumes.
            phase = str(self._run.get("elevator_phase") or "")
            if phase in ("CLOSING_SRC", "RIDING", "SWITCH_FLOOR", "OPENING_DST", "RESET_BEFORE_EXIT", "EXITING", "CLOSING_DST"):
                return False
            if now - self._last_visual_replan_t < 5.0:
                return False
            allowed = set(self._allowed_leg_indoor_ids)
            floor_mismatch = bool(confirmed_indoor_id and allowed and confirmed_indoor_id not in allowed)
            jump_m = (
                math.hypot(new_xy[0] - old_xy[0], new_xy[1] - old_xy[1])
                if old_xy is not None
                else 0.0
            )
            if not floor_mismatch and jump_m < pose_jump_threshold_m:
                return False
            reasons: list[str] = []
            if floor_mismatch:
                reasons.append(f"楼层不符(实测 {confirmed_floor})")
            if jump_m >= pose_jump_threshold_m:
                reasons.append(f"位置偏差 {jump_m:.1f}m")
            reason = "、".join(reasons)
            self._last_visual_replan_t = now
            self._replan_reason = reason
            self._replan_event.set()
            self._run["detail"] = f"标牌校验发现{reason}，正在基于真实楼层重新规划路线..."
            log.warning(
                "in-flight visual check mismatch (%s, confirmed=%s, allowed=%s) -> triggering replan",
                reason,
                confirmed_indoor_id,
                sorted(allowed),
            )
            return True

    @staticmethod
    def _leg_allowed_indoor_ids(leg: RouteLeg, start_indoor_id: str) -> set[str]:
        allowed: set[str] = {start_indoor_id}
        if leg.kind == "walk":
            allowed.add(leg.indoor_id)
            if leg.route is not None:
                for nid in leg.route.node_ids:
                    if nid.startswith("in:"):
                        parts = nid.split(":", 2)
                        if len(parts) == 3:
                            allowed.add(parts[1])
                    elif nid.startswith("out:"):
                        allowed.add("")
        elif leg.kind == "elevator":
            allowed.add(leg.from_indoor_id)
            allowed.add(leg.to_indoor_id)
        return allowed

    def _send_local_subgoal(
        self, cx: float, cy: float, yaw: float, *, nudge_radius_m: float = 3.0
    ) -> tuple[bool, str, str]:
        ox, oy = self._fusion.campus_to_odom(cx, cy)
        grid = self._cost_grid_provider()
        if grid is not None and nudge_radius_m > 0.0:
            ox, oy = grid.nudge_to_passable(ox, oy, search_radius_m=nudge_radius_m)
        half = yaw / 2.0
        resp = self._rpc(
            "navigate",
            {
                "goal": {
                    "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "minecraft_odom"},
                    "pose": {
                        "position": {"x": float(ox), "y": float(oy), "z": 0.0},
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": math.sin(half),
                            "w": math.cos(half),
                        },
                    },
                }
            },
            10.0,
        )
        accepted = bool(resp.get("accepted", False))
        rid = str(resp.get("run_id") or "")
        detail = str(resp.get("detail") or "")
        return accepted and bool(rid), rid, detail

    def _run_legs(
        self,
        run_id: str,
        legs: list[RouteLeg],
        initial_indoor_id: str,
        deadline: float,
    ) -> None:
        cur_legs = list(legs)
        cur_indoor_id = initial_indoor_id
        visual_replans = 0
        max_visual_replans = 4

        while not self._stop_event.is_set():
            total_legs = len(cur_legs)
            need_replan = False
            for idx, leg in enumerate(cur_legs):
                if self._stop_event.is_set():
                    return
                with self._lock:
                    self._run["leg_index"] = idx + 1
                    self._run["total_legs"] = total_legs
                    self._run["elevator_phase"] = ""
                    self._allowed_leg_indoor_ids = self._leg_allowed_indoor_ids(leg, cur_indoor_id)
                is_last_leg = idx == total_legs - 1
                if leg.kind == "walk":
                    if leg.route is None or leg.goal_xy is None:
                        continue
                    ok, fail_msg = self._execute_walk_leg(
                        run_id,
                        leg,
                        cur_indoor_id,
                        deadline,
                        is_last_leg=is_last_leg,
                        leg_prefix=f"[{idx + 1}/{total_legs}] " if total_legs > 1 else "",
                    )
                    if not ok:
                        if fail_msg == "__VISUAL_REPLAN__":
                            need_replan = True
                            break
                        if not self._stop_event.is_set():
                            self._finish(run_id, "FAILED", fail_msg, "")
                        return
                    if leg.indoor_id:
                        cur_indoor_id = leg.indoor_id
                elif leg.kind == "elevator":
                    ok, fail_msg = self._execute_elevator_leg(
                        run_id,
                        leg,
                        deadline,
                        leg_prefix=f"[{idx + 1}/{total_legs}] ",
                    )
                    if not ok:
                        if fail_msg == "__VISUAL_REPLAN__":
                            need_replan = True
                            break
                        if not self._stop_event.is_set():
                            self._finish(run_id, "FAILED", fail_msg, "")
                        return
                    cur_indoor_id = leg.to_indoor_id

            if not need_replan:
                break

            visual_replans += 1
            if visual_replans > max_visual_replans:
                self._finish(run_id, "FAILED", "视觉重规划次数超限", "")
                return

            with self._lock:
                self._replan_event.clear()
                reason = self._replan_reason
                goal = dict(self._active_goal or {})
            pose = self._fusion.campus_pose()
            if pose is None or not goal:
                self._finish(run_id, "FAILED", f"重规划失败：当前无有效位姿 ({reason})", "")
                return
            new_start_iid = self._fusion.current_indoor_id
            indoor_store = self._indoor_store_provider() if self._indoor_store_provider else None
            if indoor_store is not None and (indoor_store.all_maps() or goal["goal_indoor_id"] or new_start_iid):
                new_route = self._planner.plan_unified(
                    self._map_provider(),
                    indoor_store.all_maps(),
                    (pose[0], pose[1]),
                    (goal["gx"], goal["gy"]),
                    start_indoor_id=new_start_iid,
                    goal_indoor_id=goal["goal_indoor_id"],
                    goal_node_id=goal["goal_node_id"],
                )
            else:
                new_route = self._planner.plan(
                    self._map_provider(),
                    (pose[0], pose[1]),
                    (goal["gx"], goal["gy"]),
                    goal_node_id=goal["goal_node_id"],
                )
            if new_route is None:
                self._finish(run_id, "FAILED", f"基于新定位({new_start_iid or '室外'})重规划无可达路径", "")
                return

            cur_legs = list(new_route.legs) if new_route.legs else [
                RouteLeg(
                    kind="walk",
                    description=goal["place"],
                    route=new_route,
                    indoor_id=goal["goal_indoor_id"],
                    goal_xy=(goal["gx"], goal["gy"]),
                    goal_node_id=goal["goal_node_id"],
                )
            ]
            cur_indoor_id = new_start_iid
            if self._on_floor_changed is not None and new_start_iid:
                self._on_floor_changed(
                    new_start_iid,
                    self._fusion.current_building_name,
                    self._fusion.current_floor,
                )
            first_walk = next((lg.route for lg in cur_legs if lg.kind == "walk" and lg.route is not None), new_route)
            with self._lock:
                self._run["reroutes"] = int(self._run.get("reroutes", 0)) + 1
                self._run["route_points"] = [[round(p[0], 3), round(p[1], 3)] for p in first_walk.points]
                self._run["route_edge_ids"] = list(first_walk.edge_ids)
                self._run["total_segments"] = max(1, len(first_walk.points) - 1)
                self._run["total_legs"] = len(cur_legs)
                self._run["leg_index"] = 1
                self._run["active_indoor_id"] = new_start_iid
                self._run["remaining_m"] = round(new_route.total_length_m, 2)
                self._run["detail"] = f"已根据标牌校验({reason})重新规划路线 ({new_route.total_length_m:.1f} m)"
            log.info(
                "campus nav %s replanned from %s (%s) -> %d leg(s), %.1fm",
                run_id,
                new_start_iid or "outdoor",
                reason,
                len(cur_legs),
                new_route.total_length_m,
            )

        if not self._stop_event.is_set():
            self._finish(run_id, "SUCCEEDED", f"arrived at {self._run['place']}", "")

    def _execute_walk_leg(
        self,
        run_id: str,
        leg: RouteLeg,
        start_indoor_id: str,
        deadline: float,
        *,
        is_last_leg: bool,
        leg_prefix: str = "",
    ) -> tuple[bool, str]:
        assert leg.route is not None and leg.goal_xy is not None
        active_route = leg.route
        gx, gy = leg.goal_xy
        goal_node_id = leg.goal_node_id
        goal_indoor_id = leg.indoor_id
        local_run_id = ""
        subgoal_xy: tuple[float, float] | None = None
        subgoal_is_final = False
        best_dist_to_subgoal = float("inf")
        last_progress_time = time.monotonic()
        consecutive_failures = 0
        max_reroutes = 4

        with self._lock:
            self._run["route_points"] = [[round(p[0], 3), round(p[1], 3)] for p in active_route.points]
            self._run["route_edge_ids"] = list(active_route.edge_ids)
            self._run["total_segments"] = max(1, len(active_route.points) - 1)
            self._run["current_segment"] = 1
            self._run["active_indoor_id"] = goal_indoor_id or start_indoor_id

        while not self._stop_event.is_set():
            if self._replan_event.is_set():
                if local_run_id:
                    self._rpc("cancel", {"run_id": local_run_id}, 5.0)
                    with self._lock:
                        self._run["local_run_id"] = ""
                return False, "__VISUAL_REPLAN__"
            now = time.monotonic()
            if deadline > 0 and now >= deadline:
                if local_run_id:
                    self._rpc("cancel", {"run_id": local_run_id}, 5.0)
                return False, "navigation timeout"

            pose = self._fusion.campus_pose()
            if pose is None:
                self._stop_event.wait(self._tick_s)
                continue
            rx, ry, _ = pose
            dist_to_final = math.hypot(gx - rx, gy - ry)

            target_xy, target_yaw, is_final, seg_idx, remaining = polyline_lookahead(
                active_route.points,
                (rx, ry),
                self.cfg.lookahead_m,
                point_kinds=active_route.point_kinds,
                seg_classes=active_route.seg_classes,
                indoor_lookahead_m=self.cfg.indoor_lookahead_m,
                corner_clamp_deg=self.cfg.corner_clamp_deg,
            )
            seg_indoor = (
                seg_idx < len(active_route.seg_classes)
                and active_route.seg_classes[seg_idx] in INDOOR_SEG_CLASSES
            )
            eff_advance_r = (
                self.cfg.indoor_advance_radius_m if seg_indoor else self.cfg.advance_radius_m
            )
            final_indoor = bool(
                active_route.point_kinds
                and active_route.point_kinds[-1] in ("room_nav", "room_door", "corridor")
            ) or (
                bool(active_route.seg_classes)
                and active_route.seg_classes[-1] in INDOOR_SEG_CLASSES
            )
            eff_goal_tol = min(self.cfg.goal_tolerance_m, 1.2) if (final_indoor or not is_last_leg) else self.cfg.goal_tolerance_m
            with self._lock:
                self._run["remaining_m"] = round(max(remaining, dist_to_final), 2)
                self._run["current_segment"] = min(
                    seg_idx + 1, max(1, len(active_route.points) - 1)
                )

            if dist_to_final <= eff_goal_tol:
                if local_run_id:
                    self._rpc("cancel", {"run_id": local_run_id}, 5.0)
                    with self._lock:
                        self._run["local_run_id"] = ""
                return True, ""

            need_new_subgoal = local_run_id == ""
            subgoal_failed = False
            fail_reason = ""

            if local_run_id and subgoal_xy is not None:
                dist_sub = math.hypot(subgoal_xy[0] - rx, subgoal_xy[1] - ry)
                if dist_sub + 0.35 < best_dist_to_subgoal:
                    best_dist_to_subgoal = dist_sub
                    last_progress_time = now

                if not subgoal_is_final and dist_sub <= eff_advance_r:
                    self._rpc("cancel", {"run_id": local_run_id}, 5.0)
                    local_run_id = ""
                    consecutive_failures = 0
                    need_new_subgoal = True
                else:
                    st = self._rpc("status", {"run_id": local_run_id}, 5.0)
                    lstate = str(st.get("state") or "RUNNING").upper()
                    if lstate == "SUCCEEDED":
                        local_run_id = ""
                        consecutive_failures = 0
                        if subgoal_is_final and math.hypot(gx - rx, gy - ry) <= eff_goal_tol + 0.8:
                            with self._lock:
                                self._run["local_run_id"] = ""
                            return True, ""
                        need_new_subgoal = True
                    elif lstate in ("FAILED", "CANCELED", "CANCELLED"):
                        local_run_id = ""
                        subgoal_failed = True
                        fail_reason = str(st.get("detail") or f"local nav {lstate.lower()}")
                    elif now - last_progress_time >= self.cfg.subgoal_timeout_s:
                        self._rpc("cancel", {"run_id": local_run_id}, 5.0)
                        local_run_id = ""
                        subgoal_failed = True
                        fail_reason = "local subgoal stalled"

            if subgoal_failed:
                consecutive_failures += 1
                log.warning("campus nav %s subgoal failed (%d): %s", run_id, consecutive_failures, fail_reason)
                if consecutive_failures >= 2:
                    blocked_eid = self._current_edge_id(active_route, seg_idx)
                    if blocked_eid:
                        self._planner.mark_edge_blocked(blocked_eid)
                    with self._lock:
                        reroutes = int(self._run["reroutes"]) + 1
                    if reroutes > max_reroutes:
                        return False, f"blocked after {max_reroutes} reroutes: {fail_reason}"
                    indoor_store = self._indoor_store_provider() if self._indoor_store_provider else None
                    if indoor_store is not None and (indoor_store.all_maps() or goal_indoor_id or start_indoor_id):
                        new_route = self._planner.plan_unified(
                            self._map_provider(),
                            indoor_store.all_maps(),
                            (rx, ry),
                            (gx, gy),
                            start_indoor_id=start_indoor_id,
                            goal_indoor_id=goal_indoor_id,
                            goal_node_id=goal_node_id,
                        )
                    else:
                        new_route = self._planner.plan(
                            self._map_provider(), (rx, ry), (gx, gy), goal_node_id=goal_node_id
                        )
                    if new_route is None:
                        return False, f"no alternate campus path ({fail_reason})"
                    # Pick the walk sub-route if plan_unified returned legs
                    if new_route.legs and new_route.legs[0].kind == "walk" and new_route.legs[0].route is not None:
                        active_route = new_route.legs[0].route
                    else:
                        active_route = new_route
                    consecutive_failures = 0
                    with self._lock:
                        self._run["reroutes"] = reroutes
                        self._run["route_points"] = [[round(p[0], 3), round(p[1], 3)] for p in active_route.points]
                        self._run["route_edge_ids"] = list(active_route.edge_ids)
                        self._run["total_segments"] = max(1, len(active_route.points) - 1)
                        self._run["detail"] = f"{leg_prefix}rerouted around blocked edge ({reroutes})"
                    target_xy, target_yaw, is_final, seg_idx, remaining = polyline_lookahead(
                        active_route.points,
                        (rx, ry),
                        self.cfg.lookahead_m,
                        point_kinds=active_route.point_kinds,
                        seg_classes=active_route.seg_classes,
                        indoor_lookahead_m=self.cfg.indoor_lookahead_m,
                        corner_clamp_deg=self.cfg.corner_clamp_deg,
                    )
                need_new_subgoal = True

            if need_new_subgoal and not self._stop_event.is_set():
                ok, rid, detail = self._send_local_subgoal(target_xy[0], target_xy[1], target_yaw)
                if ok:
                    local_run_id = rid
                    subgoal_xy = target_xy
                    subgoal_is_final = is_final
                    best_dist_to_subgoal = math.hypot(target_xy[0] - rx, target_xy[1] - ry)
                    last_progress_time = now
                    with self._lock:
                        self._run["local_run_id"] = rid
                        self._run["subgoal_x"] = round(target_xy[0], 3)
                        self._run["subgoal_y"] = round(target_xy[1], 3)
                        self._run["subgoal_is_final"] = is_final
                        self._run["detail"] = (
                            f"{leg_prefix}approaching {leg.description or self._run['place']}"
                            if is_final
                            else f"{leg_prefix}segment {seg_idx + 1}/{max(1, len(active_route.points) - 1)} ({remaining:.1f} m left)"
                        )
                else:
                    consecutive_failures += 1
                    with self._lock:
                        self._run["detail"] = f"{leg_prefix}waiting for local nav: {detail}"
                    if consecutive_failures >= 4:
                        return False, f"local nav unavailable: {detail}"

            self._stop_event.wait(self._tick_s)

        if local_run_id:
            self._rpc("cancel", {"run_id": local_run_id}, 5.0)
        return False, "canceled"

    def _navigate_to_exact_point(
        self,
        target_xy: tuple[float, float],
        target_yaw: float,
        deadline: float,
        *,
        tol_m: float = 0.65,
        align_first: bool = True,
        nudge_radius_m: float = 0.0,
        step_timeout_s: float = 45.0,
        verify_fn: Callable[[float, float], bool] | None = None,
        settle_s: float = 0.35,
    ) -> tuple[bool, str]:
        """Drive precisely into or out of an elevator cabin using local Nav2.

        If `verify_fn` is provided, requires `verify_fn(rx, ry)` to stay True continuously
        for `settle_s` seconds before reporting arrival (preventing premature door closure
        while the robot is still straddling the doorway).
        """
        tx, ty = float(target_xy[0]), float(target_xy[1])
        pose = self._fusion.campus_pose()
        if pose is None:
            return False, "no robot pose during elevator maneuver"
        rx, ry, ryaw = pose

        def _is_satisfied(px: float, py: float, *, allow_loose: bool = False) -> bool:
            if verify_fn is not None:
                return bool(verify_fn(px, py))
            return math.hypot(tx - px, ty - py) <= (tol_m + 0.35 if allow_loose else tol_m)

        # First rotate in place toward target if facing away (prevents forward-only
        # Nav2 controller oscillation inside the tight 4x5 block elevator cabin).
        if align_first and math.hypot(tx - rx, ty - ry) > 0.75:
            desired_heading = math.atan2(ty - ry, tx - rx)
            if abs(_wrap_angle(desired_heading - ryaw)) > math.radians(55.0):
                ok_rot, rot_id, _ = self._send_local_subgoal(rx, ry, desired_heading, nudge_radius_m=0.0)
                if ok_rot:
                    with self._lock:
                        self._run["local_run_id"] = rot_id
                    rot_deadline = min(deadline, time.monotonic() + 10.0) if deadline > 0 else time.monotonic() + 10.0
                    while not self._stop_event.is_set() and time.monotonic() < rot_deadline:
                        p_now = self._fusion.campus_pose()
                        if p_now is not None and abs(_wrap_angle(desired_heading - p_now[2])) <= math.radians(25.0):
                            break
                        st = self._rpc("status", {"run_id": rot_id}, 5.0)
                        if str(st.get("state") or "").upper() in NAV_TERMINAL:
                            break
                        self._stop_event.wait(self._tick_s)
                    self._rpc("cancel", {"run_id": rot_id}, 5.0)
                    with self._lock:
                        self._run["local_run_id"] = ""

        step_deadline = time.monotonic() + step_timeout_s
        if deadline > 0:
            step_deadline = min(step_deadline, deadline)

        attempts = 0
        satisfied_since: float | None = None
        while not self._stop_event.is_set() and time.monotonic() < step_deadline:
            pose = self._fusion.campus_pose()
            if pose is not None and _is_satisfied(pose[0], pose[1]):
                if settle_s <= 0.0:
                    return True, ""
                t0 = time.monotonic()
                while not self._stop_event.is_set() and (time.monotonic() - t0) < settle_s:
                    p_chk = self._fusion.campus_pose()
                    if p_chk is None or not _is_satisfied(p_chk[0], p_chk[1]):
                        break
                    self._stop_event.wait(min(0.1, self._tick_s))
                else:
                    return True, ""

            ok, rid, detail = self._send_local_subgoal(tx, ty, target_yaw, nudge_radius_m=nudge_radius_m)
            if not ok:
                attempts += 1
                if attempts >= 4:
                    return False, f"elevator maneuver nav rejected: {detail}"
                self._stop_event.wait(self._tick_s)
                continue
            with self._lock:
                self._run["local_run_id"] = rid
                self._run["subgoal_x"] = round(tx, 3)
                self._run["subgoal_y"] = round(ty, 3)
                self._run["subgoal_is_final"] = True
            satisfied_since = None
            try:
                while not self._stop_event.is_set() and time.monotonic() < step_deadline:
                    now = time.monotonic()
                    pose = self._fusion.campus_pose()
                    if pose is not None and _is_satisfied(pose[0], pose[1]):
                        if satisfied_since is None:
                            satisfied_since = now
                        if now - satisfied_since >= settle_s:
                            return True, ""
                    else:
                        satisfied_since = None
                    st = self._rpc("status", {"run_id": rid}, 5.0)
                    lstate = str(st.get("state") or "RUNNING").upper()
                    if lstate == "SUCCEEDED":
                        pose = self._fusion.campus_pose()
                        if pose is not None and _is_satisfied(pose[0], pose[1], allow_loose=True):
                            if settle_s > 0.0:
                                self._stop_event.wait(settle_s)
                                pose = self._fusion.campus_pose()
                            if pose is not None and _is_satisfied(pose[0], pose[1], allow_loose=True):
                                return True, ""
                        # Nav2 stopped short before passing strict cabin check -> retry deeper!
                        attempts += 1
                        log.warning(
                            "elevator maneuver stopped short at %s (target=%s, attempt=%d)",
                            (round(pose[0], 2), round(pose[1], 2)) if pose else None,
                            (round(tx, 2), round(ty, 2)),
                            attempts,
                        )
                        if attempts >= 4:
                            return False, "机器人未能完全进入/离开电梯安全区域"
                        break
                    if lstate in ("FAILED", "CANCELED", "CANCELLED"):
                        attempts += 1
                        if attempts >= 4:
                            return False, str(st.get("detail") or f"elevator maneuver {lstate.lower()}")
                        break
                    self._stop_event.wait(self._tick_s)
            finally:
                self._rpc("cancel", {"run_id": rid}, 5.0)
                with self._lock:
                    self._run["local_run_id"] = ""
        if self._stop_event.is_set():
            return False, "canceled"
        return False, "elevator maneuver timed out"

    def _set_elev_phase(self, phase: str, detail: str, *, indoor_id: str | None = None) -> None:
        with self._lock:
            self._run["elevator_phase"] = phase
            self._run["detail"] = detail
            if indoor_id is not None:
                self._run["active_indoor_id"] = indoor_id
        log.info("elevator [%s]: %s", phase, detail)

    def _reset_and_settle_local_map(self) -> None:
        if self._reset_local_map is not None:
            try:
                self._reset_local_map()
            except Exception as exc:  # noqa: BLE001
                log.warning("local_mapping reset failed during elevator transition: %s", exc)
        if self.cfg.elevator_settle_s > 0.0:
            self._stop_event.wait(self.cfg.elevator_settle_s)

    def _execute_elevator_leg(
        self,
        run_id: str,
        leg: RouteLeg,
        deadline: float,
        *,
        leg_prefix: str = "",
    ) -> tuple[bool, str]:
        if self._elevator is None:
            return False, "elevator client is not configured"
        if leg.from_door_xy is None or leg.from_cabin_xy is None or leg.to_door_xy is None or leg.to_cabin_xy is None:
            return False, "elevator leg missing door/cabin coordinates"

        def _rem_timeout() -> float:
            if deadline <= 0:
                return float(getattr(self._elevator, "wait_timeout_s", 90.0))
            return max(5.0, deadline - time.monotonic())

        try:
            if self._replan_event.is_set():
                return False, "__VISUAL_REPLAN__"
            # 1. Call elevator to source floor and open both landing & cabin doors.
            self._set_elev_phase(
                "CALLING",
                f"{leg_prefix}正在呼叫电梯到 {leg.from_floor} 并开门...",
                indoor_id=leg.from_indoor_id,
            )
            self._elevator.call_and_open(leg.from_floor, cancel_event=self._stop_event, timeout_s=_rem_timeout())
            if self._stop_event.is_set():
                return False, "canceled"
            if self._replan_event.is_set():
                return False, "__VISUAL_REPLAN__"

            # 2. Clear stale closed-door obstacles in rolling local map so Nav2 sees open doorway.
            self._set_elev_phase(
                "RESET_BEFORE_BOARD",
                f"{leg_prefix}{leg.from_floor} 电梯门已开，刷新局部地图...",
                indoor_id=leg.from_indoor_id,
            )
            self._reset_and_settle_local_map()
            if self._stop_event.is_set():
                return False, "canceled"

            # 3. Drive from waiting area (room_door) deep into elevator cabin (clear of glass door).
            self._set_elev_phase(
                "BOARDING",
                f"{leg_prefix}正在进入 {leg.from_floor} 电梯轿厢...",
                indoor_id=leg.from_indoor_id,
            )
            cabin_target_xy = (
                self._elevator.safe_cabin_interior_xy(leg.from_floor)
                if hasattr(self._elevator, "safe_cabin_interior_xy")
                else None
            ) or leg.from_cabin_xy
            yaw_in = math.atan2(
                cabin_target_xy[1] - leg.from_door_xy[1],
                cabin_target_xy[0] - leg.from_door_xy[0],
            )
            inside_check = (
                (lambda px, py: self._elevator.is_xy_safely_inside_cabin(leg.from_floor, px, py, margin_m=0.45))
                if hasattr(self._elevator, "is_xy_safely_inside_cabin")
                else None
            )
            ok_in, err_in = self._navigate_to_exact_point(
                cabin_target_xy,
                yaw_in,
                deadline,
                tol_m=0.55,
                align_first=True,
                nudge_radius_m=0.0,
                verify_fn=inside_check,
                settle_s=min(0.4, self.cfg.elevator_settle_s),
            )
            if not ok_in:
                return False, f"进入电梯失败（门保持打开，未启动轿厢）: {err_in}"

            # 4. Close source landing & cabin doors.
            self._set_elev_phase(
                "CLOSING_SRC",
                f"{leg_prefix}已完全进入轿厢，正在关闭 {leg.from_floor} 电梯门...",
                indoor_id=leg.from_indoor_id,
            )
            self._elevator.close_door(leg.from_floor, cancel_event=self._stop_event, timeout_s=_rem_timeout())
            if self._stop_event.is_set():
                return False, "canceled"

            # 5. Dispatch physical moving cabin to target floor and wait for settle.
            self._set_elev_phase(
                "RIDING",
                f"{leg_prefix}乘梯运行中：{leg.from_floor} → {leg.to_floor}...",
                indoor_id=leg.from_indoor_id,
            )
            self._elevator.move_cabin(leg.to_floor, cancel_event=self._stop_event, timeout_s=_rem_timeout())
            if self._stop_event.is_set():
                return False, "canceled"

            # Pure-sensor sanity check: verify onboard chassis odom Z actually reached target floor elevation!
            target_elev_y = (
                self._elevator.floor_elevation_y(leg.to_floor)
                if hasattr(self._elevator, "floor_elevation_y")
                else None
            )
            if target_elev_y is not None and self._fusion.odom_z is not None:
                if abs(self._fusion.odom_z - target_elev_y) > 2.2:
                    return (
                        False,
                        f"机器人未随轿厢抵达 {leg.to_floor} (odom_z={self._fusion.odom_z:.2f}, 期望={target_elev_y:.2f})",
                    )

            # 6. Floor switch: wipe old floor's rolling local map and update active floor state!
            self._set_elev_phase(
                "SWITCH_FLOOR",
                f"{leg_prefix}已抵达 {leg.to_floor}，清空旧楼层局部地图并切换楼层...",
                indoor_id=leg.to_indoor_id,
            )
            if self._reset_local_map is not None:
                try:
                    self._reset_local_map()
                except Exception as exc:  # noqa: BLE001
                    log.warning("local_mapping floor-switch reset failed: %s", exc)
            if self._on_floor_changed is not None:
                self._on_floor_changed(leg.to_indoor_id, leg.building_name, leg.to_floor)
            else:
                self._fusion.set_indoor_floor(leg.to_indoor_id, leg.building_name, leg.to_floor)

            # 7. Open destination floor landing & cabin doors.
            self._set_elev_phase(
                "OPENING_DST",
                f"{leg_prefix}正在打开 {leg.to_floor} 电梯门...",
                indoor_id=leg.to_indoor_id,
            )
            self._elevator.call_and_open(leg.to_floor, cancel_event=self._stop_event, timeout_s=_rem_timeout())
            if self._stop_event.is_set():
                return False, "canceled"

            # 8. Refresh local map so live LiDAR scans the open target floor hallway cleanly.
            self._set_elev_phase(
                "RESET_BEFORE_EXIT",
                f"{leg_prefix}{leg.to_floor} 电梯门已开，刷新目标楼层局部地图...",
                indoor_id=leg.to_indoor_id,
            )
            self._reset_and_settle_local_map()
            if self._stop_event.is_set():
                return False, "canceled"

            # 9. Rotate toward doorway and exit elevator cabin to target floor waiting point.
            self._set_elev_phase(
                "EXITING",
                f"{leg_prefix}正在走出电梯到 {leg.to_floor} 候梯厅...",
                indoor_id=leg.to_indoor_id,
            )
            yaw_out = math.atan2(
                leg.to_door_xy[1] - leg.to_cabin_xy[1],
                leg.to_door_xy[0] - leg.to_cabin_xy[0],
            )
            outside_check = (
                (
                    lambda px, py: self._elevator.is_xy_safely_outside_cabin(leg.to_floor, px, py, margin_m=0.45)
                    and math.hypot(px - leg.to_door_xy[0], py - leg.to_door_xy[1]) <= 3.5
                )
                if hasattr(self._elevator, "is_xy_safely_outside_cabin")
                else None
            )
            ok_out, err_out = self._navigate_to_exact_point(
                leg.to_door_xy,
                yaw_out,
                deadline,
                tol_m=1.0,
                align_first=True,
                nudge_radius_m=0.5,
                verify_fn=outside_check,
                settle_s=min(0.3, self.cfg.elevator_settle_s),
            )
            if not ok_out:
                return False, f"走出电梯失败: {err_out}"

            # 10. Close destination floor doors after exiting.
            self._set_elev_phase(
                "CLOSING_DST",
                f"{leg_prefix}已离开轿厢，正在关闭 {leg.to_floor} 电梯门...",
                indoor_id=leg.to_indoor_id,
            )
            self._elevator.close_door(leg.to_floor, cancel_event=self._stop_event, timeout_s=_rem_timeout())
            return True, ""
        except Exception as exc:  # noqa: BLE001
            if self._stop_event.is_set():
                return False, "canceled"
            log.exception("campus nav %s elevator leg failed", run_id)
            return False, f"电梯乘梯异常 ({leg.from_floor}→{leg.to_floor}): {exc}"

    @staticmethod
    def _current_edge_id(route: GlobalRoute, seg_idx: int) -> str:
        if not route.edge_ids:
            return ""
        idx = max(0, min(seg_idx, len(route.edge_ids) - 1))
        return route.edge_ids[idx]

    def _finish(self, run_id: str, state: str, detail: str, local_run_id: str) -> None:
        if local_run_id:
            self._rpc("cancel", {"run_id": local_run_id}, 5.0)
        with self._lock:
            if self._run.get("run_id") == run_id and self._run.get("state") not in NAV_TERMINAL:
                self._run["state"] = state
                self._run["detail"] = detail
                self._run["local_run_id"] = ""
                self._run["elevator_phase"] = ""
        log.info("campus nav %s finished: %s (%s)", run_id, state, detail)
