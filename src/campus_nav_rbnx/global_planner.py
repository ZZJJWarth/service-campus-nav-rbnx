# SPDX-License-Identifier: MulanPSL-2.0
"""Global A* planner on the campus road graph with dynamic blocked-edge rerouting
and multi-floor elevator leg decomposition.
"""
from __future__ import annotations

import heapq
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from campus_nav_rbnx.campus_map import CampusMap

ELEVATOR_KEYWORDS = ("电梯", "elevator", "lift")


@dataclass(frozen=True)
class RouteLeg:
    """One atomic stage in a multi-floor campus navigation plan."""

    kind: str  # "walk" | "elevator"
    description: str = ""
    # Walk leg fields
    route: "GlobalRoute | None" = None
    indoor_id: str = ""
    floor: str = ""
    goal_xy: tuple[float, float] | None = None
    goal_node_id: str = ""
    # Elevator leg fields
    elevator_id: str = ""
    building_id: str = ""
    building_name: str = ""
    from_indoor_id: str = ""
    from_floor: str = ""
    from_floor_label: str = ""
    from_door_xy: tuple[float, float] | None = None
    from_cabin_xy: tuple[float, float] | None = None
    from_door_node_id: str = ""
    from_cabin_node_id: str = ""
    from_cabin_bounds_enu: tuple[float, float, float, float] | None = None
    from_door_bounds_enu: tuple[float, float, float, float] | None = None
    from_elevation_y: float | None = None
    to_indoor_id: str = ""
    to_floor: str = ""
    to_floor_label: str = ""
    to_door_xy: tuple[float, float] | None = None
    to_cabin_xy: tuple[float, float] | None = None
    to_door_node_id: str = ""
    to_cabin_node_id: str = ""
    to_cabin_bounds_enu: tuple[float, float, float, float] | None = None
    to_door_bounds_enu: tuple[float, float, float, float] | None = None
    to_elevation_y: float | None = None


@dataclass(frozen=True)
class GlobalRoute:
    points: list[tuple[float, float]]
    node_ids: list[str]
    edge_ids: list[str]
    total_length_m: float
    point_kinds: list[str] = field(default_factory=list)
    seg_classes: list[str] = field(default_factory=list)
    legs: list[RouteLeg] = field(default_factory=list)


def _project_point_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float, float]:
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-9:
        return ax, ay, math.hypot(px - ax, py - ay), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    qx, qy = ax + t * dx, ay + t * dy
    return qx, qy, math.hypot(px - qx, py - qy), t


def _floor_number(floor_label: str) -> int:
    s = str(floor_label or "").strip().upper()
    m = re.search(r"B(\d+)", s)
    if m:
        return -int(m.group(1))
    m = re.search(r"(-?\d+)", s)
    if m:
        return int(m.group(1))
    return 1


def _to_box4(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    return None


def find_elevator_bindings(floor_map: Any) -> list[dict[str, Any]]:
    """Locate all elevator shafts on an IndoorFloorMap.

    Priority:
    1. Dedicated `floor_map.elevators` entries (bound via indoor_mapper right-click or 8093 UI).
    2. Legacy fallback: rooms named '电梯'/'elevator'/'lift' or standalone `kind == 'elevator'` nodes.
    """
    nodes = floor_map.node_map()
    fm_elev_y = getattr(floor_map, "elevation_y", None)
    fm_floor = str(getattr(floor_map, "floor", "") or "")
    out: list[dict[str, Any]] = []

    for el in getattr(floor_map, "elevators", []) or []:
        if not isinstance(el, dict):
            continue
        cabin_nid = str(el.get("cabin_node_id") or "")
        door_nid = str(el.get("door_node_id") or "")
        cabin_node = nodes.get(cabin_nid)
        door_node = nodes.get(door_nid)
        if cabin_node is None and door_node is None:
            continue
        if cabin_node is None:
            cabin_node = door_node
            cabin_nid = door_nid
        if door_node is None:
            door_node = cabin_node
            door_nid = cabin_nid
        elev_y = el.get("elevation_y")
        out.append({
            "elevator": el,
            "room": None,
            "elevator_id": str(el.get("elevator_id") or "").strip(),
            "name": str(el.get("name") or "电梯"),
            "floor_label": str(el.get("floor_label") or fm_floor),
            "door_node_id": door_nid,
            "cabin_node_id": cabin_nid,
            "door_xy": (float(door_node["x"]), float(door_node["y"])),
            "cabin_xy": (float(cabin_node["x"]), float(cabin_node["y"])),
            "cabin_bounds_enu": _to_box4(el.get("cabin_bounds_enu")),
            "door_bounds_enu": _to_box4(el.get("door_bounds_enu")),
            "elevation_y": float(elev_y) if elev_y is not None else (float(fm_elev_y) if fm_elev_y is not None else None),
        })
    if out:
        return out

    # Legacy fallback: room named 电梯 / elevator / lift
    for r in getattr(floor_map, "rooms", []) or []:
        names = [str(r.get("name") or ""), *(str(a) for a in (r.get("aliases") or []))]
        if any(kw in n.lower() for n in names for kw in ELEVATOR_KEYWORDS):
            cabin_nid = str(r.get("node_id") or "")
            door_nid = str(r.get("door_node_id") or "")
            cabin_node = nodes.get(cabin_nid)
            door_node = nodes.get(door_nid)
            if cabin_node is None and door_node is None:
                continue
            if cabin_node is None:
                cabin_node = door_node
                cabin_nid = door_nid
            if door_node is None:
                door_node = cabin_node
                door_nid = cabin_nid
            out.append({
                "elevator": None,
                "room": r,
                "elevator_id": "",
                "name": str(r.get("name") or "电梯"),
                "floor_label": fm_floor,
                "door_node_id": door_nid,
                "cabin_node_id": cabin_nid,
                "door_xy": (float(door_node["x"]), float(door_node["y"])),
                "cabin_xy": (float(cabin_node["x"]), float(cabin_node["y"])),
                "cabin_bounds_enu": None,
                "door_bounds_enu": None,
                "elevation_y": float(fm_elev_y) if fm_elev_y is not None else None,
            })
    if out:
        return out

    # Legacy fallback: explicit node with kind == 'elevator'
    for n in getattr(floor_map, "nodes", []) or []:
        if n.get("kind") == "elevator":
            xy = (float(n["x"]), float(n["y"]))
            out.append({
                "elevator": None,
                "room": None,
                "elevator_id": "",
                "name": str(n.get("label") or "电梯"),
                "floor_label": fm_floor,
                "door_node_id": n["id"],
                "cabin_node_id": n["id"],
                "door_xy": xy,
                "cabin_xy": xy,
                "cabin_bounds_enu": None,
                "door_bounds_enu": None,
                "elevation_y": float(fm_elev_y) if fm_elev_y is not None else None,
            })
    return out


def find_elevator_binding(floor_map: Any) -> dict[str, Any] | None:
    """Locate the primary elevator binding (waiting door + cabin nav node) on an IndoorFloorMap."""
    bindings = find_elevator_bindings(floor_map)
    return bindings[0] if bindings else None


def _build_polyline_route(
    nodes: dict[str, dict[str, Any]],
    edge_by_id: dict[str, dict[str, Any]],
    node_ids: list[str],
    edge_ids: list[str],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
) -> GlobalRoute:
    sx, sy = float(start_xy[0]), float(start_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    pts: list[tuple[float, float]] = [(sx, sy)]
    pkinds: list[str] = ["start"]
    sclasses: list[str] = []

    for idx, nid in enumerate(node_ids):
        pt = (float(nodes[nid]["x"]), float(nodes[nid]["y"]))
        nk = str(nodes[nid].get("kind") or "waypoint")
        prev_ec = (
            str(edge_by_id[edge_ids[idx - 1]].get("road_class", "walkway"))
            if idx > 0 and (idx - 1) < len(edge_ids) and edge_ids[idx - 1] in edge_by_id
            else (
                str(edge_by_id[edge_ids[0]].get("road_class", "walkway"))
                if edge_ids and edge_ids[0] in edge_by_id
                else "walkway"
            )
        )
        if math.hypot(pt[0] - pts[-1][0], pt[1] - pts[-1][1]) > 0.4 or nk in (
            "entrance",
            "room_door",
            "room_nav",
        ):
            pts.append(pt)
            pkinds.append(nk)
            sclasses.append(prev_ec)
    if math.hypot(gx - pts[-1][0], gy - pts[-1][1]) > 0.4:
        last_ec = sclasses[-1] if sclasses else "walkway"
        pts.append((gx, gy))
        pkinds.append("goal")
        sclasses.append(last_ec)

    # Avoid unnecessary backtracking to start_node if robot is already along the first edge,
    # but NEVER shortcut past a doorway/entrance node (keeps robot aligned through doors).
    if len(pts) >= 3 and pkinds[1] not in ("entrance", "room_door", "room_nav"):
        qx, qy, d_seg, t = _project_point_to_segment(
            sx, sy, pts[1][0], pts[1][1], pts[2][0], pts[2][1]
        )
        if t > 0.05 and d_seg < math.hypot(sx - pts[1][0], sy - pts[1][1]):
            pts = [(sx, sy), (qx, qy), *pts[2:]]
            pkinds = ["start", "waypoint", *pkinds[2:]]

    total = sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )
    return GlobalRoute(
        points=pts,
        node_ids=list(node_ids),
        edge_ids=list(edge_ids),
        total_length_m=total,
        point_kinds=pkinds,
        seg_classes=sclasses,
    )


class GlobalPlanner:
    def __init__(self, blocked_edge_ttl_s: float = 120.0) -> None:
        self.blocked_edge_ttl_s = float(blocked_edge_ttl_s)
        self._blocked: dict[str, float] = {}

    def mark_edge_blocked(self, edge_id: str, ttl_s: float | None = None) -> None:
        if not edge_id:
            return
        ttl = self.blocked_edge_ttl_s if ttl_s is None else float(ttl_s)
        self._blocked[edge_id] = time.monotonic() + ttl

    def clear_blocked_edges(self) -> None:
        self._blocked.clear()

    def active_blocked_edges(self) -> set[str]:
        now = time.monotonic()
        expired = [eid for eid, exp in self._blocked.items() if exp <= now]
        for eid in expired:
            del self._blocked[eid]
        return set(self._blocked)

    def plan(
        self,
        campus_map: CampusMap,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        start_node_id: str = "",
        goal_node_id: str = "",
    ) -> GlobalRoute | None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        nodes = campus_map.node_map()
        if not nodes or not campus_map.edges:
            # Direct straight-line fallback when user navigates without a road graph yet.
            dist = math.hypot(gx - sx, gy - sy)
            return GlobalRoute(points=[(sx, sy), (gx, gy)], node_ids=[], edge_ids=[], total_length_m=dist)

        blocked = self.active_blocked_edges()
        adj: dict[str, list[tuple[str, float, str]]] = {nid: [] for nid in nodes}
        for e in campus_map.edges:
            eid = e["id"]
            if eid in blocked:
                continue
            u, v = e["from"], e["to"]
            if u not in nodes or v not in nodes:
                continue
            if "fixed_cost" in e:
                cost = float(e["fixed_cost"])
            else:
                d = math.hypot(nodes[u]["x"] - nodes[v]["x"], nodes[u]["y"] - nodes[v]["y"])
                cost = d * float(e.get("weight", 1.0))
            adj[u].append((v, cost, eid))
            if e.get("bidirectional", True):
                adj[v].append((u, cost, eid))

        start_node = nodes.get(start_node_id) if start_node_id else None
        if start_node is None:
            start_node, _ = campus_map.find_nearest_node(sx, sy)
        goal_node = nodes.get(goal_node_id) if goal_node_id else None
        if goal_node is None:
            goal_node, _ = campus_map.find_nearest_node(gx, gy)
        if start_node is None or goal_node is None:
            return None

        s_id, g_id = start_node["id"], goal_node["id"]
        if s_id == g_id:
            pts = [(sx, sy), (gx, gy)]
            return GlobalRoute(
                points=pts,
                node_ids=[s_id],
                edge_ids=[],
                total_length_m=math.hypot(gx - sx, gy - sy),
            )

        # Standard A* over the campus/stitched graph (Euclidean XY heuristic stays admissible
        # even across vertical elevator edges because fixed_cost >= horizontal delta).
        g_score: dict[str, float] = {s_id: 0.0}
        came_from: dict[str, tuple[str, str]] = {}
        heap: list[tuple[float, str]] = [
            (math.hypot(nodes[s_id]["x"] - nodes[g_id]["x"], nodes[s_id]["y"] - nodes[g_id]["y"]), s_id)
        ]
        closed: set[str] = set()

        while heap:
            _, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            if cur == g_id:
                break
            closed.add(cur)
            cur_g = g_score[cur]
            for nxt, w, eid in adj.get(cur, ()):
                if nxt in closed:
                    continue
                cand = cur_g + w
                if cand < g_score.get(nxt, float("inf")):
                    g_score[nxt] = cand
                    came_from[nxt] = (cur, eid)
                    h = math.hypot(nodes[nxt]["x"] - nodes[g_id]["x"], nodes[nxt]["y"] - nodes[g_id]["y"])
                    heapq.heappush(heap, (cand + h, nxt))

        if g_id not in came_from:
            return None

        node_ids = [g_id]
        edge_ids: list[str] = []
        cur = g_id
        while cur != s_id:
            prev, eid = came_from[cur]
            edge_ids.append(eid)
            node_ids.append(prev)
            cur = prev
        node_ids.reverse()
        edge_ids.reverse()

        edge_by_id = {e["id"]: e for e in campus_map.edges}
        return _build_polyline_route(nodes, edge_by_id, node_ids, edge_ids, (sx, sy), (gx, gy))

    def plan_unified(
        self,
        outdoor_map: CampusMap,
        indoor_maps: list[Any],
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        start_indoor_id: str = "",
        goal_indoor_id: str = "",
        goal_node_id: str = "",
    ) -> GlobalRoute | None:
        """Plan across outdoor campus roads and multi-floor indoor maps, including elevators.

        - Same indoor floor: plans purely on that IndoorFloorMap (zero cross-floor overlap).
        - Outdoor <-> Indoor or Floor A <-> Floor B: stitches outdoor roads, entrance bridges,
          and all-pairs intra-building elevator links, runs A*, and decomposes any elevator
          crossings into ordered `RouteLeg` stages (`walk` -> `elevator` -> `walk`).
        """
        by_id = {fm.indoor_id: fm for fm in indoor_maps}

        # Case 1: Pure intra-floor navigation (e.g. 3F room A -> 3F room B)
        if start_indoor_id and start_indoor_id == goal_indoor_id and start_indoor_id in by_id:
            fm = by_id[start_indoor_id]
            sub = CampusMap(
                campus_id=fm.indoor_id,
                name=f"{fm.building_name} {fm.floor}",
                nodes=list(fm.nodes),
                edges=list(fm.edges),
            )
            route = self.plan(sub, start_xy, goal_xy, goal_node_id=goal_node_id)
            if route is None:
                return None
            leg = RouteLeg(
                kind="walk",
                description=f"{fm.building_name} {fm.floor}",
                route=route,
                indoor_id=fm.indoor_id,
                floor=fm.floor,
                goal_xy=goal_xy,
                goal_node_id=goal_node_id,
            )
            return GlobalRoute(
                points=route.points,
                node_ids=route.node_ids,
                edge_ids=route.edge_ids,
                total_length_m=route.total_length_m,
                point_kinds=route.point_kinds,
                seg_classes=route.seg_classes,
                legs=[leg],
            )

        # Case 2: Outdoor <-> Indoor or Multi-floor Indoor <-> Multi-floor Indoor
        stitched = CampusMap(campus_id=outdoor_map.campus_id, name=outdoor_map.name)
        for n in outdoor_map.nodes:
            stitched.nodes.append({**n, "id": f"out:{n['id']}"})
        for e in outdoor_map.edges:
            stitched.edges.append({
                **e,
                "id": f"out:{e['id']}",
                "from": f"out:{e['from']}",
                "to": f"out:{e['to']}",
            })

        include_ids: set[str] = {fm.indoor_id for fm in indoor_maps if fm.nodes}
        out_node_ids = {n["id"] for n in outdoor_map.nodes}
        elev_by_cabin: dict[str, dict[str, Any]] = {}
        shafts_by_key: dict[tuple[str, str], list[tuple[Any, dict[str, Any]]]] = {}
        stairs_by_key: dict[tuple[str, str], list[tuple[Any, dict[str, Any]]]] = {}

        for iid in sorted(include_ids):
            fm = by_id[iid]
            prefix = f"in:{iid}:"
            for n in fm.nodes:
                stitched.nodes.append({**n, "id": f"{prefix}{n['id']}"})
            for e in fm.edges:
                stitched.edges.append({
                    **e,
                    "id": f"{prefix}{e['id']}",
                    "from": f"{prefix}{e['from']}",
                    "to": f"{prefix}{e['to']}",
                })
            # Bridge every outdoor-linked entrance node
            for ent in fm.entrances:
                in_nid = str(ent.get("node_id") or "")
                out_nid = str(ent.get("outdoor_node_id") or "")
                if not in_nid:
                    continue
                if out_nid and out_nid in out_node_ids:
                    bridge_out = f"out:{out_nid}"
                else:
                    near_o, _ = outdoor_map.find_nearest_node(
                        float(ent.get("x", 0.0)), float(ent.get("y", 0.0)), max_dist_m=10.0
                    )
                    bridge_out = f"out:{near_o['id']}" if near_o is not None else ""
                if bridge_out:
                    stitched.edges.append({
                        "id": f"bridge:{iid}:{ent['id']}",
                        "from": bridge_out,
                        "to": f"{prefix}{in_nid}",
                        "name": str(ent.get("name") or "建筑出入口"),
                        "road_class": "walkway",
                        "width_m": 2.5,
                        "weight": 1.0,
                        "bidirectional": True,
                    })

            for binding in find_elevator_bindings(fm):
                cabin_full = f"{prefix}{binding['cabin_node_id']}"
                elev_by_cabin[cabin_full] = binding
                shaft_key = (fm.building_id, binding.get("elevator_id") or "__default__")
                shafts_by_key.setdefault(shaft_key, []).append((fm, binding))

            # Collect dedicated walkable stairs landings per (building_id, stairs_id)
            fm_nodes = fm.node_map()
            for st in getattr(fm, "stairs", []) or []:
                snid = str(st.get("node_id") or "")
                if snid and snid in fm_nodes:
                    skey = (fm.building_id, str(st.get("stairs_id") or "stairs_main"))
                    stairs_by_key.setdefault(skey, []).append((fm, st))

        # Connect adjacent floors belonging to the same staircase via walkable stairs edges.
        for (bid, st_id), st_entries in stairs_by_key.items():
            if len(st_entries) < 2:
                continue
            st_sorted = sorted(st_entries, key=lambda pair: (_floor_number(pair[0].floor), pair[0].indoor_id))
            for i in range(len(st_sorted) - 1):
                fm_a, sa = st_sorted[i]
                fm_b, sb = st_sorted[i + 1]
                if fm_a.indoor_id == fm_b.indoor_id:
                    continue
                na = f"in:{fm_a.indoor_id}:{sa['node_id']}"
                nb = f"in:{fm_b.indoor_id}:{sb['node_id']}"
                stitched.edges.append({
                    "id": f"stairs:{bid}:{st_id}:{fm_a.indoor_id}:{fm_b.indoor_id}",
                    "from": na,
                    "to": nb,
                    "name": f"{fm_a.building_name}{sa.get('name') or '楼梯'} {fm_a.floor}<->{fm_b.floor}",
                    "road_class": "corridor",
                    "width_m": 1.6,
                    "weight": 1.2,
                    "fixed_cost": 4.0,
                    "bidirectional": True,
                })

        # Connect all floors belonging to the same elevator shaft via virtual vertical edges.
        for (bid, shaft_id), entries in shafts_by_key.items():
            if len(entries) < 2:
                continue
            entries_sorted = sorted(entries, key=lambda pair: (_floor_number(pair[0].floor), pair[0].indoor_id))
            for i in range(len(entries_sorted)):
                fm_a, ba = entries_sorted[i]
                na = f"in:{fm_a.indoor_id}:{ba['cabin_node_id']}"
                for j in range(i + 1, len(entries_sorted)):
                    fm_b, bb = entries_sorted[j]
                    if fm_a.indoor_id == fm_b.indoor_id:
                        continue
                    nb = f"in:{fm_b.indoor_id}:{bb['cabin_node_id']}"
                    span = max(1, abs(_floor_number(fm_b.floor) - _floor_number(fm_a.floor)))
                    stitched.edges.append({
                        "id": f"elev:{bid}:{shaft_id}:{fm_a.indoor_id}:{fm_b.indoor_id}",
                        "from": na,
                        "to": nb,
                        "name": f"{fm_a.building_name}{ba.get('name') or '电梯'} {fm_a.floor}<->{fm_b.floor}",
                        "road_class": "elevator",
                        "width_m": 2.0,
                        "weight": 1.0,
                        "fixed_cost": 15.0 + 3.0 * span,
                        "bidirectional": True,
                    })

        # Strictly scope start_node and goal_node to their respective layer so overlapping
        # multi-floor (x, y) coordinates never snap to the wrong floor!
        mapped_start_nid = ""
        if start_indoor_id and start_indoor_id in by_id:
            sn, _ = by_id[start_indoor_id].find_nearest_node(start_xy[0], start_xy[1])
            if sn is not None:
                mapped_start_nid = f"in:{start_indoor_id}:{sn['id']}"
        elif outdoor_map.nodes:
            sn, _ = outdoor_map.find_nearest_node(start_xy[0], start_xy[1])
            if sn is not None:
                mapped_start_nid = f"out:{sn['id']}"

        mapped_goal_nid = ""
        if goal_indoor_id and goal_indoor_id in by_id:
            if goal_node_id and goal_node_id in by_id[goal_indoor_id].node_map():
                mapped_goal_nid = f"in:{goal_indoor_id}:{goal_node_id}"
            else:
                gn, _ = by_id[goal_indoor_id].find_nearest_node(goal_xy[0], goal_xy[1])
                if gn is not None:
                    mapped_goal_nid = f"in:{goal_indoor_id}:{gn['id']}"
        elif goal_node_id and goal_node_id in out_node_ids:
            mapped_goal_nid = f"out:{goal_node_id}"
        elif not goal_indoor_id and outdoor_map.nodes:
            gn, _ = outdoor_map.find_nearest_node(goal_xy[0], goal_xy[1])
            if gn is not None:
                mapped_goal_nid = f"out:{gn['id']}"

        raw = self.plan(
            stitched,
            start_xy,
            goal_xy,
            start_node_id=mapped_start_nid,
            goal_node_id=mapped_goal_nid,
        )
        if raw is None:
            return None

        # Check if any elevator edge was traversed; if not, single walk leg.
        elev_indices = [idx for idx, eid in enumerate(raw.edge_ids) if eid.startswith("elev:")]
        if not elev_indices:
            single_leg = RouteLeg(
                kind="walk",
                description=(
                    f"{by_id[goal_indoor_id].building_name} {by_id[goal_indoor_id].floor}"
                    if goal_indoor_id in by_id
                    else "室外园区"
                ),
                route=raw,
                indoor_id=goal_indoor_id,
                floor=by_id[goal_indoor_id].floor if goal_indoor_id in by_id else "",
                goal_xy=goal_xy,
                goal_node_id=goal_node_id,
            )
            return GlobalRoute(
                points=raw.points,
                node_ids=raw.node_ids,
                edge_ids=raw.edge_ids,
                total_length_m=raw.total_length_m,
                point_kinds=raw.point_kinds,
                seg_classes=raw.seg_classes,
                legs=[single_leg],
            )

        # Decompose into ordered [walk, elevator, walk, ...] legs.
        stitched_nodes = stitched.node_map()
        edge_by_id = {e["id"]: e for e in stitched.edges}
        legs: list[RouteLeg] = []

        def _parse_iid(nid: str) -> str:
            if nid.startswith("in:"):
                parts = nid.split(":", 2)
                if len(parts) == 3:
                    return parts[1]
            return ""

        seg_start_idx = 0
        cur_start_xy = start_xy

        for e_idx in elev_indices:
            src_nid = raw.node_ids[e_idx]
            dst_nid = raw.node_ids[e_idx + 1]
            src_iid = _parse_iid(src_nid)
            dst_iid = _parse_iid(dst_nid)
            src_fm = by_id[src_iid]
            dst_fm = by_id[dst_iid]
            src_bind = elev_by_cabin[src_nid]
            dst_bind = elev_by_cabin[dst_nid]
            eff_elev_id = str(src_bind.get("elevator_id") or dst_bind.get("elevator_id") or "")

            # Build walk leg up to the source floor elevator waiting door (NOT inside cabin).
            walk_nids = list(raw.node_ids[seg_start_idx : e_idx + 1])
            walk_eids = list(raw.edge_ids[seg_start_idx:e_idx])
            src_door_full = f"in:{src_iid}:{src_bind['door_node_id']}"
            src_cabin_full = f"in:{src_iid}:{src_bind['cabin_node_id']}"
            if (
                len(walk_nids) >= 2
                and walk_nids[-1] == src_cabin_full
                and walk_nids[-2] == src_door_full
            ):
                walk_nids.pop()
                if walk_eids:
                    walk_eids.pop()
            elif walk_nids and walk_nids[-1] == src_cabin_full and src_door_full != src_cabin_full:
                # Robot started near cabin; target waiting door first
                walk_nids = [src_door_full]
                walk_eids = []

            wait_xy = src_bind["door_xy"]
            walk_route = _build_polyline_route(
                stitched_nodes, edge_by_id, walk_nids, walk_eids, cur_start_xy, wait_xy
            )
            legs.append(
                RouteLeg(
                    kind="walk",
                    description=f"前往 {src_fm.building_name} {src_fm.floor} 电梯口",
                    route=walk_route,
                    indoor_id=src_iid,
                    floor=src_fm.floor,
                    goal_xy=wait_xy,
                    goal_node_id=src_bind["door_node_id"],
                )
            )

            # Elevator ride leg: src_floor -> dst_floor (carrying offline-bound geometry!)
            legs.append(
                RouteLeg(
                    kind="elevator",
                    description=f"乘电梯：{src_fm.building_name} {src_fm.floor} → {dst_fm.floor}",
                    elevator_id=eff_elev_id,
                    building_id=src_fm.building_id,
                    building_name=src_fm.building_name,
                    from_indoor_id=src_iid,
                    from_floor=src_fm.floor,
                    from_floor_label=str(src_bind.get("floor_label") or src_fm.floor),
                    from_door_xy=src_bind["door_xy"],
                    from_cabin_xy=src_bind["cabin_xy"],
                    from_door_node_id=src_bind["door_node_id"],
                    from_cabin_node_id=src_bind["cabin_node_id"],
                    from_cabin_bounds_enu=src_bind.get("cabin_bounds_enu"),
                    from_door_bounds_enu=src_bind.get("door_bounds_enu"),
                    from_elevation_y=src_bind.get("elevation_y"),
                    to_indoor_id=dst_iid,
                    to_floor=dst_fm.floor,
                    to_floor_label=str(dst_bind.get("floor_label") or dst_fm.floor),
                    to_door_xy=dst_bind["door_xy"],
                    to_cabin_xy=dst_bind["cabin_xy"],
                    to_door_node_id=dst_bind["door_node_id"],
                    to_cabin_node_id=dst_bind["cabin_node_id"],
                    to_cabin_bounds_enu=dst_bind.get("cabin_bounds_enu"),
                    to_door_bounds_enu=dst_bind.get("door_bounds_enu"),
                    to_elevation_y=dst_bind.get("elevation_y"),
                )
            )

            # Next walk leg begins at destination floor elevator waiting door (after egress).
            dst_door_full = f"in:{dst_iid}:{dst_bind['door_node_id']}"
            dst_cabin_full = f"in:{dst_iid}:{dst_bind['cabin_node_id']}"
            nxt_idx = e_idx + 1
            if (
                nxt_idx + 1 < len(raw.node_ids)
                and raw.node_ids[nxt_idx] == dst_cabin_full
                and raw.node_ids[nxt_idx + 1] == dst_door_full
            ):
                nxt_idx += 1
            seg_start_idx = nxt_idx
            cur_start_xy = dst_bind["door_xy"]

        # Final walk leg after the last elevator ride (skip only if goal IS the destination elevator itself).
        tail_nids = list(raw.node_ids[seg_start_idx:])
        tail_eids = list(raw.edge_ids[seg_start_idx:])
        last_elev = legs[-1]
        is_goal_elev_itself = (
            len(tail_nids) <= 1
            and last_elev.to_cabin_xy is not None
            and last_elev.to_door_xy is not None
            and (
                math.hypot(goal_xy[0] - last_elev.to_cabin_xy[0], goal_xy[1] - last_elev.to_cabin_xy[1]) <= 1.5
                or math.hypot(goal_xy[0] - last_elev.to_door_xy[0], goal_xy[1] - last_elev.to_door_xy[1]) <= 1.5
            )
        )
        if not is_goal_elev_itself and tail_nids:
            tail_route = _build_polyline_route(
                stitched_nodes, edge_by_id, tail_nids, tail_eids, cur_start_xy, goal_xy
            )
            legs.append(
                RouteLeg(
                    kind="walk",
                    description=(
                        f"{by_id[goal_indoor_id].building_name} {by_id[goal_indoor_id].floor}"
                        if goal_indoor_id in by_id
                        else "前往室外目的地"
                    ),
                    route=tail_route,
                    indoor_id=goal_indoor_id,
                    floor=by_id[goal_indoor_id].floor if goal_indoor_id in by_id else "",
                    goal_xy=goal_xy,
                    goal_node_id=goal_node_id,
                )
            )

        walk_total = sum(leg.route.total_length_m for leg in legs if leg.kind == "walk" and leg.route is not None)
        return GlobalRoute(
            points=raw.points,
            node_ids=raw.node_ids,
            edge_ids=raw.edge_ids,
            total_length_m=walk_total,
            point_kinds=raw.point_kinds,
            seg_classes=raw.seg_classes,
            legs=legs,
        )
