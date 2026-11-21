# SPDX-License-Identifier: MulanPSL-2.0
"""Multi-building, multi-floor indoor topological map store and odometry survey recorder.

Each building floor is stored as its own JSON file under `maps/indoor/<indoor_id>.json`
so multi-floor corridors and rooms never overlap on 2D views and can be edited/reset
independently, while remaining logically connected to the outdoor campus road graph
through explicit entrance anchor links (`outdoor_node_id` <-> `entrance.node_id`).
"""
from __future__ import annotations

import copy
import json
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from campus_nav_rbnx.campus_map import CampusMap

INDOOR_NODE_KINDS = frozenset({"entrance", "elevator", "stairs", "corridor", "room_door", "room_nav"})
INDOOR_EDGE_CLASSES = frozenset({"corridor", "room_entry"})
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_\-]")


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def make_indoor_id(building_id: str, floor: str) -> str:
    b = _SAFE_ID_RE.sub("_", (building_id or "building").strip()) or "building"
    f = _SAFE_ID_RE.sub("_", (floor or "1F").strip().upper()) or "1F"
    return f"{b}__{f}"


@dataclass
class IndoorFloorMap:
    indoor_id: str
    building_id: str
    building_name: str = ""
    floor: str = "1F"
    elevation_y: float | None = None
    entrances: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    rooms: list[dict[str, Any]] = field(default_factory=list)
    markers: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "indoor_id": self.indoor_id,
            "building_id": self.building_id,
            "building_name": self.building_name,
            "floor": self.floor,
            "elevation_y": self.elevation_y,
            "entrances": copy.deepcopy(self.entrances),
            "nodes": copy.deepcopy(self.nodes),
            "edges": copy.deepcopy(self.edges),
            "rooms": copy.deepcopy(self.rooms),
            "markers": copy.deepcopy(self.markers),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "indoor_id": self.indoor_id,
            "building_id": self.building_id,
            "building_name": self.building_name or self.building_id,
            "floor": self.floor,
            "elevation_y": self.elevation_y,
            "entrance_count": len(self.entrances),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "room_count": len(self.rooms),
            "marker_count": len(self.markers),
            "entrances": copy.deepcopy(self.entrances),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IndoorFloorMap":
        if not isinstance(raw, dict):
            raise ValueError("indoor floor map must be a JSON object")
        building_id = str(raw.get("building_id") or "building")
        floor = str(raw.get("floor") or "1F").upper()
        indoor_id = str(raw.get("indoor_id") or make_indoor_id(building_id, floor))
        elev = raw.get("elevation_y")
        m = cls(
            indoor_id=indoor_id,
            building_id=building_id,
            building_name=str(raw.get("building_name") or building_id),
            floor=floor,
            elevation_y=float(elev) if elev is not None and math.isfinite(float(elev)) else None,
        )
        seen_nodes: set[str] = set()
        for n in raw.get("nodes") or []:
            nid = str(n.get("id") or _short_id("in"))
            x, y = float(n["x"]), float(n["y"])
            if not (math.isfinite(x) and math.isfinite(y)) or nid in seen_nodes:
                continue
            seen_nodes.add(nid)
            nk = str(n.get("kind") or "corridor")
            if nk not in INDOOR_NODE_KINDS:
                nk = "corridor"
            obj: dict[str, Any] = {
                "id": nid,
                "x": round(x, 3),
                "y": round(y, 3),
                "kind": nk,
                "indoor": True,
                "floor": floor,
            }
            if n.get("label"):
                obj["label"] = str(n["label"])
            m.nodes.append(obj)

        for e in raw.get("edges") or []:
            u, v = str(e.get("from") or ""), str(e.get("to") or "")
            if u not in seen_nodes or v not in seen_nodes or u == v:
                continue
            rc = str(e.get("road_class") or "corridor")
            if rc not in INDOOR_EDGE_CLASSES:
                rc = "corridor"
            m.edges.append({
                "id": str(e.get("id") or _short_id("ie")),
                "from": u,
                "to": v,
                "name": str(e.get("name") or ""),
                "road_class": rc,
                "width_m": max(1.0, min(10.0, float(e.get("width_m", 2.0)))),
                "weight": max(0.1, min(20.0, float(e.get("weight", 1.0)))),
                "bidirectional": bool(e.get("bidirectional", True)),
            })

        for ent in raw.get("entrances") or []:
            nid = str(ent.get("node_id") or "")
            if nid not in seen_nodes:
                continue
            m.entrances.append({
                "id": str(ent.get("id") or _short_id("ent")),
                "name": str(ent.get("name") or f"{m.building_name}{floor}出口"),
                "node_id": nid,
                "outdoor_node_id": str(ent.get("outdoor_node_id") or ""),
                "x": round(float(ent.get("x", 0.0)), 3),
                "y": round(float(ent.get("y", 0.0)), 3),
            })

        for r in raw.get("rooms") or []:
            name = str(r.get("name") or "").strip()
            if not name:
                continue
            rx, ry = float(r["x"]), float(r["y"])
            if not (math.isfinite(rx) and math.isfinite(ry)):
                continue
            nid = str(r.get("node_id") or "")
            if nid not in seen_nodes:
                nid = ""
            dnid = str(r.get("door_node_id") or "")
            if dnid not in seen_nodes:
                dnid = ""
            m.rooms.append({
                "id": str(r.get("id") or _short_id("rm")),
                "name": name,
                "aliases": [str(a).strip() for a in (r.get("aliases") or []) if str(a).strip()],
                "category": "room",
                "x": round(rx, 3),
                "y": round(ry, 3),
                "node_id": nid,
                "door_node_id": dnid,
                "indoor_id": indoor_id,
                "building_id": building_id,
                "building_name": m.building_name,
                "floor": floor,
            })
        for mk in raw.get("markers") or []:
            try:
                mid = int(mk["marker_id"])
                mx, my = float(mk["x"]), float(mk["y"])
                mz = float(mk.get("z", 0.0))
                wyaw = float(mk.get("wall_yaw", 0.0))
                sz = float(mk.get("size_m", 98.0 / 128.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(mx) and math.isfinite(my) and math.isfinite(mz)):
                continue
            m.markers.append({
                "id": str(mk.get("id") or f"mk_{mid}"),
                "marker_id": mid,
                "label": str(mk.get("label") or f"标牌#{mid}"),
                "x": round(mx, 4),
                "y": round(my, 4),
                "z": round(mz, 4),
                "wall_yaw": round(wyaw, 5),
                "size_m": round(max(0.1, min(3.0, sz)), 6),
                "indoor_id": indoor_id,
                "building_id": building_id,
                "building_name": m.building_name,
                "floor": floor,
            })
        return m

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "IndoorFloorMap":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def node_map(self) -> dict[str, dict[str, Any]]:
        return {n["id"]: n for n in self.nodes}

    def find_nearest_node(
        self,
        x: float,
        y: float,
        max_dist_m: float = float("inf"),
        exclude: str | None = None,
        *,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> tuple[dict[str, Any] | None, float]:
        best = None
        best_d = max_dist_m
        for n in self.nodes:
            if exclude and n["id"] == exclude:
                continue
            if kinds is not None and str(n.get("kind")) not in kinds:
                continue
            d = math.hypot(n["x"] - x, n["y"] - y)
            if d < best_d:
                best = n
                best_d = d
        return best, best_d

    def add_node(self, x: float, y: float, kind: str = "corridor", *, label: str = "") -> dict[str, Any]:
        k = kind if kind in INDOOR_NODE_KINDS else "corridor"
        node: dict[str, Any] = {
            "id": _short_id("in"),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "kind": k,
            "indoor": True,
            "floor": self.floor,
        }
        if label:
            node["label"] = label
        self.nodes.append(node)
        return node

    def has_edge(self, u: str, v: str) -> bool:
        for e in self.edges:
            if (e["from"] == u and e["to"] == v) or (e["bidirectional"] and e["from"] == v and e["to"] == u):
                return True
        return False

    def add_edge(
        self,
        u: str,
        v: str,
        *,
        name: str = "",
        road_class: str = "corridor",
        width_m: float = 2.0,
        weight: float = 1.0,
    ) -> dict[str, Any] | None:
        if u == v or self.has_edge(u, v):
            return None
        rc = road_class if road_class in INDOOR_EDGE_CLASSES else "corridor"
        edge = {
            "id": _short_id("ie"),
            "from": u,
            "to": v,
            "name": name,
            "road_class": rc,
            "width_m": max(1.0, float(width_m)),
            "weight": max(0.1, float(weight)),
            "bidirectional": True,
        }
        self.edges.append(edge)
        return edge

    def attach_node_to_nearest_corridor(
        self,
        node: dict[str, Any],
        *,
        max_dist_m: float = 15.0,
        split_edge_max_dist_m: float = 2.2,
    ) -> list[str]:
        """Connect `node` to the nearest existing corridor node or split the closest corridor edge.

        Guarantees room doorways always attach to the geometrically closest point on the
        corridor graph rather than chaining onto a faraway corridor dead-end.
        """
        nid = node["id"]
        x, y = float(node["x"]), float(node["y"])
        nodes = self.node_map()
        near_c, d_node = self.find_nearest_node(
            x,
            y,
            max_dist_m=max_dist_m,
            exclude=nid,
            kinds={"corridor", "entrance", "elevator", "stairs"},
        )
        # Fallback to another room_door only if no corridor/entrance node is closer
        if near_c is None:
            near_c, d_node = self.find_nearest_node(
                x,
                y,
                max_dist_m=max_dist_m,
                exclude=nid,
                kinds={"room_door"},
            )

        best_edge = None
        best_edge_d = split_edge_max_dist_m
        for e in self.edges:
            if e.get("road_class") != "corridor":
                continue
            u_id, v_id = e["from"], e["to"]
            if u_id == nid or v_id == nid or u_id not in nodes or v_id not in nodes:
                continue
            ux, uy = nodes[u_id]["x"], nodes[u_id]["y"]
            vx, vy = nodes[v_id]["x"], nodes[v_id]["y"]
            dx, dy = vx - ux, vy - uy
            len2 = dx * dx + dy * dy
            if len2 <= 1e-6:
                continue
            t = max(0.0, min(1.0, ((x - ux) * dx + (y - uy) * dy) / len2))
            if t < 0.08 or t > 0.92:
                continue
            qx, qy = ux + t * dx, uy + t * dy
            d = math.hypot(x - qx, y - qy)
            if d < best_edge_d:
                best_edge_d = d
                best_edge = e

        if best_edge is not None and (near_c is None or (d_node > 1.2 and best_edge_d + 0.15 < d_node)):
            u_id, v_id = best_edge["from"], best_edge["to"]
            self.edges = [e for e in self.edges if e["id"] != best_edge["id"]]
            self.add_edge(u_id, nid, road_class="corridor", width_m=float(best_edge.get("width_m", 2.0)))
            self.add_edge(nid, v_id, road_class="corridor", width_m=float(best_edge.get("width_m", 2.0)))
            return [u_id, v_id]

        if near_c is not None:
            self.add_edge(near_c["id"], nid, road_class="corridor", width_m=2.0)
            return [near_c["id"]]
        return []

    def add_room(
        self,
        name: str,
        x: float,
        y: float,
        *,
        node_id: str,
        door_node_id: str = "",
    ) -> dict[str, Any]:
        room = {
            "id": _short_id("rm"),
            "name": name.strip(),
            "aliases": [],
            "category": "room",
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "node_id": node_id,
            "door_node_id": door_node_id,
            "indoor_id": self.indoor_id,
            "building_id": self.building_id,
            "building_name": self.building_name or self.building_id,
            "floor": self.floor,
        }
        self.rooms.append(room)
        return room

    def upsert_marker(
        self,
        marker_id: int,
        x: float,
        y: float,
        z: float,
        wall_yaw: float,
        *,
        label: str = "",
        size_m: float = 98.0 / 128.0,
    ) -> dict[str, Any]:
        mid = int(marker_id)
        entry = {
            "id": f"mk_{mid}",
            "marker_id": mid,
            "label": label.strip() or f"标牌#{mid}",
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "z": round(float(z), 4),
            "wall_yaw": round(float(wall_yaw), 5),
            "size_m": round(float(size_m), 6),
            "indoor_id": self.indoor_id,
            "building_id": self.building_id,
            "building_name": self.building_name or self.building_id,
            "floor": self.floor,
        }
        self.markers = [m for m in self.markers if int(m.get("marker_id", -1)) != mid]
        self.markers.append(entry)
        return entry


class IndoorMapStore:
    """Manages separate per-building, per-floor indoor map files under maps/indoor/."""

    def __init__(self, indoor_dir: Path) -> None:
        self.indoor_dir = indoor_dir
        self.indoor_dir.mkdir(parents=True, exist_ok=True)
        self._maps: dict[str, IndoorFloorMap] = {}
        self.reload()

    def reload(self) -> None:
        self._maps.clear()
        if not self.indoor_dir.is_dir():
            return
        for p in sorted(self.indoor_dir.glob("*.json")):
            try:
                fm = IndoorFloorMap.load(p)
                self._maps[fm.indoor_id] = fm
            except Exception:  # noqa: BLE001
                continue

    def _path(self, indoor_id: str) -> Path:
        safe = _SAFE_ID_RE.sub("_", indoor_id)
        return self.indoor_dir / f"{safe}.json"

    def get(self, indoor_id: str) -> IndoorFloorMap | None:
        return self._maps.get(indoor_id)

    def get_or_create(
        self,
        building_id: str,
        building_name: str,
        floor: str = "1F",
        *,
        elevation_y: float | None = None,
    ) -> IndoorFloorMap:
        iid = make_indoor_id(building_id, floor)
        existing = self._maps.get(iid)
        if existing is not None:
            if building_name and not existing.building_name:
                existing.building_name = building_name
            return existing
        fm = IndoorFloorMap(
            indoor_id=iid,
            building_id=building_id,
            building_name=building_name or building_id,
            floor=(floor or "1F").strip().upper(),
            elevation_y=elevation_y,
        )
        self._maps[iid] = fm
        self.save(fm)
        return fm

    def save(self, fm: IndoorFloorMap) -> None:
        self._maps[fm.indoor_id] = fm
        fm.save(self._path(fm.indoor_id))

    def replace(self, raw: dict[str, Any], *, save: bool = True) -> IndoorFloorMap:
        fm = IndoorFloorMap.from_dict(raw)
        self._maps[fm.indoor_id] = fm
        if save:
            self.save(fm)
        return fm

    def reset_floor(self, indoor_id: str) -> IndoorFloorMap | None:
        old = self._maps.get(indoor_id)
        if old is None:
            return None
        fresh = IndoorFloorMap(
            indoor_id=old.indoor_id,
            building_id=old.building_id,
            building_name=old.building_name,
            floor=old.floor,
            elevation_y=old.elevation_y,
        )
        self.save(fresh)
        return fresh

    def list_summaries(self) -> list[dict[str, Any]]:
        return [fm.summary() for fm in sorted(self._maps.values(), key=lambda m: (m.building_name, m.floor))]

    def all_maps(self) -> list[IndoorFloorMap]:
        return list(self._maps.values())

    def all_rooms(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fm in sorted(self._maps.values(), key=lambda m: (m.building_name, m.floor)):
            for r in fm.rooms:
                out.append(copy.deepcopy(r))
        return out

    def all_markers(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for fm in sorted(self._maps.values(), key=lambda m: (m.building_name, m.floor)):
            for mk in fm.markers:
                out.append(copy.deepcopy(mk))
        return out

    def find_marker(self, marker_id: int) -> tuple[IndoorFloorMap, dict[str, Any]] | None:
        mid = int(marker_id)
        for fm in self._maps.values():
            for mk in fm.markers:
                if int(mk.get("marker_id", -1)) == mid:
                    return fm, mk
        return None

    def find_room(self, query: str, *, indoor_id: str = "") -> tuple[IndoorFloorMap, dict[str, Any]] | None:
        q = query.strip().lower()
        if not q:
            return None
        q_norm = re.sub(r"[·\s_\-]+", " ", q).strip()
        # Normalize Chinese floor suffixes like '3楼'/'三楼' -> '3f' for matching
        cn_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6"}
        q_floor_norm = q_norm
        for cn, num in cn_map.items():
            q_floor_norm = q_floor_norm.replace(f"{cn}楼", f"{num}f").replace(f"{cn}层", f"{num}f")
        q_floor_norm = re.sub(r"(\d+)\s*[楼层]", r"\1f", q_floor_norm)

        maps_ordered = sorted(
            self._maps.values(),
            key=lambda m: (0 if (indoor_id and m.indoor_id == indoor_id) else 1, m.building_name, m.floor),
        )
        # Exact match first
        for fm in maps_ordered:
            for r in fm.rooms:
                full = f"{fm.building_name} {fm.floor} {r['name']}".lower()
                floor_room = f"{fm.floor} {r['name']}".lower()
                candidates = {
                    r["id"].lower(),
                    r["name"].lower(),
                    full,
                    floor_room,
                    f"{fm.floor}{r['name']}".lower(),
                    *(a.lower() for a in r.get("aliases", [])),
                }
                if q in candidates or q_norm in candidates or q_floor_norm in candidates:
                    return fm, r
        # Substring fallback (check all tokens present first, then plain substring)
        tokens = [t for t in q_floor_norm.split() if t]
        if len(tokens) >= 2:
            for fm in maps_ordered:
                for r in fm.rooms:
                    hay = f"{fm.building_name} {fm.floor} {r['name']} {' '.join(r.get('aliases', []))}".lower()
                    if all(t in hay for t in tokens):
                        return fm, r
        for fm in maps_ordered:
            for r in fm.rooms:
                full = f"{fm.building_name} {fm.floor} {r['name']}".lower()
                if (
                    q in r["name"].lower()
                    or q_norm in full
                    or q_floor_norm in full
                    or any(q in a.lower() for a in r.get("aliases", []))
                ):
                    return fm, r
        return None


class IndoorFloorSurveyRecorder:
    """Records corridors and paired (room_door + room_nav) rooms into a selected IndoorFloorMap."""

    def __init__(
        self,
        corridor_spacing_m: float = 2.5,
        turn_threshold_deg: float = 25.0,
        merge_radius_m: float = 1.6,
        outdoor_snap_m: float = 8.0,
    ) -> None:
        self.corridor_spacing_m = float(corridor_spacing_m)
        self.turn_threshold_rad = math.radians(turn_threshold_deg)
        self.merge_radius_m = float(merge_radius_m)
        self.outdoor_snap_m = float(outdoor_snap_m)
        self.state: str = "idle"  # idle | corridor | awaiting_room_nav | in_room_paused
        self.active_indoor_id: str = ""
        self.room_name: str = ""
        self.entrance_node_id: str | None = None
        self.last_corridor_node_id: str | None = None
        self.last_corridor_xy: tuple[float, float] | None = None
        self.last_heading: float | None = None
        self.pending_door_node_id: str | None = None
        self.last_door_node_id: str | None = None
        self._last_mark_seq: int = 0
        self.events: list[str] = []

    @property
    def active(self) -> bool:
        return self.state != "idle"

    def reset(self) -> None:
        self.state = "idle"
        self.entrance_node_id = None
        self.last_corridor_node_id = None
        self.last_corridor_xy = None
        self.last_heading = None
        self.pending_door_node_id = None
        self.last_door_node_id = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "active": self.active,
            "active_indoor_id": self.active_indoor_id,
            "room_name": self.room_name,
            "entrance_node_id": self.entrance_node_id,
            "last_corridor_node_id": self.last_corridor_node_id,
            "pending_door_node_id": self.pending_door_node_id,
            "last_door_node_id": self.last_door_node_id,
            "recent_events": list(self.events[-8:]),
        }

    def anchor_entrance(
        self,
        outdoor_map: CampusMap,
        floor_map: IndoorFloorMap,
        x: float,
        y: float,
        *,
        elevation_y: float | None = None,
        label: str = "",
    ) -> dict[str, Any]:
        self.active_indoor_id = floor_map.indoor_id
        if elevation_y is not None and floor_map.elevation_y is None:
            floor_map.elevation_y = round(float(elevation_y), 2)

        near_ent, _ = floor_map.find_nearest_node(x, y, max_dist_m=1.5, kinds={"entrance", "elevator"})
        if near_ent is not None:
            node = near_ent
        else:
            ent_label = label or f"{floor_map.building_name or ''}{floor_map.floor}出口"
            node = floor_map.add_node(x, y, kind="entrance", label=ent_label)
            # Only ground-floor (1F) entrances logically link to the outdoor road graph by default
            outdoor_node_id = ""
            if floor_map.floor in ("1F", "G", "GF", "1"):
                near_out, _ = outdoor_map.find_nearest_node(
                    x, y, max_dist_m=self.outdoor_snap_m, kinds={"waypoint", "junction", "entrance"}
                )
                if near_out is not None:
                    outdoor_node_id = near_out["id"]
            floor_map.entrances.append({
                "id": _short_id("ent"),
                "name": ent_label,
                "node_id": node["id"],
                "outdoor_node_id": outdoor_node_id,
                "x": node["x"],
                "y": node["y"],
            })

        self.entrance_node_id = node["id"]
        self.last_corridor_node_id = node["id"]
        self.last_corridor_xy = (node["x"], node["y"])
        self.last_heading = None
        self.pending_door_node_id = None
        self.state = "corridor"
        self._log(
            f"[{floor_map.building_name or floor_map.building_id} {floor_map.floor}] "
            f"设出口锚点 {node['id']} ({node['x']:.1f}, {node['y']:.1f})，开始走廊自动打点"
        )
        return node

    def mark_room_door(
        self,
        floor_map: IndoorFloorMap,
        x: float,
        y: float,
        *,
        room_name: str = "",
    ) -> dict[str, Any]:
        self.active_indoor_id = floor_map.indoor_id
        if room_name.strip():
            self.room_name = room_name.strip()
        door_label = f"{self.room_name}门口" if self.room_name else "房间门口"
        door = floor_map.add_node(x, y, kind="room_door", label=door_label)

        # Always attach to the geometrically nearest corridor node/segment (never chain to a faraway corridor end).
        floor_map.attach_node_to_nearest_corridor(door, max_dist_m=15.0)

        self.pending_door_node_id = door["id"]
        self.last_door_node_id = door["id"]
        self.last_corridor_node_id = door["id"]
        self.last_corridor_xy = (door["x"], door["y"])
        self.last_heading = None
        self.state = "awaiting_room_nav"
        self._log(f"[{floor_map.floor}] 标房间门口点 {door['id']} ({door['x']:.1f}, {door['y']:.1f})，已暂停走廊")
        return door

    def mark_room_nav(
        self,
        floor_map: IndoorFloorMap,
        x: float,
        y: float,
        *,
        room_name: str = "",
    ) -> dict[str, Any]:
        self.active_indoor_id = floor_map.indoor_id
        clean_name = room_name.strip() or self.room_name.strip()
        if not clean_name:
            clean_name = f"{floor_map.floor}-房间{len(floor_map.rooms) + 1}"

        nodes = floor_map.node_map()
        door_id = self.pending_door_node_id if (self.pending_door_node_id in nodes) else None
        if door_id is None:
            near_d, _ = floor_map.find_nearest_node(
                x, y, max_dist_m=15.0, kinds={"room_door", "corridor", "entrance"}
            )
            door_id = near_d["id"] if near_d is not None else ""

        nav_node = floor_map.add_node(x, y, kind="room_nav", label=clean_name)
        if door_id and door_id in nodes:
            floor_map.add_edge(
                door_id,
                nav_node["id"],
                name=f"{clean_name}入口",
                road_class="room_entry",
                width_m=1.5,
            )
            door_node = nodes[door_id]
            if not door_node.get("label") or door_node.get("label") == "房间门口":
                door_node["label"] = f"{clean_name}门口"

        room = floor_map.add_room(clean_name, x, y, node_id=nav_node["id"], door_node_id=door_id or "")
        self.pending_door_node_id = None
        if door_id:
            self.last_door_node_id = door_id
        self.room_name = ""
        self.state = "in_room_paused"
        self._log(f"[{floor_map.floor}] 标房间导航点「{clean_name}」({nav_node['x']:.1f}, {nav_node['y']:.1f})，待手动恢复走廊")
        return {"room": room, "nav_node": nav_node, "door_node_id": door_id or ""}

    def pause_corridor(self) -> dict[str, Any]:
        if self.state != "idle":
            self.state = "in_room_paused"
            self._log("已暂停走廊自动打点")
        return self.state_dict()

    def resume_corridor(
        self,
        floor_map: IndoorFloorMap,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        self.active_indoor_id = floor_map.indoor_id
        nodes = floor_map.node_map()
        anchor_id = None
        if self.last_door_node_id and self.last_door_node_id in nodes:
            anchor_id = self.last_door_node_id
        elif self.last_corridor_node_id and self.last_corridor_node_id in nodes:
            anchor_id = self.last_corridor_node_id
        if x is not None and y is not None:
            near_c, dist = floor_map.find_nearest_node(
                x, y, max_dist_m=4.0, kinds={"corridor", "room_door", "entrance", "elevator"}
            )
            if near_c is not None and (
                anchor_id is None
                or dist < math.hypot(nodes[anchor_id]["x"] - x, nodes[anchor_id]["y"] - y)
            ):
                anchor_id = near_c["id"]
        if anchor_id and anchor_id in nodes:
            n = nodes[anchor_id]
            self.last_corridor_node_id = n["id"]
            self.last_corridor_xy = (n["x"], n["y"])
        elif x is not None and y is not None:
            n = floor_map.add_node(x, y, kind="corridor")
            self.last_corridor_node_id = n["id"]
            self.last_corridor_xy = (n["x"], n["y"])
        self.pending_door_node_id = None
        self.last_heading = None
        self.state = "corridor"
        self._log(f"[{floor_map.floor}] 已手动恢复走廊自动打点")
        return self.state_dict()

    def ingest_sample(
        self,
        outdoor_map: CampusMap,
        floor_map: IndoorFloorMap,
        x: float,
        y: float,
        *,
        anchored: bool,
        elevation_y: float | None = None,
        mark_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changed = False
        auto_saved = False
        if mark_event and int(mark_event.get("seq", 0)) > self._last_mark_seq:
            self._last_mark_seq = int(mark_event["seq"])
            action = str(mark_event.get("action") or "mark_step")
            mpose = mark_event.get("pose") or {}
            mx = float(mpose.get("east", x))
            my = float(mpose.get("north", y))
            mup = mpose.get("up", elevation_y)
            if action == "anchor_entrance":
                self.anchor_entrance(outdoor_map, floor_map, mx, my, elevation_y=mup)
                changed = True
                auto_saved = True
            elif action == "bind_fiducial":
                extra = mark_event.get("extra") or {}
                if "marker_id" in extra:
                    mk = floor_map.upsert_marker(
                        int(extra["marker_id"]),
                        float(extra.get("x", mx)),
                        float(extra.get("y", my)),
                        float(extra.get("z", mup or 0.0)),
                        float(extra.get("wall_yaw", 0.0)),
                        size_m=float(extra.get("size_m", 98.0 / 128.0)),
                    )
                    self._log(
                        f"[{floor_map.building_name or floor_map.building_id} {floor_map.floor}] "
                        f"已绑定视觉标牌 #{mk['marker_id']} ({mk['x']:.1f}, {mk['y']:.1f})"
                    )
                    changed = True
                    auto_saved = True
            elif action == "mark_step":
                if self.state == "awaiting_room_nav":
                    self.mark_room_nav(floor_map, mx, my)
                else:
                    self.mark_room_door(floor_map, mx, my)
                changed = True
                auto_saved = True

        if self.state == "corridor" and (anchored or self.entrance_node_id is not None):
            if self._drop_corridor_node(floor_map, x, y):
                changed = True

        return {"changed": changed, "auto_saved": auto_saved, "state": self.state}

    def _drop_corridor_node(self, floor_map: IndoorFloorMap, x: float, y: float) -> bool:
        if self.last_corridor_xy is None or not self.last_corridor_node_id:
            near, _ = floor_map.find_nearest_node(
                x, y, max_dist_m=self.merge_radius_m, kinds={"corridor", "room_door", "entrance", "elevator"}
            )
            node = near if near is not None else floor_map.add_node(x, y, kind="corridor")
            self.last_corridor_node_id = node["id"]
            self.last_corridor_xy = (node["x"], node["y"])
            return True

        lx, ly = self.last_corridor_xy
        dist = math.hypot(x - lx, y - ly)
        heading = math.atan2(y - ly, x - lx) if dist > 0.4 else self.last_heading
        turned = False
        if heading is not None and self.last_heading is not None and dist >= 1.2:
            diff = abs((heading - self.last_heading + math.pi) % (2 * math.pi) - math.pi)
            turned = diff >= self.turn_threshold_rad

        if dist < self.corridor_spacing_m and not turned:
            return False

        near, _ = floor_map.find_nearest_node(
            x,
            y,
            max_dist_m=self.merge_radius_m,
            exclude=self.last_corridor_node_id,
            kinds={"corridor", "room_door", "entrance", "elevator"},
        )
        target = near if near is not None else floor_map.add_node(x, y, kind="corridor")
        if self.last_corridor_node_id != target["id"]:
            floor_map.add_edge(self.last_corridor_node_id, target["id"], road_class="corridor", width_m=2.0)
        self.last_heading = heading
        self.last_corridor_node_id = target["id"]
        self.last_corridor_xy = (target["x"], target["y"])
        return True

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")
        if len(self.events) > 30:
            self.events = self.events[-30:]
