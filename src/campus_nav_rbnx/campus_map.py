# SPDX-License-Identifier: MulanPSL-2.0
"""Google Maps-style vector campus map data model, persistence, and RTK recorder."""
from __future__ import annotations

import copy
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AREA_KINDS = frozenset({"building", "grass", "plaza", "water", "restricted"})
ROAD_CLASSES = frozenset({"primary", "walkway", "corridor", "room_entry"})
NODE_KINDS = frozenset({"waypoint", "junction", "entrance", "corridor", "room_door", "room_nav"})
POI_CATEGORIES = frozenset({"lab", "canteen", "dorm", "gate", "plaza", "room", "landmark", "other"})


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class CampusMap:
    campus_id: str = "campus_default"
    name: str = "主园区地图"
    frame: str = "campus_rtk"
    areas: list[dict[str, Any]] = field(default_factory=list)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    pois: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "campus_id": self.campus_id,
            "name": self.name,
            "frame": self.frame,
            "areas": copy.deepcopy(self.areas),
            "nodes": copy.deepcopy(self.nodes),
            "edges": copy.deepcopy(self.edges),
            "pois": copy.deepcopy(self.pois),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CampusMap":
        if not isinstance(raw, dict):
            raise ValueError("campus map must be a JSON object")
        m = cls(
            campus_id=str(raw.get("campus_id") or "campus_default"),
            name=str(raw.get("name") or "主园区地图"),
            frame=str(raw.get("frame") or "campus_rtk"),
        )
        seen_nodes: set[str] = set()
        for n in raw.get("nodes") or []:
            nid = str(n.get("id") or _short_id("n"))
            x, y = float(n["x"]), float(n["y"])
            if not (math.isfinite(x) and math.isfinite(y)) or nid in seen_nodes:
                continue
            seen_nodes.add(nid)
            kind = str(n.get("kind") or "waypoint")
            if kind not in NODE_KINDS:
                kind = "waypoint"
            node_obj: dict[str, Any] = {
                "id": nid,
                "x": round(x, 3),
                "y": round(y, 3),
                "kind": kind,
            }
            if n.get("indoor") or kind in ("corridor", "room_door", "room_nav"):
                node_obj["indoor"] = True
            if n.get("building_id"):
                node_obj["building_id"] = str(n["building_id"])
            if n.get("floor"):
                node_obj["floor"] = str(n["floor"])
            if n.get("label"):
                node_obj["label"] = str(n["label"])
            m.nodes.append(node_obj)
        for e in raw.get("edges") or []:
            u, v = str(e.get("from") or ""), str(e.get("to") or "")
            if u not in seen_nodes or v not in seen_nodes or u == v:
                continue
            rc = str(e.get("road_class") or "walkway")
            if rc not in ROAD_CLASSES:
                rc = "walkway"
            m.edges.append({
                "id": str(e.get("id") or _short_id("e")),
                "from": u,
                "to": v,
                "name": str(e.get("name") or ""),
                "road_class": rc,
                "width_m": max(1.0, min(20.0, float(e.get("width_m", 3.0)))),
                "weight": max(0.1, min(20.0, float(e.get("weight", 1.0)))),
                "bidirectional": bool(e.get("bidirectional", True)),
            })
        for a in raw.get("areas") or []:
            poly = []
            for pt in a.get("polygon") or []:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px, py = float(pt[0]), float(pt[1])
                    if math.isfinite(px) and math.isfinite(py):
                        poly.append([round(px, 3), round(py, 3)])
            if len(poly) < 3:
                continue
            kind = str(a.get("kind") or "building")
            if kind not in AREA_KINDS:
                kind = "building"
            m.areas.append({
                "id": str(a.get("id") or _short_id("a")),
                "name": str(a.get("name") or ""),
                "kind": kind,
                "polygon": poly,
            })
        for p in raw.get("pois") or []:
            name = str(p.get("name") or "").strip()
            if not name:
                continue
            px, py = float(p["x"]), float(p["y"])
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            cat = str(p.get("category") or "landmark")
            if cat not in POI_CATEGORIES:
                cat = "landmark"
            aliases = [str(al).strip() for al in (p.get("aliases") or []) if str(al).strip()]
            nid = str(p.get("node_id") or "")
            if nid not in seen_nodes:
                nid = ""
            door_nid = str(p.get("door_node_id") or "")
            if door_nid not in seen_nodes:
                door_nid = ""
            poi_obj: dict[str, Any] = {
                "id": str(p.get("id") or _short_id("p")),
                "name": name,
                "aliases": aliases,
                "category": cat,
                "x": round(px, 3),
                "y": round(py, 3),
                "node_id": nid,
            }
            if door_nid:
                poi_obj["door_node_id"] = door_nid
            if p.get("indoor") or cat == "room":
                poi_obj["indoor"] = True
            if p.get("building_id"):
                poi_obj["building_id"] = str(p["building_id"])
            if p.get("floor"):
                poi_obj["floor"] = str(p["floor"])
            m.pois.append(poi_obj)
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
    def load(cls, path: Path) -> "CampusMap":
        if not path.is_file():
            return cls(campus_id=path.stem)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ── query & editing helpers ──────────────────────────────────────────────
    def node_map(self) -> dict[str, dict[str, Any]]:
        return {n["id"]: n for n in self.nodes}

    def find_building_at(self, x: float, y: float) -> dict[str, Any] | None:
        for a in self.areas:
            if a.get("kind") == "building" and _point_in_polygon(x, y, a.get("polygon") or []):
                return a
        return None

    def find_nearest_node(
        self,
        x: float,
        y: float,
        max_dist_m: float = float("inf"),
        exclude: str | None = None,
        *,
        kinds: set[str] | frozenset[str] | None = None,
        exclude_kinds: set[str] | frozenset[str] | None = None,
    ) -> tuple[dict[str, Any] | None, float]:
        best = None
        best_d = max_dist_m
        for n in self.nodes:
            if exclude and n["id"] == exclude:
                continue
            nk = str(n.get("kind") or "waypoint")
            if kinds is not None and nk not in kinds:
                continue
            if exclude_kinds is not None and nk in exclude_kinds:
                continue
            d = math.hypot(n["x"] - x, n["y"] - y)
            if d < best_d:
                best = n
                best_d = d
        return best, best_d

    def add_node(
        self,
        x: float,
        y: float,
        kind: str = "waypoint",
        *,
        indoor: bool = False,
        building_id: str = "",
        floor: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        k = kind if kind in NODE_KINDS else "waypoint"
        node: dict[str, Any] = {
            "id": _short_id("n"),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "kind": k,
        }
        if indoor or k in ("corridor", "room_door", "room_nav"):
            node["indoor"] = True
        if building_id:
            node["building_id"] = building_id
        if floor:
            node["floor"] = floor
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
        road_class: str = "walkway",
        width_m: float = 3.5,
        weight: float = 1.0,
        bidirectional: bool = True,
    ) -> dict[str, Any] | None:
        if u == v or self.has_edge(u, v):
            return None
        nodes = self.node_map()
        if u not in nodes or v not in nodes:
            raise ValueError("edge endpoints must exist")
        rc = road_class if road_class in ROAD_CLASSES else "walkway"
        edge = {
            "id": _short_id("e"),
            "from": u,
            "to": v,
            "name": name,
            "road_class": rc,
            "width_m": max(1.0, float(width_m)),
            "weight": max(0.1, float(weight)),
            "bidirectional": bool(bidirectional),
        }
        self.edges.append(edge)
        return edge

    def add_area(self, name: str, kind: str, polygon: list[list[float]]) -> dict[str, Any]:
        if len(polygon) < 3:
            raise ValueError("area polygon requires at least 3 vertices")
        k = kind if kind in AREA_KINDS else "building"
        area = {
            "id": _short_id("a"),
            "name": name.strip(),
            "kind": k,
            "polygon": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in polygon],
        }
        self.areas.append(area)
        return area

    def add_poi(
        self,
        name: str,
        x: float,
        y: float,
        *,
        category: str = "landmark",
        aliases: list[str] | None = None,
        node_id: str = "",
        door_node_id: str = "",
        indoor: bool = False,
        building_id: str = "",
        floor: str = "",
    ) -> dict[str, Any]:
        clean = name.strip()
        if not clean:
            raise ValueError("POI name must be non-empty")
        if not node_id and self.nodes:
            nearest, dist = self.find_nearest_node(x, y, max_dist_m=15.0)
            if nearest is not None:
                node_id = nearest["id"]
        cat = category if category in POI_CATEGORIES else "landmark"
        poi: dict[str, Any] = {
            "id": _short_id("p"),
            "name": clean,
            "aliases": [a.strip() for a in (aliases or []) if a.strip()],
            "category": cat,
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "node_id": node_id,
        }
        if door_node_id:
            poi["door_node_id"] = door_node_id
        if indoor or cat == "room":
            poi["indoor"] = True
        if building_id:
            poi["building_id"] = building_id
        if floor:
            poi["floor"] = floor
        self.pois.append(poi)
        return poi

    def find_poi(self, query: str) -> dict[str, Any] | None:
        q = query.strip().lower()
        if not q:
            return None
        # Exact match on id, name, or alias first.
        for p in self.pois:
            if p["id"].lower() == q or p["name"].lower() == q:
                return p
            if any(a.lower() == q for a in p.get("aliases", [])):
                return p
        # Substring match fallback.
        for p in self.pois:
            if q in p["name"].lower() or any(q in a.lower() for a in p.get("aliases", [])):
                return p
        return None


class SurveyRecorder:
    """Turns handheld rtk_mapper player motion into clean road graphs or area polygons."""

    def __init__(
        self,
        node_spacing_m: float = 4.0,
        turn_threshold_deg: float = 25.0,
        merge_radius_m: float = 2.2,
    ) -> None:
        self.node_spacing_m = float(node_spacing_m)
        self.turn_threshold_rad = math.radians(turn_threshold_deg)
        self.merge_radius_m = float(merge_radius_m)
        self.active: bool = False
        self.mode: str = "road"  # "road" | "area" | "poi"
        self.road_class: str = "walkway"
        self.road_name: str = ""
        self.area_kind: str = "building"
        self.area_name: str = ""
        self.poi_category: str = "landmark"
        self.poi_name: str = ""
        self._last_node_id: str | None = None
        self._last_xy: tuple[float, float] | None = None
        self._last_heading: float | None = None
        self._area_draft: list[list[float]] = []
        self._last_mark_seq: int = 0
        self.events: list[str] = []

    def start(
        self,
        mode: str = "road",
        *,
        road_class: str = "walkway",
        road_name: str = "",
        area_kind: str = "building",
        area_name: str = "",
        poi_category: str = "landmark",
        poi_name: str = "",
    ) -> None:
        self.active = True
        self.mode = mode if mode in ("road", "area", "poi") else "road"
        self.road_class = road_class if road_class in ROAD_CLASSES else "walkway"
        self.road_name = road_name.strip()
        self.area_kind = area_kind if area_kind in AREA_KINDS else "building"
        self.area_name = area_name.strip()
        self.poi_category = poi_category if poi_category in POI_CATEGORIES else "landmark"
        self.poi_name = poi_name.strip()
        self._last_node_id = None
        self._last_xy = None
        self._last_heading = None
        if self.mode == "area":
            self._area_draft = []

    def stop(self) -> None:
        self.active = False
        self._last_node_id = None
        self._last_xy = None
        self._last_heading = None

    def finish_area(self, campus_map: CampusMap, name: str = "", kind: str = "") -> dict[str, Any] | None:
        if len(self._area_draft) < 3:
            return None
        area = campus_map.add_area(
            name or self.area_name or "新建区域",
            kind or self.area_kind,
            self._area_draft,
        )
        self._area_draft = []
        self._log(f"closed area '{area['name']}' ({area['kind']}, {len(area['polygon'])} pts)")
        return area

    def state_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "mode": self.mode,
            "road_class": self.road_class,
            "road_name": self.road_name,
            "area_kind": self.area_kind,
            "area_name": self.area_name,
            "area_draft": copy.deepcopy(self._area_draft),
            "last_node_id": self._last_node_id,
            "recent_events": list(self.events[-8:]),
        }

    def ingest_sample(
        self,
        campus_map: CampusMap,
        x: float,
        y: float,
        fix_type: str,
        *,
        require_rtk: bool = True,
        mark_event: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process one RTK sample from the player holding rtk_mapper."""
        if not self.active:
            return {"changed": False, "reason": "recorder inactive"}
        allowed_fixes = {"RTK_FIXED", "RTK_FLOAT"} if require_rtk else {"RTK_FIXED", "RTK_FLOAT", "SINGLE_GNSS"}
        if fix_type not in allowed_fixes:
            return {"changed": False, "reason": f"waiting for RTK fix (current={fix_type})"}

        changed = False
        auto_saved = False
        # Handle explicit right-click (mark) / Shift+right-click (close) from rtk_mapper item.
        if mark_event and int(mark_event.get("seq", 0)) > self._last_mark_seq:
            self._last_mark_seq = int(mark_event["seq"])
            action = str(mark_event.get("action") or "mark")
            mfix = mark_event.get("fix") or {}
            if action == "close":
                if self.mode == "area":
                    closed = self.finish_area(campus_map)
                    if closed is not None:
                        changed = True
                        auto_saved = True
            elif mfix.get("valid"):
                mx, my = float(mfix["east"]), float(mfix["north"])
                if self.mode == "area":
                    self._area_draft.append([round(mx, 3), round(my, 3)])
                    self._log(f"采建筑/地块角点 #{len(self._area_draft)} ({mx:.1f}, {my:.1f})")
                    changed = True
                elif self.mode == "poi":
                    label = self.poi_name or f"地标 ({mx:.0f},{my:.0f})"
                    poi = campus_map.add_poi(label, mx, my, category=self.poi_category)
                    self._log(f"标记地标「{poi['name']}」@ ({mx:.1f}, {my:.1f})")
                    changed = True
                    auto_saved = True
                else:
                    self._drop_road_node(campus_map, mx, my, force=True)
                    changed = True

        if self.mode == "road":
            if self._drop_road_node(campus_map, x, y, force=False):
                changed = True

        return {"changed": changed, "auto_saved": auto_saved, "reason": "ok"}

    def _drop_road_node(self, campus_map: CampusMap, x: float, y: float, *, force: bool) -> bool:
        if self._last_xy is None:
            near, dist = campus_map.find_nearest_node(x, y, max_dist_m=self.merge_radius_m)
            node = near if near is not None else campus_map.add_node(x, y, kind="waypoint")
            self._last_node_id = node["id"]
            self._last_xy = (node["x"], node["y"])
            self._log(f"started road at {node['id']} ({node['x']:.1f}, {node['y']:.1f})")
            return True

        lx, ly = self._last_xy
        dist = math.hypot(x - lx, y - ly)
        heading = math.atan2(y - ly, x - lx) if dist > 0.5 else self._last_heading
        turned = False
        if heading is not None and self._last_heading is not None and dist >= 1.8:
            diff = abs((heading - self._last_heading + math.pi) % (2 * math.pi) - math.pi)
            turned = diff >= self.turn_threshold_rad

        if not force and dist < self.node_spacing_m and not turned:
            return False

        near, _ = campus_map.find_nearest_node(
            x, y, max_dist_m=self.merge_radius_m, exclude=self._last_node_id
        )
        if near is not None:
            target = near
            if target.get("kind") == "waypoint":
                target["kind"] = "junction"
        else:
            target = campus_map.add_node(x, y, kind="waypoint")

        if self._last_node_id and self._last_node_id != target["id"]:
            campus_map.add_edge(
                self._last_node_id,
                target["id"],
                name=self.road_name,
                road_class=self.road_class,
                width_m=4.5 if self.road_class == "primary" else 3.0,
            )
        self._last_heading = heading
        self._last_node_id = target["id"]
        self._last_xy = (target["x"], target["y"])
        return True

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")
        if len(self.events) > 30:
            self.events = self.events[-30:]


class IndoorSurveyRecorder:
    """Turns handheld indoor_mapper odometry motion into indoor corridor graphs + paired room points.

    States:
      - "idle": not actively recording indoor corridors
      - "corridor": auto-dropping corridor breadcrumbs every corridor_spacing_m or sharp turn
      - "awaiting_room_nav": room_door dropped at doorway; corridor auto-drop paused until room_nav is marked
      - "in_room_paused": room_nav marked (or manually paused); stays paused until user clicks Resume Corridor
    """

    def __init__(
        self,
        corridor_spacing_m: float = 2.5,
        turn_threshold_deg: float = 25.0,
        merge_radius_m: float = 1.6,
        outdoor_snap_m: float = 6.0,
    ) -> None:
        self.corridor_spacing_m = float(corridor_spacing_m)
        self.turn_threshold_rad = math.radians(turn_threshold_deg)
        self.merge_radius_m = float(merge_radius_m)
        self.outdoor_snap_m = float(outdoor_snap_m)
        self.state: str = "idle"
        self.building_id: str = ""
        self.floor: str = "1F"
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
            "building_id": self.building_id,
            "floor": self.floor,
            "room_name": self.room_name,
            "entrance_node_id": self.entrance_node_id,
            "last_corridor_node_id": self.last_corridor_node_id,
            "pending_door_node_id": self.pending_door_node_id,
            "last_door_node_id": self.last_door_node_id,
            "recent_events": list(self.events[-8:]),
        }

    def anchor_entrance(
        self,
        campus_map: CampusMap,
        x: float,
        y: float,
        *,
        floor: str = "",
        building_id: str = "",
        label: str = "",
    ) -> dict[str, Any]:
        if floor:
            self.floor = floor.strip()
        if building_id:
            self.building_id = building_id.strip()
        else:
            bldg = campus_map.find_building_at(x, y)
            if bldg is not None:
                self.building_id = bldg["id"]

        # Reuse an existing entrance node if right on top of one; otherwise create a fresh entrance node.
        near_ent, _ = campus_map.find_nearest_node(x, y, max_dist_m=1.5, kinds={"entrance"})
        if near_ent is not None:
            node = near_ent
        else:
            node = campus_map.add_node(
                x,
                y,
                kind="entrance",
                indoor=True,
                building_id=self.building_id,
                floor=self.floor,
                label=label or "建筑出口",
            )
            # Connect seamlessly to nearest outdoor road node if within outdoor_snap_m.
            near_out, _ = campus_map.find_nearest_node(
                x,
                y,
                max_dist_m=self.outdoor_snap_m,
                exclude=node["id"],
                kinds={"waypoint", "junction", "entrance"},
            )
            if near_out is not None:
                campus_map.add_edge(
                    near_out["id"],
                    node["id"],
                    name="建筑出入口",
                    road_class="walkway",
                    width_m=2.5,
                )

        self.entrance_node_id = node["id"]
        self.last_corridor_node_id = node["id"]
        self.last_corridor_xy = (node["x"], node["y"])
        self.last_heading = None
        self.pending_door_node_id = None
        self.state = "corridor"
        self._log(f"设建筑出口点 {node['id']} ({node['x']:.1f}, {node['y']:.1f})，开始走廊自动打点")
        return node

    def mark_room_door(
        self,
        campus_map: CampusMap,
        x: float,
        y: float,
        *,
        room_name: str = "",
    ) -> dict[str, Any]:
        if room_name.strip():
            self.room_name = room_name.strip()
        bldg = campus_map.find_building_at(x, y)
        bid = self.building_id or (bldg["id"] if bldg else "")
        door_label = f"{self.room_name}门口" if self.room_name else "房间门口"
        door = campus_map.add_node(
            x,
            y,
            kind="room_door",
            indoor=True,
            building_id=bid,
            floor=self.floor,
            label=door_label,
        )
        # Always connect to the geometrically nearest corridor/entrance node (never jump to a distant dead-end).
        near_c, _ = campus_map.find_nearest_node(
            x,
            y,
            max_dist_m=15.0,
            exclude=door["id"],
            kinds={"corridor", "entrance"},
        )
        if near_c is None:
            near_c, _ = campus_map.find_nearest_node(
                x,
                y,
                max_dist_m=15.0,
                exclude=door["id"],
                kinds={"room_door"},
            )
        if near_c is not None and near_c["id"] != door["id"]:
            campus_map.add_edge(near_c["id"], door["id"], road_class="corridor", width_m=2.0)

        self.pending_door_node_id = door["id"]
        self.last_door_node_id = door["id"]
        self.last_corridor_node_id = door["id"]
        self.last_corridor_xy = (door["x"], door["y"])
        self.last_heading = None
        self.state = "awaiting_room_nav"
        self._log(f"标房间门口点 {door['id']} ({door['x']:.1f}, {door['y']:.1f})，已暂停走廊打点")
        return door

    def mark_room_nav(
        self,
        campus_map: CampusMap,
        x: float,
        y: float,
        *,
        room_name: str = "",
    ) -> dict[str, Any]:
        clean_name = room_name.strip() or self.room_name.strip()
        if not clean_name:
            room_count = sum(1 for p in campus_map.pois if p.get("category") == "room")
            clean_name = f"房间{room_count + 1}"
        bldg = campus_map.find_building_at(x, y)
        bid = self.building_id or (bldg["id"] if bldg else "")

        nodes = campus_map.node_map()
        door_id = self.pending_door_node_id if (self.pending_door_node_id in nodes) else None
        if door_id is None:
            near_d, _ = campus_map.find_nearest_node(
                x,
                y,
                max_dist_m=15.0,
                kinds={"room_door", "corridor", "entrance"},
            )
            door_id = near_d["id"] if near_d is not None else ""

        nav_node = campus_map.add_node(
            x,
            y,
            kind="room_nav",
            indoor=True,
            building_id=bid,
            floor=self.floor,
            label=clean_name,
        )
        if door_id and door_id in nodes:
            campus_map.add_edge(
                door_id,
                nav_node["id"],
                name=f"{clean_name}入口",
                road_class="room_entry",
                width_m=1.5,
            )
            door_node = nodes[door_id]
            if not door_node.get("label") or door_node.get("label") == "房间门口":
                door_node["label"] = f"{clean_name}门口"

        poi = campus_map.add_poi(
            clean_name,
            x,
            y,
            category="room",
            node_id=nav_node["id"],
            door_node_id=door_id or "",
            indoor=True,
            building_id=bid,
            floor=self.floor,
        )
        self.pending_door_node_id = None
        if door_id:
            self.last_door_node_id = door_id
        self.room_name = ""
        self.state = "in_room_paused"
        self._log(f"标房间导航点「{clean_name}」({nav_node['x']:.1f}, {nav_node['y']:.1f})，保持暂停待手动恢复走廊")
        return {"poi": poi, "nav_node": nav_node, "door_node_id": door_id or ""}

    def pause_corridor(self) -> dict[str, Any]:
        if self.state != "idle":
            self.state = "in_room_paused"
            self._log("已暂停走廊自动打点")
        return self.state_dict()

    def resume_corridor(
        self,
        campus_map: CampusMap,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        nodes = campus_map.node_map()
        anchor_id = None
        if self.last_door_node_id and self.last_door_node_id in nodes:
            anchor_id = self.last_door_node_id
        elif self.last_corridor_node_id and self.last_corridor_node_id in nodes:
            anchor_id = self.last_corridor_node_id
        if x is not None and y is not None:
            near_c, dist = campus_map.find_nearest_node(
                x, y, max_dist_m=4.0, kinds={"corridor", "room_door", "entrance"}
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
            n = campus_map.add_node(
                x, y, kind="corridor", indoor=True, building_id=self.building_id, floor=self.floor
            )
            self.last_corridor_node_id = n["id"]
            self.last_corridor_xy = (n["x"], n["y"])
        self.pending_door_node_id = None
        self.last_heading = None
        self.state = "corridor"
        self._log("已手动恢复走廊自动打点")
        return self.state_dict()

    def ingest_sample(
        self,
        campus_map: CampusMap,
        x: float,
        y: float,
        *,
        anchored: bool,
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
            if action == "anchor_entrance":
                self.anchor_entrance(campus_map, mx, my)
                changed = True
                auto_saved = True
            elif action == "mark_step":
                if self.state == "awaiting_room_nav":
                    self.mark_room_nav(campus_map, mx, my)
                else:
                    self.mark_room_door(campus_map, mx, my)
                changed = True
                auto_saved = True

        if self.state == "corridor" and (anchored or self.entrance_node_id is not None):
            if self._drop_corridor_node(campus_map, x, y):
                changed = True

        return {"changed": changed, "auto_saved": auto_saved, "state": self.state}

    def _drop_corridor_node(self, campus_map: CampusMap, x: float, y: float) -> bool:
        if self.last_corridor_xy is None or not self.last_corridor_node_id:
            near, _ = campus_map.find_nearest_node(
                x, y, max_dist_m=self.merge_radius_m, kinds={"corridor", "room_door", "entrance"}
            )
            node = (
                near
                if near is not None
                else campus_map.add_node(
                    x, y, kind="corridor", indoor=True, building_id=self.building_id, floor=self.floor
                )
            )
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

        near, _ = campus_map.find_nearest_node(
            x,
            y,
            max_dist_m=self.merge_radius_m,
            exclude=self.last_corridor_node_id,
            kinds={"corridor", "room_door", "entrance"},
        )
        if near is not None:
            target = near
        else:
            bldg = campus_map.find_building_at(x, y)
            bid = self.building_id or (bldg["id"] if bldg else "")
            target = campus_map.add_node(
                x, y, kind="corridor", indoor=True, building_id=bid, floor=self.floor
            )

        if self.last_corridor_node_id != target["id"]:
            campus_map.add_edge(self.last_corridor_node_id, target["id"], road_class="corridor", width_m=2.0)
        self.last_heading = heading
        self.last_corridor_node_id = target["id"]
        self.last_corridor_xy = (target["x"], target["y"])
        return True

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{stamp}] {msg}")
        if len(self.events) > 30:
            self.events = self.events[-30:]
