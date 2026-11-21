# SPDX-License-Identifier: MulanPSL-2.0
"""Standalone campus map editor & handheld RTK surveyor (no rbnx/ROS2/Atlas needed).

Connects directly to the Minecraft dedicated server's loopback RTK endpoint
(127.0.0.1:25577) and serves the Google Maps-style editor on port 8093 so a
player can survey and edit campus maps without booting the full Robonix stack.
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

from campus_nav_rbnx import params
from campus_nav_rbnx.campus_map import CampusMap, SurveyRecorder
from campus_nav_rbnx.indoor_map import IndoorFloorMap, IndoorFloorSurveyRecorder, IndoorMapStore
from campus_nav_rbnx.mapper_client import ServerMapperClient
from campus_nav_rbnx.webui import CampusWebUI

logging.basicConfig(
    level=logging.INFO,
    format="[campus_mapper] %(levelname)s %(message)s",
)
log = logging.getLogger("campus_nav_rbnx.standalone")

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class StandaloneCampusMapper:
    def __init__(self, cfg: params.CampusNavConfig) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.cfg.map_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.cfg.map_file
        self.campus_map = CampusMap.load(self.cfg.map_dir / self.active_file)
        self.indoor_store = IndoorMapStore(self.cfg.map_dir / "indoor")
        self.active_indoor_id: str = ""  # "" = outdoor overview; "<bldg>__<floor>" = indoor floor view
        self.recorder = SurveyRecorder()
        self.indoor_recorder = IndoorFloorSurveyRecorder()
        self.mapper_client = ServerMapperClient(cfg.server_host, cfg.server_port)
        self._stations: list[dict[str, Any]] = []
        self._survey_holders: list[dict[str, Any]] = []
        self._indoor_holders: list[dict[str, Any]] = []
        self._survey_error: str = ""
        self._mapping_ui_open: bool = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._webui = CampusWebUI(
            cfg.webui_host,
            cfg.webui_port,
            self._handlers(),
        )

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="campus-mapper-poll", daemon=True
        )
        self._thread.start()
        self._webui.start()
        log.info(
            "standalone campus mapper ready at %s (map=%s, server=%s:%d)",
            self._webui.url,
            self.cfg.map_dir / self.active_file,
            self.cfg.server_host,
            self.cfg.server_port,
        )

    def stop(self) -> None:
        self._stop.set()
        self._webui.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                res = self.mapper_client.poll_mapper(
                    dimension=self.cfg.dimension,
                    max_range_m=self.cfg.rtk_max_range_m,
                    noise_enabled=self.cfg.rtk_noise_enabled,
                )
                holders = list(res.get("holders") or [])
                indoor_holders = list(res.get("indoor_holders") or [])
                stations = list(res.get("stations") or [])
                last_mark = res.get("last_mark")
                last_indoor_mark = res.get("last_indoor_mark")
                with self.lock:
                    self._stations = stations
                    self._survey_holders = holders
                    self._indoor_holders = indoor_holders
                    self._survey_error = ""
                    if self.recorder.active and holders:
                        fix = holders[0].get("fix") or {}
                        if fix.get("valid"):
                            out = self.recorder.ingest_sample(
                                self.campus_map,
                                float(fix["east"]),
                                float(fix["north"]),
                                str(fix.get("fix_type") or "NO_FIX"),
                                require_rtk=self.cfg.require_base_station_for_mapping,
                                mark_event=last_mark,
                            )
                            if out.get("auto_saved"):
                                self.campus_map.save(self.cfg.map_dir / self.active_file)
                    if indoor_holders:
                        ipose = indoor_holders[0].get("pose") or {}
                        fm = self._resolve_active_floor_map(
                            float(ipose.get("east", 0.0)),
                            float(ipose.get("north", 0.0)),
                            ipose.get("up"),
                        )
                        iout = self.indoor_recorder.ingest_sample(
                            self.campus_map,
                            fm,
                            float(ipose.get("east", 0.0)),
                            float(ipose.get("north", 0.0)),
                            anchored=bool(ipose.get("anchored", False)),
                            elevation_y=ipose.get("up"),
                            mark_event=last_indoor_mark,
                        )
                        if iout.get("changed") or iout.get("auto_saved"):
                            self.indoor_store.save(fm)
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self._survey_error = str(exc)
            self._stop.wait(0.35)

    def _handlers(self) -> dict[str, Any]:
        return {
            "GET:api/state": self._api_state,
            "POST:api/map/replace": self._api_map_replace,
            "POST:api/map/save": self._api_map_save,
            "POST:api/map/load": self._api_map_load,
            "POST:api/map/new": self._api_map_new,
            "POST:api/map/reset": self._api_map_reset,
            "POST:api/survey/mode": self._api_survey_mode,
            "POST:api/survey/start": self._api_survey_start,
            "POST:api/survey/stop": self._api_survey_stop,
            "POST:api/survey/finish_area": self._api_survey_finish_area,
            "POST:api/survey/drop_poi": self._api_survey_drop_poi,
            "POST:api/indoor/select_floor": self._api_indoor_select_floor,
            "POST:api/indoor/replace": self._api_indoor_replace,
            "POST:api/indoor/reset_floor": self._api_indoor_reset_floor,
            "POST:api/indoor/anchor_entrance": self._api_indoor_anchor_entrance,
            "POST:api/indoor/mark_room_door": self._api_indoor_mark_room_door,
            "POST:api/indoor/mark_room_nav": self._api_indoor_mark_room_nav,
            "POST:api/indoor/corridor_toggle": self._api_indoor_corridor_toggle,
            "POST:api/navigate": self._api_nav_unavailable,
            "POST:api/navigate/cancel": self._api_nav_unavailable,
            "POST:api/navigate/clear_blocked": lambda _: {"ok": True},
            "POST:api/relocalize": self._api_nav_unavailable,
        }

    def _api_nav_unavailable(self, _: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": False,
            "detail": "当前为独立建图模式（未启动 rbnx）；请启动室外 profile 后再发起机器人导航",
        }

    def _api_state(self, _: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            files = sorted(p.name for p in self.cfg.map_dir.glob("*.json"))
            if self.active_file not in files:
                files.append(self.active_file)
            active_fm = self.indoor_store.get(self.active_indoor_id) if self.active_indoor_id else None
            return {
                "ok": True,
                "standalone": True,
                "active_file": self.active_file,
                "map_files": files,
                "map": self.campus_map.to_dict(),
                "active_indoor_id": self.active_indoor_id,
                "active_indoor_map": active_fm.to_dict() if active_fm is not None else None,
                "indoor_maps": self.indoor_store.list_summaries(),
                "all_rooms": self.indoor_store.all_rooms(),
                "robot": {
                    "has_pose": False,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw": 0.0,
                    "rtk_status": {
                        "valid": False,
                        "fix_type": "STANDALONE",
                        "num_sats": 0,
                        "sky_visibility": 0.0,
                        "base_distance_m": -1.0,
                    },
                },
                "local_window": None,
                "stations": list(self._stations),
                "rtk_max_range_m": self.cfg.rtk_max_range_m,
                "survey": {
                    "ui_open": self._mapping_ui_open,
                    "holders": list(self._survey_holders),
                    "indoor_holders": list(self._indoor_holders),
                    "error": self._survey_error,
                    "recorder": self.recorder.state_dict(),
                    "indoor_recorder": self.indoor_recorder.state_dict(),
                },
                "navigation": {
                    "run_id": "",
                    "state": "IDLE",
                    "place": "",
                    "remaining_m": 0.0,
                    "current_segment": 0,
                    "total_segments": 0,
                    "route_points": [],
                    "blocked_edges": [],
                    "detail": "独立建图模式（无需启动 rbnx）",
                },
            }

    def _api_map_replace(self, body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("map")
        if not isinstance(raw, dict):
            return {"ok": False, "detail": "missing map object"}
        new_map = CampusMap.from_dict(raw)
        with self.lock:
            self.campus_map = new_map
            if body.get("save", True):
                self.campus_map.save(self.cfg.map_dir / self.active_file)
        return {"ok": True, "detail": f"updated {self.active_file}"}

    def _api_map_save(self, body: dict[str, Any]) -> dict[str, Any]:
        fname = str(body.get("file") or self.active_file).strip()
        if not fname.endswith(".json"):
            fname += ".json"
        if "/" in fname or "\\" in fname or ".." in fname:
            return {"ok": False, "detail": "invalid filename"}
        with self.lock:
            self.active_file = fname
            self.campus_map.campus_id = Path(fname).stem
            self.campus_map.save(self.cfg.map_dir / fname)
        return {"ok": True, "file": fname, "detail": f"saved {fname}"}

    def _api_map_load(self, body: dict[str, Any]) -> dict[str, Any]:
        fname = str(body.get("file") or "").strip()
        if not fname or "/" in fname or "\\" in fname or ".." in fname:
            return {"ok": False, "detail": "invalid filename"}
        loaded = CampusMap.load(self.cfg.map_dir / fname)
        with self.lock:
            self.active_file = fname
            self.campus_map = loaded
        return {"ok": True, "file": fname, "detail": f"loaded {fname}"}

    def _api_map_new(self, body: dict[str, Any]) -> dict[str, Any]:
        fname = str(body.get("file") or "campus_new.json").strip()
        if not fname.endswith(".json"):
            fname += ".json"
        if "/" in fname or "\\" in fname or ".." in fname:
            return {"ok": False, "detail": "invalid filename"}
        fresh = CampusMap(campus_id=Path(fname).stem, name=str(body.get("name") or Path(fname).stem))
        with self.lock:
            self.active_file = fname
            self.campus_map = fresh
            self.campus_map.save(self.cfg.map_dir / fname)
        return {"ok": True, "file": fname, "detail": f"created {fname}"}

    def _api_map_reset(self, _: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.recorder.stop()
            self.recorder._area_draft = []
            self.indoor_recorder.reset()
            self.campus_map = CampusMap(
                campus_id=self.campus_map.campus_id,
                name=self.campus_map.name,
            )
            self.campus_map.save(self.cfg.map_dir / self.active_file)
        return {"ok": True, "file": self.active_file, "detail": f"已重置地图 {self.active_file}"}

    def _api_survey_mode(self, body: dict[str, Any]) -> dict[str, Any]:
        open_flag = bool(body.get("open", True))
        with self.lock:
            self._mapping_ui_open = open_flag
            if not open_flag:
                self.recorder.stop()
        return {"ok": True, "ui_open": open_flag}

    def _api_survey_start(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._mapping_ui_open = True
            self.recorder.start(
                str(body.get("mode") or "road"),
                road_class=str(body.get("road_class") or "walkway"),
                road_name=str(body.get("road_name") or ""),
                area_kind=str(body.get("area_kind") or "building"),
                area_name=str(body.get("area_name") or ""),
                poi_category=str(body.get("poi_category") or "landmark"),
                poi_name=str(body.get("poi_name") or ""),
            )
            return {"ok": True, "recorder": self.recorder.state_dict()}

    def _api_survey_stop(self, _: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.recorder.stop()
            self.campus_map.save(self.cfg.map_dir / self.active_file)
            return {"ok": True, "recorder": self.recorder.state_dict()}

    def _api_survey_finish_area(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            area = self.recorder.finish_area(
                self.campus_map,
                name=str(body.get("name") or ""),
                kind=str(body.get("kind") or ""),
            )
            if area is None:
                return {"ok": False, "detail": "需要至少 3 个角点才能闭合建筑/地块"}
            self.campus_map.save(self.cfg.map_dir / self.active_file)
            return {"ok": True, "area": area}

    def _api_survey_drop_poi(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name") or "").strip()
        if not name:
            return {"ok": False, "detail": "POI name is required"}
        cat = str(body.get("category") or "landmark")
        with self.lock:
            if "x" in body and "y" in body:
                px, py = float(body["x"]), float(body["y"])
            elif self._survey_holders and (self._survey_holders[0].get("fix") or {}).get("valid"):
                fix = self._survey_holders[0]["fix"]
                px, py = float(fix["east"]), float(fix["north"])
            else:
                return {"ok": False, "detail": "未检测到手持 rtk_mapper 且定位有效的玩家"}
            poi = self.campus_map.add_poi(name, px, py, category=cat)
            self.campus_map.save(self.cfg.map_dir / self.active_file)
            return {"ok": True, "poi": poi}

    def _current_indoor_xy(self, body: dict[str, Any]) -> tuple[float, float] | None:
        if "x" in body and "y" in body and body["x"] is not None and body["y"] is not None:
            return float(body["x"]), float(body["y"])
        if self._indoor_holders:
            p = self._indoor_holders[0].get("pose") or {}
            return float(p.get("east", 0.0)), float(p.get("north", 0.0))
        return None

    def _resolve_active_floor_map(
        self,
        x: float,
        y: float,
        elevation_y: Any = None,
        *,
        building_id: str = "",
        building_name: str = "",
        floor: str = "",
    ) -> IndoorFloorMap:
        if building_id:
            bname = building_name
            if not bname:
                for a in self.campus_map.areas:
                    if a.get("id") == building_id:
                        bname = str(a.get("name") or building_id)
                        break
            fm = self.indoor_store.get_or_create(
                building_id, bname or building_id, floor or "1F",
                elevation_y=float(elevation_y) if elevation_y is not None else None,
            )
            self.active_indoor_id = fm.indoor_id
            return fm
        if self.active_indoor_id:
            existing = self.indoor_store.get(self.active_indoor_id)
            if existing is not None:
                return existing
        bldg = self.campus_map.find_building_at(x, y)
        bid = bldg["id"] if bldg is not None else "building_default"
        bname = (bldg.get("name") or bid) if bldg is not None else "默认建筑"
        fm = self.indoor_store.get_or_create(
            bid, bname, floor or "1F",
            elevation_y=float(elevation_y) if elevation_y is not None else None,
        )
        self.active_indoor_id = fm.indoor_id
        return fm

    def _api_indoor_select_floor(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            iid = str(body.get("indoor_id") or "").strip()
            building_id = str(body.get("building_id") or "").strip()
            floor = str(body.get("floor") or "").strip().upper()
            if not iid and not building_id:
                self.active_indoor_id = ""
                return {"ok": True, "active_indoor_id": "", "detail": "已切回室外园区全景"}
            if iid:
                fm = self.indoor_store.get(iid)
                if fm is None:
                    return {"ok": False, "detail": f"未找到室内楼层地图: {iid}"}
                self.active_indoor_id = fm.indoor_id
                return {"ok": True, "active_indoor_id": fm.indoor_id, "indoor_map": fm.to_dict(), "detail": f"已切换到 {fm.building_name} {fm.floor}"}
            bname = str(body.get("building_name") or "").strip()
            if not bname:
                for a in self.campus_map.areas:
                    if a.get("id") == building_id:
                        bname = str(a.get("name") or building_id)
                        break
            fm = self.indoor_store.get_or_create(building_id, bname or building_id, floor or "1F")
            self.active_indoor_id = fm.indoor_id
            return {"ok": True, "active_indoor_id": fm.indoor_id, "indoor_map": fm.to_dict(), "detail": f"已切换到 {fm.building_name} {fm.floor}"}

    def _api_indoor_replace(self, body: dict[str, Any]) -> dict[str, Any]:
        raw = body.get("indoor_map")
        if not isinstance(raw, dict):
            return {"ok": False, "detail": "missing indoor_map object"}
        with self.lock:
            fm = self.indoor_store.replace(raw, save=bool(body.get("save", True)))
            self.active_indoor_id = fm.indoor_id
        return {"ok": True, "indoor_id": fm.indoor_id, "detail": f"已保存 {fm.building_name} {fm.floor}"}

    def _api_indoor_reset_floor(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            iid = str(body.get("indoor_id") or self.active_indoor_id).strip()
            if not iid:
                return {"ok": False, "detail": "当前未选中室内楼层地图"}
            self.indoor_recorder.reset()
            fm = self.indoor_store.reset_floor(iid)
            if fm is None:
                return {"ok": False, "detail": f"未找到楼层 {iid}"}
        return {"ok": True, "indoor_id": fm.indoor_id, "detail": f"已重置室内楼层 {fm.building_name} {fm.floor}"}

    def _api_indoor_anchor_entrance(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            elev = body.get("up")
            if elev is None and self._indoor_holders:
                elev = (self._indoor_holders[0].get("pose") or {}).get("up")
            fm = self._resolve_active_floor_map(
                xy[0], xy[1], elev,
                building_id=str(body.get("building_id") or ""),
                building_name=str(body.get("building_name") or ""),
                floor=str(body.get("floor") or "1F"),
            )
            node = self.indoor_recorder.anchor_entrance(
                self.campus_map,
                fm,
                xy[0],
                xy[1],
                elevation_y=float(elev) if elev is not None else None,
                label=str(body.get("label") or ""),
            )
            self.indoor_store.save(fm)
            return {
                "ok": True,
                "indoor_id": fm.indoor_id,
                "node": node,
                "indoor_recorder": self.indoor_recorder.state_dict(),
            }

    def _api_indoor_mark_room_door(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            fm = self._resolve_active_floor_map(
                xy[0], xy[1],
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            door = self.indoor_recorder.mark_room_door(
                fm, xy[0], xy[1], room_name=str(body.get("room_name") or "")
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, "door": door, "indoor_recorder": self.indoor_recorder.state_dict()}

    def _api_indoor_mark_room_nav(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            fm = self._resolve_active_floor_map(
                xy[0], xy[1],
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            out = self.indoor_recorder.mark_room_nav(
                fm, xy[0], xy[1], room_name=str(body.get("room_name") or "")
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, **out, "indoor_recorder": self.indoor_recorder.state_dict()}

    def _api_indoor_corridor_toggle(self, body: dict[str, Any]) -> dict[str, Any]:
        action = str(body.get("action") or "").strip().lower()
        with self.lock:
            if action == "stop":
                self.indoor_recorder.reset()
                return {"ok": True, "indoor_recorder": self.indoor_recorder.state_dict()}
            if action == "pause" or (not action and self.indoor_recorder.state == "corridor"):
                st = self.indoor_recorder.pause_corridor()
                return {"ok": True, "indoor_recorder": st}
            xy = self._current_indoor_xy(body)
            fm = self._resolve_active_floor_map(
                xy[0] if xy else 0.0,
                xy[1] if xy else 0.0,
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            st = self.indoor_recorder.resume_corridor(
                fm, xy[0] if xy else None, xy[1] if xy else None
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, "indoor_recorder": st}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone campus RTK mapper & Web UI (no rbnx required)")
    parser.add_argument("--host", default="127.0.0.1", help="Web UI bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8093, help="Web UI port (default: 8093)")
    parser.add_argument("--server-host", default="127.0.0.1", help="Minecraft server RTK host")
    parser.add_argument("--server-port", type=int, default=25577, help="Minecraft server RTK port")
    parser.add_argument("--map-file", default="campus_default.json", help="Campus map filename in maps/")
    args = parser.parse_args()

    cfg = params.from_config(
        {
            "webui_host": args.host,
            "webui_port": args.port,
            "server_host": args.server_host,
            "server_port": args.server_port,
            "map_file": args.map_file,
        },
        PACKAGE_ROOT,
    )
    app = StandaloneCampusMapper(cfg)
    stop_ev = threading.Event()

    def _sig(_signum, _frame):
        stop_ev.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    app.start()
    try:
        while not stop_ev.is_set():
            time.sleep(0.3)
    finally:
        app.stop()


if __name__ == "__main__":
    main()
