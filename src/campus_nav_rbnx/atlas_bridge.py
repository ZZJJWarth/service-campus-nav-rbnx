# SPDX-License-Identifier: MulanPSL-2.0
"""Atlas & ROS2 bridge for service-campus-nav-rbnx."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import outdoor_navigation_mcp
from robonix_api import Err, Ok, Service
from urllib.request import ProxyHandler, Request, build_opener

from campus_nav_rbnx import params
from campus_nav_rbnx.campus_map import CampusMap, SurveyRecorder
from campus_nav_rbnx.door_controller import DoorInteractor
from campus_nav_rbnx.elevator_client import ElevatorClient, MultiElevatorManager
from campus_nav_rbnx.executive import HierarchicalNavExecutive, LocalCostGridSnapshot
from campus_nav_rbnx.fiducial_localizer import VisualFiducialLocalizer, VisualRelocalizerFilter
from campus_nav_rbnx.global_planner import GlobalPlanner
from campus_nav_rbnx.indoor_map import IndoorFloorMap, IndoorFloorSurveyRecorder, IndoorMapStore
from campus_nav_rbnx.mapper_client import ServerMapperClient
from campus_nav_rbnx.rtk_fusion import RtkOdomFusion
from campus_nav_rbnx.webui import CampusWebUI

logging.basicConfig(
    level=os.environ.get("CAMPUS_NAV_LOG_LEVEL", "INFO").upper(),
    format="[campus_nav] %(levelname)s %(message)s",
)
log = logging.getLogger("campus_nav_rbnx.atlas_bridge")

CAP_ID = os.environ.get("ROBONIX_CAPABILITY_ID", "campus_nav")
NAMESPACE = "robonix/service/campus_nav"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]

campus_nav = Service(id=CAP_ID, namespace=NAMESPACE)


def _quat_yaw(q: Any) -> float:
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CampusNavRuntime:
    def __init__(self, cfg: params.CampusNavConfig) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.cfg.map_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.cfg.map_file
        self.campus_map = CampusMap.load(self.cfg.map_dir / self.active_file)
        self.indoor_store = IndoorMapStore(self.cfg.map_dir / "indoor")
        self.active_indoor_id: str = ""
        self.recorder = SurveyRecorder()
        self.indoor_recorder = IndoorFloorSurveyRecorder()
        self.fusion = RtkOdomFusion(require_global_lock=cfg.require_global_lock)
        self.fiducial_localizer = VisualFiducialLocalizer(default_size_m=cfg.fiducial_size_m)
        self.fiducial_filter = VisualRelocalizerFilter(min_consecutive=2)
        self._last_rgb_proc_t: float = 0.0
        self._last_vis_log_key: tuple[int, str] = (-1, "")
        self._last_vis_log_t: float = 0.0
        self._last_vis_log_xy: tuple[float, float] | None = None
        self._cmd_vel_pub: Any = None
        self._reloc_active: bool = False
        self._reloc_thread: threading.Thread | None = None
        self._reloc_stop = threading.Event()
        self._loc_paused: bool = False
        self._boot_loc_thread: threading.Thread | None = None
        self._reloc_state: dict[str, Any] = {
            "active": False,
            "phase": "IDLE",
            "detail": "",
            "hops_done": 0,
        }
        self.planner = GlobalPlanner(blocked_edge_ttl_s=cfg.blocked_edge_ttl_s)
        self.mapper_client = ServerMapperClient(cfg.server_host, cfg.server_port)
        legacy_elevator: ElevatorClient | None = None
        if cfg.elevator_config_file:
            ep = Path(cfg.elevator_config_file)
            if ep.is_file():
                try:
                    legacy_elevator = ElevatorClient.from_config_file(
                        ep,
                        host=cfg.server_host,
                        port=cfg.server_port if cfg.server_port > 0 else None,
                    )
                    log.info(
                        "loaded legacy elevator controller config %s (%d floors)",
                        ep,
                        len(legacy_elevator.floors),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to load elevator config %s: %s", ep, exc)
            else:
                log.warning("elevator_config_file does not exist: %s", ep)
        self.elevator_client = MultiElevatorManager(
            host=cfg.server_host,
            port=cfg.server_port,
            dimension=cfg.dimension,
            legacy=legacy_elevator,
        )
        self._cost_grid: LocalCostGridSnapshot | None = None
        self._stations: list[dict[str, Any]] = []
        self._survey_holders: list[dict[str, Any]] = []
        self._indoor_holders: list[dict[str, Any]] = []
        self._survey_error: str = ""
        self._mapping_ui_open: bool = False
        self._stop = threading.Event()
        self._survey_thread: threading.Thread | None = None
        self._ros_node: Any = None
        self._executor: Any = None
        self._ros_thread: threading.Thread | None = None
        self._webui: CampusWebUI | None = None
        self.door_interactor = DoorInteractor(
            host=os.environ.get("RBNX_MINECRAFT_MOD_HOST", "127.0.0.1"),
            port=int(os.environ.get("RBNX_MINECRAFT_MOD_PORT", "25576")),
        )
        self.executive = HierarchicalNavExecutive(
            cfg=cfg,
            map_provider=self.get_map,
            fusion=self.fusion,
            planner=self.planner,
            cost_grid_provider=self.get_cost_grid,
            indoor_store_provider=lambda: self.indoor_store,
            elevator_client=self.elevator_client,
            door_interactor=self.door_interactor,
            visual_object_url=os.environ.get("ROBONIX_VISUAL_OBJECT_URL", "http://127.0.0.1:8094"),
            reset_local_map_fn=self.reset_local_map,
            on_floor_changed=self._on_floor_changed,
        )

    def reset_local_map(self) -> bool:
        """Ask service-local-mapping-rbnx to reset its rolling costmap (e.g. on floor switch)."""
        base = (self.cfg.local_mapping_url or "").strip().rstrip("/")
        if not base:
            return False
        url = f"{base}/api/map/reset"
        opener = build_opener(ProxyHandler({}))
        req = Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
        with opener.open(req, timeout=3.0) as resp:
            body = json.loads(resp.read(16384).decode("utf-8"))
        ok = bool(body.get("ok", False))
        log.info("requested local_mapping reset (%s): ok=%s", url, ok)
        return ok

    def _on_floor_changed(self, indoor_id: str, building_name: str, floor: str) -> None:
        with self.lock:
            self.fusion.set_indoor_floor(indoor_id, building_name, floor)
            if indoor_id:
                self.active_indoor_id = indoor_id

    def _infer_start_indoor_id(self) -> str:
        """Determine robot's starting indoor floor from visual lock or building+elevation_y."""
        with self.lock:
            if self.cfg.require_global_lock and not self.fusion.initialized_from_rtk:
                return ""
            if self.fusion.current_indoor_id:
                return self.fusion.current_indoor_id
            # Outdoor RTK_FIXED means robot is outdoors, not inside any floor map.
            if (
                self.fusion.rtk_status.get("valid")
                and self.fusion.rtk_status.get("fix_type") == "RTK_FIXED"
                and self.fusion.rtk_wall > 0.0
                and (time.time() - self.fusion.rtk_wall) < 3.0
            ):
                return ""
            pose = self.fusion.campus_pose()
            oz = self.fusion.odom_z
            if pose is None or oz is None:
                return ""
            bldg = self.campus_map.find_building_at(pose[0], pose[1])
            if bldg is None:
                return ""
            bid = bldg["id"]
            best_iid = ""
            best_dz = 2.2
            for fm in self.indoor_store.all_maps():
                if fm.building_id != bid or fm.elevation_y is None:
                    continue
                dz = abs(float(oz) - float(fm.elevation_y))
                if dz < best_dz:
                    best_dz = dz
                    best_iid = fm.indoor_id
            return best_iid

    def get_map(self) -> CampusMap:
        with self.lock:
            return self.campus_map

    def get_cost_grid(self) -> LocalCostGridSnapshot | None:
        with self.lock:
            return self._cost_grid

    def start(self) -> None:
        self._start_ros()
        self._stop.clear()
        self._survey_thread = threading.Thread(
            target=self._survey_loop, name="campus-nav-survey", daemon=True
        )
        self._survey_thread.start()
        if self.cfg.require_global_lock and self.cfg.auto_relocalize_on_boot and self.cfg.visual_reloc_enabled:
            self._boot_loc_thread = threading.Thread(
                target=self._boot_localize_loop, name="campus-boot-localize", daemon=True
            )
            self._boot_loc_thread.start()
        if self.cfg.webui_enabled and self.cfg.webui_port > 0:
            self._webui = CampusWebUI(
                self.cfg.webui_host,
                self.cfg.webui_port,
                self._webui_handlers(),
            )
            try:
                self._webui.start()
            except OSError as exc:
                log.warning("failed to bind campus web UI: %s", exc)
                self._webui = None

    def stop(self) -> None:
        self._stop.set()
        self._reloc_stop.set()
        self.executive.shutdown()
        if self._webui is not None:
            self._webui.stop()
            self._webui = None
        if self._survey_thread is not None:
            self._survey_thread.join(timeout=2.0)
            self._survey_thread = None
        if self._executor is not None:
            try:
                self._executor.shutdown(timeout_sec=1.0)
            except Exception:  # noqa: BLE001
                pass
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=2.0)
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def _start_ros(self) -> None:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Twist
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        node = Node(
            "campus_nav",
            parameter_overrides=[Parameter("use_sim_time", value=self.cfg.use_sim_time)],
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(
            PoseWithCovarianceStamped, self.cfg.rtk_fix_topic, self._on_rtk_fix, qos_profile_sensor_data
        )
        node.create_subscription(String, self.cfg.rtk_status_topic, self._on_rtk_status, 10)
        node.create_subscription(Odometry, self.cfg.odom_topic, self._on_odom, qos_profile_sensor_data)
        node.create_subscription(OccupancyGrid, self.cfg.local_cost_grid_topic, self._on_cost_grid, latched)
        if self.cfg.visual_reloc_enabled:
            node.create_subscription(CameraInfo, self.cfg.intrinsics_topic, self._on_camera_info, qos_profile_sensor_data)
            node.create_subscription(
                TransformStamped, self.cfg.extrinsics_topic, self._on_camera_extrinsics, qos_profile_sensor_data
            )
            node.create_subscription(Image, self.cfg.rgb_topic, self._on_rgb_image, qos_profile_sensor_data)
        self._cmd_vel_pub = node.create_publisher(Twist, self.cfg.cmd_vel_topic, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        self._ros_node = node
        self._executor = executor
        self._ros_thread = threading.Thread(target=self._spin_ros, name="campus-nav-ros", daemon=True)
        self._ros_thread.start()

    def _spin_ros(self) -> None:
        while not self._stop.is_set() and self._executor is not None:
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    break
                log.exception("campus_nav ROS2 spin_once error (continuing)")
                time.sleep(0.1)

    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose.position
        yaw = _quat_yaw(msg.pose.pose.orientation)
        with self.lock:
            self.fusion.update_odom(float(p.x), float(p.y), yaw, z=float(p.z))

    def _on_rtk_fix(self, msg: Any) -> None:
        p = msg.pose.pose.position
        yaw = _quat_yaw(msg.pose.pose.orientation)
        cov = msg.pose.covariance
        sigma_xy = (
            math.sqrt(max(1e-6, float(cov[0])))
            if (cov is not None and len(cov) > 0)
            else 0.05
        )
        with self.lock:
            if self._loc_paused:
                return
            bldg = self.campus_map.find_building_at(float(p.x), float(p.y))
            inside_bldg = bldg is not None
            inside_multifloor = False
            if bldg is not None:
                bid = bldg["id"]
                bldg_floors = [fm for fm in self.indoor_store.all_maps() if fm.building_id == bid and fm.nodes]
                inside_multifloor = len(bldg_floors) >= 2
            self.fusion.update_rtk_fix(
                float(p.x),
                float(p.y),
                yaw,
                sigma_xy,
                inside_multifloor_building=inside_multifloor,
                inside_building=inside_bldg,
            )

    def _on_rtk_status(self, msg: Any) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            return
        if isinstance(data, dict):
            with self.lock:
                self.fusion.update_rtk_status(data)

    def _on_cost_grid(self, msg: Any) -> None:
        info = msg.info
        w, h = int(info.width), int(info.height)
        if w <= 0 or h <= 0 or len(msg.data) != w * h:
            return
        arr = np.asarray(msg.data, dtype=np.int8).reshape((h, w))
        snap = LocalCostGridSnapshot(
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            resolution=float(info.resolution),
            width=w,
            height=h,
            data=arr,
        )
        with self.lock:
            self._cost_grid = snap

    def _on_camera_info(self, msg: Any) -> None:
        k = msg.k
        d = msg.d
        if k is not None and len(k) == 9:
            self.fiducial_localizer.update_intrinsics(k, d)

    def _on_camera_extrinsics(self, msg: Any) -> None:
        t = msg.transform.translation
        r = msg.transform.rotation
        self.fiducial_localizer.update_extrinsics(
            float(t.x), float(t.y), float(t.z),
            float(r.x), float(r.y), float(r.z), float(r.w),
        )

    def _on_rgb_image(self, msg: Any) -> None:
        if not self.cfg.visual_reloc_enabled:
            return
        now = time.monotonic()
        min_dt = 1.0 / max(0.5, float(self.cfg.visual_reloc_hz))
        if now - self._last_rgb_proc_t < min_dt:
            return
        self._last_rgb_proc_t = now
        w, h = int(msg.width), int(msg.height)
        if w <= 0 or h <= 0:
            return
        enc = str(msg.encoding or "").lower()
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        if enc in ("rgb8", "bgr8") and buf.size >= w * h * 3:
            img = buf[: w * h * 3].reshape((h, w, 3))
            if enc == "bgr8":
                img = img[:, :, ::-1]
        elif enc in ("rgba8", "bgra8") and buf.size >= w * h * 4:
            img = buf[: w * h * 4].reshape((h, w, 4))[:, :, :3]
            if enc == "bgra8":
                img = img[:, :, ::-1]
        else:
            return
        # Freeze visual relocalization during active elevator maneuvers so distant markers
        # seen through glass cannot shift the robot's offset while boarding/riding/exiting.
        if self.executive.snapshot().get("elevator_phase"):
            return
        with self.lock:
            if self._loc_paused:
                return
            if not self.indoor_store.all_markers():
                return
            fixes = self.fiducial_localizer.detect_and_localize(img, self.indoor_store)
            if not fixes:
                return
            confirmed = self.fiducial_filter.push(fixes[0])
            if confirmed is None:
                return
            was_localized = self.fusion.initialized_from_rtk
            old_pose = self.fusion.campus_pose()
            old_xy = (old_pose[0], old_pose[1]) if old_pose is not None else None
            updated = self.fusion.update_visual_fix(
                confirmed.robot_x,
                confirmed.robot_y,
                confirmed.robot_yaw,
                indoor_id=confirmed.indoor_id,
                building_name=confirmed.building_name,
                floor=confirmed.floor,
                marker_id=confirmed.marker_id,
                distance_m=confirmed.distance_m,
                reproj_err_px=confirmed.reproj_err_px,
                label=confirmed.label,
            )
            if updated:
                if not was_localized and confirmed.indoor_id:
                    self.active_indoor_id = confirmed.indoor_id
                # Throttle repetitive stationary logs while keeping 4 Hz fusion active
                log_key = (confirmed.marker_id, confirmed.indoor_id)
                moved_log = (
                    self._last_vis_log_xy is None
                    or math.hypot(confirmed.robot_x - self._last_vis_log_xy[0], confirmed.robot_y - self._last_vis_log_xy[1]) >= 0.6
                )
                if log_key != self._last_vis_log_key or moved_log or (now - self._last_vis_log_t) >= 3.0:
                    self._last_vis_log_key = log_key
                    self._last_vis_log_t = now
                    self._last_vis_log_xy = (confirmed.robot_x, confirmed.robot_y)
                    log.info(
                        "visual relocalized via marker #%d (%s %s): (%.2f, %.2f) dist=%.2fm err=%.2fpx",
                        confirmed.marker_id,
                        confirmed.building_name,
                        confirmed.floor,
                        confirmed.robot_x,
                        confirmed.robot_y,
                        confirmed.distance_m,
                        confirmed.reproj_err_px,
                    )
        if updated:
            # Non-blocking in-flight verification: only triggers replan if floor mismatches or pose jumped >= 4m
            self.executive.notify_visual_fix(
                confirmed.indoor_id,
                f"{confirmed.building_name}{confirmed.floor}",
                old_xy=old_xy,
                new_xy=(confirmed.robot_x, confirmed.robot_y),
            )

    def _survey_loop(self) -> None:
        """Polls 25577 for RTK stations (slowly) and handheld rtk_mapper (ONLY in mapping mode)."""
        last_station_poll = 0.0
        while not self._stop.is_set():
            if self.cfg.server_port <= 0:
                self._stop.wait(2.0)
                continue
            with self.lock:
                mapping_open = (
                    self._mapping_ui_open or self.recorder.active or self.indoor_recorder.active
                )
            now = time.monotonic()
            if mapping_open:
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
                            has_new_mark = bool(
                                last_indoor_mark
                                and int(last_indoor_mark.get("seq", 0)) > self.indoor_recorder._last_mark_seq
                            )
                            if self.active_indoor_id or self.indoor_recorder.active or has_new_mark:
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
                    last_station_poll = now
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self._survey_error = str(exc)
                self._stop.wait(0.35)
            else:
                if now - last_station_poll >= 5.0:
                    try:
                        stations = self.mapper_client.list_stations(self.cfg.dimension)
                        with self.lock:
                            self._stations = stations
                            self._survey_holders = []
                            self._survey_error = ""
                    except Exception:  # noqa: BLE001
                        pass
                    last_station_poll = now
                self._stop.wait(0.5)

    # ── Web UI handlers ──────────────────────────────────────────────────────
    def _webui_handlers(self) -> dict[str, Any]:
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
            "POST:api/survey/undo_draft_point": self._api_survey_undo_draft_point,
            "POST:api/survey/drop_poi": self._api_survey_drop_poi,
            "POST:api/indoor/select_floor": self._api_indoor_select_floor,
            "POST:api/indoor/replace": self._api_indoor_replace,
            "POST:api/indoor/reset_floor": self._api_indoor_reset_floor,
            "POST:api/indoor/anchor_entrance": self._api_indoor_anchor_entrance,
            "POST:api/indoor/mark_room_door": self._api_indoor_mark_room_door,
            "POST:api/indoor/mark_room_nav": self._api_indoor_mark_room_nav,
            "POST:api/indoor/mark_elevator_door": self._api_indoor_mark_elevator_door,
            "POST:api/indoor/mark_elevator_cabin": self._api_indoor_mark_elevator_cabin,
            "POST:api/indoor/mark_stairs": self._api_indoor_mark_stairs,
            "POST:api/indoor/corridor_toggle": self._api_indoor_corridor_toggle,
            "POST:api/navigate": self._api_navigate,
            "POST:api/navigate/cancel": self._api_navigate_cancel,
            "POST:api/navigate/clear_blocked": self._api_clear_blocked,
            "POST:api/navigate/tuning": self._api_nav_tuning,
            "POST:api/relocalize": self._api_relocalize,
            "POST:api/localization/invalidate": self._api_localization_invalidate,
        }

    def _api_state(self, _: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            files = sorted(p.name for p in self.cfg.map_dir.glob("*.json"))
            if self.active_file not in files:
                files.append(self.active_file)
            local_box = None
            if self._cost_grid is not None:
                ox, oy = self.fusion.odom_to_campus(self._cost_grid.origin_x, self._cost_grid.origin_y)
                local_box = {
                    "origin_x": round(ox, 2),
                    "origin_y": round(oy, 2),
                    "size_x": round(self._cost_grid.width * self._cost_grid.resolution, 2),
                    "size_y": round(self._cost_grid.height * self._cost_grid.resolution, 2),
                }
            active_fm = self.indoor_store.get(self.active_indoor_id) if self.active_indoor_id else None
            return {
                "ok": True,
                "active_file": self.active_file,
                "map_files": files,
                "map": self.campus_map.to_dict(),
                "active_indoor_id": self.active_indoor_id,
                "active_indoor_map": active_fm.to_dict() if active_fm is not None else None,
                "indoor_maps": self.indoor_store.list_summaries(),
                "all_rooms": self.indoor_store.all_rooms(),
                "all_elevators": self.indoor_store.all_elevators(),
                "all_stairs": self.indoor_store.all_stairs(),
                "robot": self.fusion.snapshot(),
                "local_window": local_box,
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
                "navigation": self.executive.snapshot(),
                "nav_tuning": self.executive.get_nav_tuning(),
                "relocalize": dict(self._reloc_state),
                "elevator": (
                    self.elevator_client.status_snapshot(self.indoor_store)
                    if self.elevator_client is not None
                    else {"enabled": False, "floors": []}
                ),
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
        path = self.cfg.map_dir / fname
        loaded = CampusMap.load(path)
        with self.lock:
            self.active_file = fname
            self.campus_map = loaded
            self.planner.clear_blocked_edges()
        return {"ok": True, "file": fname, "detail": f"loaded {fname}"}

    def _api_map_new(self, body: dict[str, Any]) -> dict[str, Any]:
        fname = str(body.get("file") or "campus_new.json").strip()
        if not fname.endswith(".json"):
            fname += ".json"
        if "/" in fname or "\\" in fname or ".." in fname:
            return {"ok": False, "detail": "invalid filename"}
        name = str(body.get("name") or Path(fname).stem)
        fresh = CampusMap(campus_id=Path(fname).stem, name=name)
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
        open_flag = bool(body.get("open", False))
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
                return {"ok": False, "detail": "need at least 3 vertices to close area"}
            self.campus_map.save(self.cfg.map_dir / self.active_file)
            return {"ok": True, "area": area}

    def _api_survey_undo_draft_point(self, _: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            removed = self.recorder.undo_area_vertex()
            if removed is None:
                return {"ok": False, "detail": "当前没有待撤销的建筑角点"}
            return {
                "ok": True,
                "removed": removed,
                "remaining": len(self.recorder._area_draft),
            }

    def _api_survey_drop_poi(self, body: dict[str, Any]) -> dict[str, Any]:
        name = str(body.get("name") or "").strip()
        if not name:
            return {"ok": False, "detail": "POI name is required"}
        cat = str(body.get("category") or "landmark")
        aliases = [s.strip() for s in str(body.get("aliases") or "").split(",") if s.strip()]
        with self.lock:
            if "x" in body and "y" in body:
                px, py = float(body["x"]), float(body["y"])
            elif self._survey_holders and (self._survey_holders[0].get("fix") or {}).get("valid"):
                fix = self._survey_holders[0]["fix"]
                px, py = float(fix["east"]), float(fix["north"])
            else:
                return {"ok": False, "detail": "no active rtk_mapper holder with valid fix"}
            poi = self.campus_map.add_poi(name, px, py, category=cat, aliases=aliases)
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
            self._mapping_ui_open = True
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
            self._mapping_ui_open = True
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
            self._mapping_ui_open = True
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

    def _api_indoor_mark_elevator_door(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._mapping_ui_open = True
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            fm = self._resolve_active_floor_map(
                xy[0], xy[1],
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            door = self.indoor_recorder.mark_elevator_door(
                fm,
                xy[0],
                xy[1],
                name=str(body.get("elevator_name") or body.get("name") or ""),
                elevator_id=str(body.get("elevator_id") or ""),
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, "door": door, "indoor_recorder": self.indoor_recorder.state_dict()}

    def _api_indoor_mark_elevator_cabin(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._mapping_ui_open = True
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            elev_y = body.get("up")
            if elev_y is None and self._indoor_holders:
                elev_y = (self._indoor_holders[0].get("pose") or {}).get("up")
            fm = self._resolve_active_floor_map(
                xy[0], xy[1], elev_y,
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            out = self.indoor_recorder.mark_elevator_cabin(
                fm,
                xy[0],
                xy[1],
                name=str(body.get("elevator_name") or body.get("name") or ""),
                elevator_id=str(body.get("elevator_id") or ""),
                floor_label=str(body.get("floor_label") or fm.floor),
                elevation_y=float(elev_y) if elev_y is not None else None,
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, **out, "indoor_recorder": self.indoor_recorder.state_dict()}

    def _api_indoor_mark_stairs(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._mapping_ui_open = True
            xy = self._current_indoor_xy(body)
            if xy is None:
                return {"ok": False, "detail": "未检测到手持 indoor_mapper（室内建图仪）的玩家"}
            elev_y = body.get("up")
            if elev_y is None and self._indoor_holders:
                elev_y = (self._indoor_holders[0].get("pose") or {}).get("up")
            fm = self._resolve_active_floor_map(
                xy[0], xy[1], elev_y,
                building_id=str(body.get("building_id") or ""),
                floor=str(body.get("floor") or ""),
            )
            st = fm.upsert_stairs(
                xy[0],
                xy[1],
                stairs_id=str(body.get("stairs_id") or "stairs_main"),
                name=str(body.get("name") or "楼梯"),
                elevation_y=float(elev_y) if elev_y is not None else None,
            )
            self.indoor_store.save(fm)
            return {"ok": True, "indoor_id": fm.indoor_id, "stairs": st, "indoor_recorder": self.indoor_recorder.state_dict()}

    def _api_indoor_corridor_toggle(self, body: dict[str, Any]) -> dict[str, Any]:
        action = str(body.get("action") or "").strip().lower()
        with self.lock:
            self._mapping_ui_open = True
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

    def _api_navigate(self, body: dict[str, Any]) -> dict[str, Any]:
        place = str(body.get("place") or "")
        node_id = str(body.get("node_id") or "")
        indoor_id = str(body.get("indoor_id") or "")
        start_indoor_id = str(body.get("start_indoor_id") or "")
        if not start_indoor_id:
            start_indoor_id = self._infer_start_indoor_id()
        x = float(body["x"]) if "x" in body and body["x"] is not None else None
        y = float(body["y"]) if "y" in body and body["y"] is not None else None
        timeout_s = float(body.get("timeout_s", 0.0))
        return self.executive.start_run(
            place=place,
            x=x,
            y=y,
            node_id=node_id,
            indoor_id=indoor_id,
            start_indoor_id=start_indoor_id,
            timeout_s=timeout_s,
        )

    def _api_nav_tuning(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            la = float(body["lookahead_m"]) if "lookahead_m" in body and body["lookahead_m"] is not None else None
            adv = float(body["advance_radius_m"]) if "advance_radius_m" in body and body["advance_radius_m"] is not None else None
            clamp = bool(body["outdoor_corner_clamp"]) if "outdoor_corner_clamp" in body and body["outdoor_corner_clamp"] is not None else None
        except (TypeError, ValueError) as exc:
            return {"ok": False, "detail": f"参数格式错误: {exc}"}
        updated = self.executive.set_nav_tuning(
            lookahead_m=la,
            advance_radius_m=adv,
            outdoor_corner_clamp=clamp,
        )
        return {
            "ok": True,
            "nav_tuning": updated,
            "detail": (
                f"已更新室外粒度：前瞻 {updated['lookahead_m']:.1f}m · "
                f"切点 {updated['advance_radius_m']:.1f}m · "
                f"街角卡点{'开' if updated['outdoor_corner_clamp'] else '关'}"
            ),
        }

    def _set_reloc_state(self, active: bool, phase: str, detail: str, hops_done: int = 0) -> None:
        with self.lock:
            self._reloc_state = {
                "active": bool(active),
                "phase": phase,
                "detail": detail,
                "hops_done": int(hops_done),
            }

    def _boot_localize_loop(self) -> None:
        """On cold boot: prefer outdoor RTK if available; otherwise auto-search for indoor ArUco marker."""
        self._set_reloc_state(True, "WAITING_ODOM", "开机等待机器人底盘与定位传感器就绪...")
        while not self._stop.is_set():
            with self.lock:
                if self.fusion.initialized_from_rtk:
                    src = self.fusion.localization_source or "rtk"
                    self._set_reloc_state(False, "LOCKED", f"已完成开机定位 ({src})")
                    return
                has_odom = self.fusion.odom_pose is not None
            if has_odom:
                break
            self._stop.wait(0.25)

        if self._stop.is_set():
            return

        # Wait briefly to see if outdoor RTK locks automatically.
        self._set_reloc_state(True, "WAITING_RTK", "检测室外 RTK 信号中...")
        deadline = time.monotonic() + float(self.cfg.boot_rtk_wait_s)
        while not self._stop.is_set() and time.monotonic() < deadline:
            with self.lock:
                if self.fusion.initialized_from_rtk:
                    src = self.fusion.localization_source or "rtk"
                    log.info("boot localization locked automatically via %s", src)
                    self._set_reloc_state(False, "LOCKED", f"已通过 {src.upper()} 自动完成开机定位")
                    return
            self._stop.wait(0.2)

        if self._stop.is_set():
            return

        if not self.indoor_store.all_markers():
            self._set_reloc_state(False, "UNLOCALIZED", "未收到室外 RTK，且室内地图尚未绑定视觉标牌")
            return

        log.info("no outdoor RTK after %.1fs on boot; starting indoor visual marker search", self.cfg.boot_rtk_wait_s)
        self._start_marker_search(
            timeout_s=70.0,
            max_hops=self.cfg.relocalize_max_hops,
            require_fresh_visual=False,
        )

    def _start_marker_search(
        self,
        *,
        timeout_s: float = 60.0,
        max_hops: int = 3,
        require_fresh_visual: bool = True,
    ) -> dict[str, Any]:
        """Two-stage pure-sensor indoor marker search: 360-deg spin + safe local corridor hops."""
        if not self.cfg.visual_reloc_enabled:
            return {"ok": False, "detail": "visual_reloc_enabled is false"}
        if not self.indoor_store.all_markers():
            return {"ok": False, "detail": "地图中尚无已绑定的视觉标牌（请先用室内建图仪右键绑定标牌）"}
        if self._reloc_active:
            return {"ok": True, "detail": self._reloc_state.get("detail") or "正在寻找视觉标牌定位中..."}

        self.executive.cancel()
        self._reloc_stop.clear()
        self._reloc_active = True
        start_wall = time.time()

        def _is_locked() -> tuple[bool, str]:
            with self.lock:
                if self.fusion.visual_wall > start_wall and self.fusion.last_visual_fix:
                    vf = self.fusion.last_visual_fix
                    return True, f"已通过标牌 #{vf['marker_id']} 定位到 {vf['building_name']}{vf['floor']}"
                if not require_fresh_visual and self.fusion.initialized_from_rtk:
                    if self.fusion.localization_source == "rtk":
                        return True, "已通过室外 RTK 完成定位"
                    vf = self.fusion.last_visual_fix
                    if vf:
                        return True, f"已通过标牌 #{vf['marker_id']} 定位到 {vf['building_name']}{vf['floor']}"
                    return True, "已完成全局定位"
            return False, ""

        def _record_seen(seen_cells: set[tuple[int, int]]) -> None:
            with self.lock:
                odom = self.fusion.odom_pose
                grid = self._cost_grid
            if odom is not None and grid is not None:
                grid.mark_camera_fov_seen(odom[0], odom[1], odom[2], seen_cells)

        def _spin_360(
            spin_sec: float,
            overall_deadline: float,
            hop_idx: int,
            seen_cells: set[tuple[int, int]],
        ) -> tuple[bool, str]:
            from geometry_msgs.msg import Twist

            phase_msg = (
                "正在原地 360° 环顾寻找视觉标牌（同步记录已看过区域）..."
                if hop_idx == 0
                else f"探索点 {hop_idx}/{max_hops}：正在原地 360° 环顾寻找标牌..."
            )
            self._set_reloc_state(True, "SPINNING", phase_msg, hops_done=hop_idx)
            spin_deadline = min(overall_deadline, time.monotonic() + spin_sec)
            try:
                while (
                    not self._stop.is_set()
                    and not self._reloc_stop.is_set()
                    and time.monotonic() < spin_deadline
                ):
                    _record_seen(seen_cells)
                    locked, msg = _is_locked()
                    if locked:
                        return True, msg
                    if self._cmd_vel_pub is not None:
                        tw = Twist()
                        tw.angular.z = 0.45
                        self._cmd_vel_pub.publish(tw)
                    time.sleep(0.15)
            finally:
                if self._cmd_vel_pub is not None:
                    self._cmd_vel_pub.publish(Twist())
            return _is_locked()

        def _explore_hop(
            hop_idx: int,
            seen_cells: set[tuple[int, int]],
            visited: list[tuple[float, float]],
            failed: list[tuple[float, float]],
            overall_deadline: float,
        ) -> tuple[bool, str, bool]:
            """Move to the best unexplored frontier or large unseen area (escaping enclosed rooms!)."""
            with self.lock:
                odom = self.fusion.odom_pose
                grid = self._cost_grid
            if odom is None or grid is None:
                return False, "", False
            _record_seen(seen_cells)
            cand = grid.pick_frontier_explore_target(
                odom[0],
                odom[1],
                seen_cells,
                visited,
                failed_xy=failed,
            )
            reason = "未探索区域"
            if cand is not None:
                tx, ty, heading, reason = cand
            else:
                fb = grid.pick_safe_explore_target(odom[0], odom[1], visited)
                if fb is None:
                    return False, "", False
                tx, ty, heading = fb
            self._set_reloc_state(
                True,
                "EXPLORING",
                f"当前区域已扫描完，前往{reason}继续找标牌 ({hop_idx}/{max_hops})...",
                hops_done=hop_idx,
            )
            log.info(
                "marker search hop %d/%d -> (%.2f, %.2f) [%s] (seen_cells=%d)",
                hop_idx,
                max_hops,
                tx,
                ty,
                reason,
                len(seen_cells),
            )
            half = heading / 2.0
            resp = self.executive._rpc(
                "navigate",
                {
                    "goal": {
                        "header": {"stamp": {"sec": 0, "nanosec": 0}, "frame_id": "minecraft_odom"},
                        "pose": {
                            "position": {"x": float(tx), "y": float(ty), "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
                        },
                    }
                },
                10.0,
            )
            rid = str(resp.get("run_id") or "")
            if not (resp.get("accepted") and rid):
                failed.append((tx, ty))
                return False, "", True  # allow next hop to pick a different frontier
            hop_deadline = min(overall_deadline, time.monotonic() + 20.0)
            start_pos = (odom[0], odom[1])
            succeeded = False
            try:
                while (
                    not self._stop.is_set()
                    and not self._reloc_stop.is_set()
                    and time.monotonic() < hop_deadline
                ):
                    _record_seen(seen_cells)
                    locked, msg = _is_locked()
                    if locked:
                        return True, msg, True
                    st = self.executive._rpc("status", {"run_id": rid}, 5.0)
                    lstate = str(st.get("state") or "RUNNING").upper()
                    if lstate == "SUCCEEDED":
                        succeeded = True
                        break
                    if lstate in ("FAILED", "CANCELED", "CANCELLED"):
                        break
                    time.sleep(0.25)
            finally:
                self.executive._rpc("cancel", {"run_id": rid}, 5.0)

            with self.lock:
                end_odom = self.fusion.odom_pose
            moved_dist = (
                math.hypot(end_odom[0] - start_pos[0], end_odom[1] - start_pos[1])
                if end_odom is not None
                else 0.0
            )
            if succeeded or moved_dist >= 1.5:
                visited.append((end_odom[0], end_odom[1]) if end_odom is not None else (tx, ty))
            else:
                failed.append((tx, ty))
            locked, msg = _is_locked()
            return locked, msg, True

        def _search_worker() -> None:
            from geometry_msgs.msg import Twist

            overall_deadline = time.monotonic() + timeout_s
            seen_cells: set[tuple[int, int]] = set()
            visited: list[tuple[float, float]] = []
            failed: list[tuple[float, float]] = []
            with self.lock:
                if self.fusion.odom_pose is not None:
                    visited.append((self.fusion.odom_pose[0], self.fusion.odom_pose[1]))
            try:
                # Stage 1: Full 360-degree scan at current spot (~15s at 0.45 rad/s), marking seen cells
                locked, msg = _spin_360(15.0, overall_deadline, 0, seen_cells)
                if locked:
                    log.info("marker search succeeded: %s", msg)
                    self._set_reloc_state(False, "LOCKED", msg, hops_done=0)
                    return

                # Stage 2: Visual-coverage frontier exploration hops + 360-degree scan
                for hop in range(1, max(0, int(max_hops)) + 1):
                    if self._stop.is_set() or self._reloc_stop.is_set() or time.monotonic() >= overall_deadline:
                        break
                    locked, msg, progressed = _explore_hop(hop, seen_cells, visited, failed, overall_deadline)
                    if locked:
                        log.info("marker search succeeded during hop %d: %s", hop, msg)
                        self._set_reloc_state(False, "LOCKED", msg, hops_done=hop)
                        return
                    if not progressed:
                        break
                    locked, msg = _spin_360(15.0, overall_deadline, hop, seen_cells)
                    if locked:
                        log.info("marker search succeeded after hop %d: %s", hop, msg)
                        self._set_reloc_state(False, "LOCKED", msg, hops_done=hop)
                        return

                if self._reloc_stop.is_set():
                    self._set_reloc_state(False, "CANCELED", "已取消找标牌定位")
                else:
                    self._set_reloc_state(False, "FAILED", "探索完周围未找到视觉标牌，请移动至标牌附近后重试")
            finally:
                if self._cmd_vel_pub is not None:
                    self._cmd_vel_pub.publish(Twist())
                self._reloc_active = False

        self._reloc_thread = threading.Thread(target=_search_worker, name="campus-relocalize-search", daemon=True)
        self._reloc_thread.start()
        return {"ok": True, "detail": "已启动找标牌定位（原地环顾 + 必要时沿走廊安全探索，扫到标牌后自动停止）"}

    def _api_localization_invalidate(self, _: dict[str, Any]) -> dict[str, Any]:
        """Force robot localization into an unlocalized state and pause background auto-locking."""
        self._reloc_stop.set()
        self.executive.cancel()
        with self.lock:
            self._loc_paused = True
            self.fusion.invalidate()
            self.fiducial_filter = VisualRelocalizerFilter(min_consecutive=2)
            self._reloc_state = {
                "active": False,
                "phase": "INVALIDATED",
                "detail": "定位已手动失效，点击「自主重新定位」后将重新检测 RTK / 寻找视觉标牌",
                "hops_done": 0,
            }
        log.info("robot localization manually invalidated via Web UI")
        return {"ok": True, "detail": "已使机器人定位失效（已暂停自动重锁，请点击「自主重新定位」）"}

    def _api_relocalize(self, body: dict[str, Any]) -> dict[str, Any]:
        """Trigger unified autonomous relocalization: outdoor RTK first, otherwise indoor marker search."""
        timeout_s = max(5.0, min(180.0, float(body.get("timeout_s", 120.0))))
        max_hops = max(0, min(15, int(body.get("max_hops", self.cfg.relocalize_max_hops))))
        rtk_wait_s = max(1.0, min(10.0, float(body.get("rtk_wait_s", min(2.5, self.cfg.boot_rtk_wait_s)))))

        self._reloc_stop.set()
        old_thread = self._reloc_thread
        if old_thread is not None and old_thread.is_alive() and old_thread is not threading.current_thread():
            old_thread.join(timeout=1.5)

        self.executive.cancel()
        self._reloc_stop.clear()
        with self.lock:
            self._loc_paused = False
            self.fusion.invalidate()
            self.fiducial_filter = VisualRelocalizerFilter(min_consecutive=2)

        def _unified_reloc_worker() -> None:
            self._set_reloc_state(True, "WAITING_RTK", "正在检测室外 RTK / 视野内标牌...")
            wait_deadline = time.monotonic() + rtk_wait_s
            while not self._stop.is_set() and not self._reloc_stop.is_set() and time.monotonic() < wait_deadline:
                with self.lock:
                    if self.fusion.initialized_from_rtk and not self.fusion.invalidated:
                        src = self.fusion.localization_source or "rtk"
                        if src == "visual" and self.fusion.last_visual_fix:
                            vf = self.fusion.last_visual_fix
                            msg = f"已通过标牌 #{vf['marker_id']} 定位到 {vf['building_name']}{vf['floor']}"
                        else:
                            msg = "已通过室外 RTK 完成重新定位"
                        log.info("autonomous relocalization succeeded immediately: %s", msg)
                        self._set_reloc_state(False, "LOCKED", msg)
                        return
                time.sleep(0.15)
            if self._stop.is_set() or self._reloc_stop.is_set():
                self._set_reloc_state(False, "CANCELED", "已取消重新定位")
                return
            # No outdoor RTK -> start indoor active marker search (360-deg spin + safe corridor hops)
            res = self._start_marker_search(
                timeout_s=timeout_s,
                max_hops=max_hops,
                require_fresh_visual=False,
            )
            if not res.get("ok"):
                self._set_reloc_state(False, "FAILED", str(res.get("detail") or "无法启动找标牌定位"))

        t = threading.Thread(target=_unified_reloc_worker, name="campus-unified-relocalize", daemon=True)
        self._reloc_thread = t
        t.start()
        return {"ok": True, "detail": "已启动自主重新定位（室外优先 RTK，室内自动环顾并沿走廊寻找标牌）"}

    def _api_navigate_cancel(self, _: dict[str, Any]) -> dict[str, Any]:
        self._reloc_stop.set()
        return self.executive.cancel()

    def _api_clear_blocked(self, _: dict[str, Any]) -> dict[str, Any]:
        self.planner.clear_blocked_edges()
        return {"ok": True, "detail": "cleared blocked edges"}


_runtime_lock = threading.Lock()
_runtime: CampusNavRuntime | None = None


@campus_nav.mcp("robonix/skill/outdoor_navigation/navigate_to_place")
def mcp_navigate_to_place(
    req: outdoor_navigation_mcp.NavigateToPlace_Request,
) -> outdoor_navigation_mcp.NavigateToPlace_Response:
    if _runtime is None:
        return outdoor_navigation_mcp.NavigateToPlace_Response(
            accepted=False, run_id="", detail="campus_nav is not active"
        )
    place = (req.place or "").strip()
    res = _runtime.executive.start_run(
        place=place,
        x=None if place else float(req.x),
        y=None if place else float(req.y),
        start_indoor_id=_runtime._infer_start_indoor_id(),
        timeout_s=float(req.timeout_s),
    )
    return outdoor_navigation_mcp.NavigateToPlace_Response(
        accepted=bool(res.get("accepted", False)),
        run_id=str(res.get("run_id") or ""),
        detail=str(res.get("detail") or ""),
    )


@campus_nav.mcp("robonix/skill/outdoor_navigation/status")
def mcp_status(
    req: outdoor_navigation_mcp.GetOutdoorNavigationStatus_Request,
) -> outdoor_navigation_mcp.GetOutdoorNavigationStatus_Response:
    if _runtime is None:
        return outdoor_navigation_mcp.GetOutdoorNavigationStatus_Response(
            known=False,
            state="IDLE",
            place="",
            remaining_m=0.0,
            current_segment=0,
            total_segments=0,
            detail="campus_nav is not active",
        )
    snap = _runtime.executive.snapshot()
    rid = (req.run_id or "").strip()
    if rid and snap["run_id"] != rid:
        return outdoor_navigation_mcp.GetOutdoorNavigationStatus_Response(
            known=False,
            state="IDLE",
            place="",
            remaining_m=0.0,
            current_segment=0,
            total_segments=0,
            detail=f"unknown run_id {rid}",
        )
    return outdoor_navigation_mcp.GetOutdoorNavigationStatus_Response(
        known=bool(snap["run_id"]),
        state=str(snap["state"]),
        place=str(snap["place"]),
        remaining_m=float(snap["remaining_m"]),
        current_segment=int(snap["current_segment"]),
        total_segments=int(snap["total_segments"]),
        detail=str(snap["detail"]),
    )


@campus_nav.mcp("robonix/skill/outdoor_navigation/cancel")
def mcp_cancel(
    req: outdoor_navigation_mcp.CancelOutdoorNavigation_Request,
) -> outdoor_navigation_mcp.CancelOutdoorNavigation_Response:
    if _runtime is None:
        return outdoor_navigation_mcp.CancelOutdoorNavigation_Response(
            accepted=True, detail="campus_nav is inactive"
        )
    res = _runtime.executive.cancel((req.run_id or "").strip())
    return outdoor_navigation_mcp.CancelOutdoorNavigation_Response(
        accepted=bool(res.get("accepted", False)),
        detail=str(res.get("detail") or ""),
    )


@campus_nav.mcp("robonix/skill/outdoor_navigation/list_places")
def mcp_list_places(
    _req: outdoor_navigation_mcp.ListCampusPlaces_Request,
) -> outdoor_navigation_mcp.ListCampusPlaces_Response:
    if _runtime is None:
        return outdoor_navigation_mcp.ListCampusPlaces_Response(
            campus_id="", places_json="[]"
        )
    cmap = _runtime.get_map()
    places = [
        {
            "id": p["id"],
            "name": p["name"],
            "aliases": p.get("aliases", []),
            "category": p.get("category", "landmark"),
            "x": p["x"],
            "y": p["y"],
        }
        for p in cmap.pois
    ]
    for r in _runtime.indoor_store.all_rooms():
        places.append({
            "id": r["id"],
            "name": r["name"],
            "aliases": r.get("aliases", []),
            "category": "room",
            "building_name": r.get("building_name", ""),
            "floor": r.get("floor", ""),
            "indoor_id": r.get("indoor_id", ""),
            "x": r["x"],
            "y": r["y"],
        })
    return outdoor_navigation_mcp.ListCampusPlaces_Response(
        campus_id=cmap.campus_id,
        places_json=json.dumps(places, ensure_ascii=False, separators=(",", ":")),
    )


@campus_nav.on_init
def on_init(cfg: dict[str, Any]):
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            return Ok()
        try:
            parsed = params.from_config(cfg or {}, PACKAGE_ROOT)
        except Exception as exc:  # noqa: BLE001
            return Err(f"invalid campus_nav config: {exc}")
        rt = CampusNavRuntime(parsed)
        try:
            rt.start()
        except Exception as exc:  # noqa: BLE001
            rt.stop()
            return Err(f"failed to start campus_nav: {exc}")
        _runtime = rt
        log.info("campus_nav active (map=%s, webui=%s:%d)", parsed.map_file, parsed.webui_host, parsed.webui_port)
        return Ok()


@campus_nav.on_shutdown
def on_shutdown() -> None:
    global _runtime
    with _runtime_lock:
        rt, _runtime = _runtime, None
    if rt is not None:
        rt.stop()
    log.info("campus_nav stopped")
