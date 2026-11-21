# SPDX-License-Identifier: MulanPSL-2.0
"""Fuses onboard RTK GNSS fixes with local minecraft_odom for smooth localization."""
from __future__ import annotations

import math
import time
from typing import Any


class RtkOdomFusion:
    """Tracks smooth offset T_campus_odom = (dx, dy) between campus_rtk and odom.

    When RTK is RTK_FIXED, updates offset with higher gain; on RTK_FLOAT uses
    gentle gain; during NO_FIX (indoors/under roof), freezes offset and coasts
    seamlessly on local odom.
    """

    def __init__(self, *, require_global_lock: bool = False) -> None:
        self.require_global_lock: bool = bool(require_global_lock)
        self.invalidated: bool = False
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.initialized_from_rtk: bool = False
        self.localization_source: str = ""  # "" | "rtk" | "visual"
        self.odom_pose: tuple[float, float, float] | None = None  # x, y, yaw
        self.odom_z: float | None = None
        self.odom_wall: float = 0.0
        self.rtk_pose: tuple[float, float, float] | None = None
        self.rtk_status: dict[str, Any] = {
            "valid": False,
            "fix_type": "NO_FIX",
            "fix_code": 0,
            "sigma_xy": 999.0,
            "sky_visibility": 0.0,
            "num_sats": 0,
            "base_id": "",
            "base_distance_m": -1.0,
            "station_count": 0,
        }
        self.rtk_wall: float = 0.0
        self.current_indoor_id: str = ""
        self.current_building_name: str = ""
        self.current_floor: str = ""
        self.last_visual_fix: dict[str, Any] | None = None
        self.visual_wall: float = 0.0

    def update_odom(self, x: float, y: float, yaw: float, z: float | None = None) -> None:
        self.odom_pose = (float(x), float(y), float(yaw))
        if z is not None and math.isfinite(float(z)):
            self.odom_z = float(z)
        self.odom_wall = time.time()

    def set_indoor_floor(self, indoor_id: str, building_name: str = "", floor: str = "") -> None:
        self.current_indoor_id = str(indoor_id or "")
        self.current_building_name = str(building_name or "")
        self.current_floor = str(floor or "")

    def invalidate(self) -> None:
        """Manually drop global localization lock (for cold-start & relocalization testing)."""
        self.invalidated = True
        self.initialized_from_rtk = False
        self.localization_source = ""
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.rtk_pose = None
        self.rtk_wall = 0.0
        self.current_indoor_id = ""
        self.current_building_name = ""
        self.current_floor = ""
        self.last_visual_fix = None
        self.visual_wall = 0.0

    def update_rtk_status(self, status: dict[str, Any]) -> None:
        self.rtk_status = dict(status)
        self.rtk_wall = time.time()

    def update_rtk_fix(
        self,
        east: float,
        north: float,
        yaw: float,
        sigma_xy: float,
        *,
        inside_multifloor_building: bool = False,
        inside_building: bool = False,
    ) -> None:
        self.rtk_pose = (float(east), float(north), float(yaw))
        self.rtk_wall = time.time()
        if self.odom_pose is None:
            return
        ox, oy, _ = self.odom_pose
        meas_dx = east - ox
        meas_dy = north - oy
        if not self.initialized_from_rtk:
            if sigma_xy > 2.0 or inside_multifloor_building:
                # Inside a multi-floor building, 2D RTK through a window cannot identify the floor;
                # wait for an indoor ArUco marker to establish floor + global lock.
                return
            self.offset_x = meas_dx
            self.offset_y = meas_dy
            self.initialized_from_rtk = True
            self.invalidated = False
            self.localization_source = "rtk"
            if sigma_xy <= 0.15 and not inside_building:
                self.current_indoor_id = ""
                self.current_building_name = ""
                self.current_floor = ""
            return
        # If recently locked by an indoor visual marker inside a building, don't let leaked
        # window RTK pull the offset or wipe the floor tag.
        recent_visual = self.visual_wall > 0.0 and (time.time() - self.visual_wall) < 10.0
        if inside_building and (recent_visual or self.current_indoor_id):
            return
        # Weight inversely with variance: FIXED (~0.05m) adapts quickly, FLOAT/SINGLE gently.
        if sigma_xy <= 0.15:
            alpha = 0.35
        elif sigma_xy <= 0.8:
            alpha = 0.08
        else:
            alpha = 0.02
        self.offset_x = (1.0 - alpha) * self.offset_x + alpha * meas_dx
        self.offset_y = (1.0 - alpha) * self.offset_y + alpha * meas_dy
        self.localization_source = "rtk"
        if sigma_xy <= 0.15 and not inside_building and not recent_visual:
            # Strong truly-outdoor RTK_FIXED clears indoor floor tag.
            self.current_indoor_id = ""
            self.current_building_name = ""
            self.current_floor = ""

    def update_visual_fix(
        self,
        east: float,
        north: float,
        yaw: float,
        *,
        indoor_id: str,
        building_name: str,
        floor: str,
        marker_id: int,
        distance_m: float,
        reproj_err_px: float = 0.0,
        label: str = "",
    ) -> bool:
        """Fuse an indoor ArUco visual marker pose fix and lock onto its floor.

        Does not override a fresh outdoor RTK_FIXED lock (< 3 s old).
        Returns True if the campus pose offset was updated.
        """
        now = time.time()
        self.last_visual_fix = {
            "marker_id": int(marker_id),
            "label": label or f"标牌#{marker_id}",
            "indoor_id": indoor_id,
            "building_name": building_name,
            "floor": floor,
            "x": round(float(east), 3),
            "y": round(float(north), 3),
            "yaw": round(float(yaw), 4),
            "distance_m": round(float(distance_m), 2),
            "reproj_err_px": round(float(reproj_err_px), 2),
            "timestamp": now,
        }
        self.visual_wall = now
        self.current_indoor_id = indoor_id
        self.current_building_name = building_name
        self.current_floor = floor

        # A confirmed 2-frame indoor wall ArUco marker is direct physical proof that the robot
        # is indoors on this floor — always fuse it even if a weak/leaked RTK fix arrived through a window.
        if self.odom_pose is None:
            self.rtk_pose = (float(east), float(north), float(yaw))
            self.initialized_from_rtk = True
            self.invalidated = False
            self.localization_source = "visual"
            return True

        ox, oy, _ = self.odom_pose
        meas_dx = float(east) - ox
        meas_dy = float(north) - oy
        if not self.initialized_from_rtk:
            self.offset_x = meas_dx
            self.offset_y = meas_dy
            self.initialized_from_rtk = True
        else:
            alpha = 0.70 if distance_m <= 4.0 else 0.45
            self.offset_x = (1.0 - alpha) * self.offset_x + alpha * meas_dx
            self.offset_y = (1.0 - alpha) * self.offset_y + alpha * meas_dy
        self.invalidated = False
        self.localization_source = "visual"
        return True

    def campus_pose(self) -> tuple[float, float, float] | None:
        if self.invalidated or (self.require_global_lock and not self.initialized_from_rtk):
            return None
        if self.odom_pose is not None:
            ox, oy, oyaw = self.odom_pose
            return (ox + self.offset_x, oy + self.offset_y, oyaw)
        if self.rtk_pose is not None:
            return self.rtk_pose
        return None

    def campus_to_odom(self, cx: float, cy: float) -> tuple[float, float]:
        return (float(cx) - self.offset_x, float(cy) - self.offset_y)

    def odom_to_campus(self, ox: float, oy: float) -> tuple[float, float]:
        return (float(ox) + self.offset_x, float(oy) + self.offset_y)

    def snapshot(self) -> dict[str, Any]:
        pose = self.campus_pose()
        return {
            "has_pose": pose is not None,
            "x": round(pose[0], 3) if pose else 0.0,
            "y": round(pose[1], 3) if pose else 0.0,
            "z": round(self.odom_z, 3) if self.odom_z is not None else None,
            "yaw": round(pose[2], 4) if pose else 0.0,
            "rtk_aligned": self.initialized_from_rtk,
            "globally_localized": self.initialized_from_rtk and not self.invalidated,
            "invalidated": self.invalidated,
            "require_global_lock": self.require_global_lock,
            "localization_source": self.localization_source,
            "offset_x": round(self.offset_x, 3),
            "offset_y": round(self.offset_y, 3),
            "rtk_status": dict(self.rtk_status),
            "rtk_age_s": round(max(0.0, time.time() - self.rtk_wall), 2) if self.rtk_wall else None,
            "current_indoor_id": self.current_indoor_id,
            "current_building_name": self.current_building_name,
            "current_floor": self.current_floor,
            "visual_fix": dict(self.last_visual_fix) if self.last_visual_fix else None,
            "visual_age_s": round(max(0.0, time.time() - self.visual_wall), 2) if self.visual_wall else None,
        }
