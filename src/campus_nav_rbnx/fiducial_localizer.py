# SPDX-License-Identifier: MulanPSL-2.0
"""Pure-sensor RGB visual fiducial relocalization using OpenCV ArUco 5x5 + solvePnP.

Consumes only standard camera streams (/minecraft/camera/rgb, /minecraft/camera/intrinsics,
/minecraft/camera/extrinsics) and the mapped indoor floor marker database. Never reads
Minecraft world/block truth at runtime.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from campus_nav_rbnx.indoor_map import IndoorMapStore

DEFAULT_MARKER_SIZE_M = 98.0 / 128.0  # 0.765625 m (98px ArUco inside 128px block face)


@dataclass(frozen=True)
class VisualFix:
    marker_id: int
    indoor_id: str
    building_id: str
    building_name: str
    floor: str
    robot_x: float
    robot_y: float
    robot_yaw: float
    distance_m: float
    reproj_err_px: float
    label: str
    timestamp_wall: float


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def default_base_to_camera() -> tuple[np.ndarray, np.ndarray]:
    """Default ROS optical frame (+X right, +Y down, +Z forward) relative to base (+X fwd, +Y left, +Z up)."""
    r_b_c = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    t_b_c = np.array([0.0, 0.0, 1.62], dtype=np.float64)
    return r_b_c, t_b_c


def marker_world_rotation(wall_yaw: float) -> np.ndarray:
    """Rotation R_W_M from marker frame (+X right, +Y up, +Z outward wall normal) to world ENU."""
    c = math.cos(wall_yaw)
    s = math.sin(wall_yaw)
    return np.array(
        [
            [-s, 0.0, c],
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


class VisualFiducialLocalizer:
    """Detects ArUco DICT_5X5_100 wall markers in RGB frames and solves 2D/3D robot pose."""

    def __init__(
        self,
        *,
        default_size_m: float = DEFAULT_MARKER_SIZE_M,
        min_distance_m: float = 0.4,
        max_distance_m: float = 8.5,
        max_reproj_err_px: float = 3.5,
    ) -> None:
        self.default_size_m = float(default_size_m)
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.max_reproj_err_px = float(max_reproj_err_px)
        self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(self._dict, params)
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray = np.zeros((5, 1), dtype=np.float64)
        self.r_base_cam, self.t_base_cam = default_base_to_camera()

    def update_intrinsics(self, k_flat: Any, d_flat: Any = None) -> None:
        arr = np.asarray(k_flat, dtype=np.float64).reshape((3, 3))
        if arr[0, 0] > 1.0 and arr[1, 1] > 1.0:
            self.camera_matrix = arr
        if d_flat is not None and len(d_flat) >= 4:
            self.dist_coeffs = np.asarray(d_flat, dtype=np.float64).reshape((-1, 1))
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def update_extrinsics(
        self,
        tx: float,
        ty: float,
        tz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        self.r_base_cam = _quat_to_rot(qx, qy, qz, qw)
        self.t_base_cam = np.array([float(tx), float(ty), float(tz)], dtype=np.float64)

    def detect_and_localize(
        self,
        rgb_image: np.ndarray,
        indoor_store: IndoorMapStore,
    ) -> list[VisualFix]:
        if self.camera_matrix is None or rgb_image is None or rgb_image.size == 0:
            return []
        gray = (
            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            if rgb_image.ndim == 3
            else rgb_image
        )
        corners_list, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        now = time.time()
        fixes: list[VisualFix] = []
        for idx, raw_id in enumerate(ids.flatten()):
            mid = int(raw_id)
            hit = indoor_store.find_marker(mid)
            if hit is None:
                continue
            fm, mk = hit
            size_m = float(mk.get("size_m") or self.default_size_m)
            half = size_m * 0.5
            obj_pts = np.array(
                [
                    [-half, half, 0.0],
                    [half, half, 0.0],
                    [half, -half, 0.0],
                    [-half, -half, 0.0],
                ],
                dtype=np.float64,
            )
            img_pts = np.asarray(corners_list[idx], dtype=np.float64).reshape((4, 2))
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            t_c_m = tvec.reshape(3)
            dist_m = float(np.linalg.norm(t_c_m))
            if not (self.min_distance_m <= dist_m <= self.max_distance_m):
                continue

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            rmse = float(np.sqrt(np.mean(np.sum((proj.reshape((4, 2)) - img_pts) ** 2, axis=1))))
            if rmse > self.max_reproj_err_px:
                continue

            r_c_m, _ = cv2.Rodrigues(rvec)
            wall_yaw = float(mk.get("wall_yaw", 0.0))
            r_w_m = marker_world_rotation(wall_yaw)
            t_w_m = np.array(
                [float(mk["x"]), float(mk["y"]), float(mk.get("z", 0.0))],
                dtype=np.float64,
            )

            # Camera pose in world ENU: T_W_C = T_W_M * inv(T_C_M)
            r_w_c = r_w_m @ r_c_m.T
            t_w_c = t_w_m - r_w_c @ t_c_m

            # Reject physically impossible mirror/behind-wall solutions.
            wall_normal = r_w_m[:, 2]
            if float(np.dot(t_w_c - t_w_m, wall_normal)) < 0.15:
                continue

            # Robot base pose in world ENU: T_W_B = T_W_C * inv(T_B_C)
            r_w_b = r_w_c @ self.r_base_cam.T
            t_w_b = t_w_c - r_w_b @ self.t_base_cam
            robot_yaw = math.atan2(float(r_w_b[1, 0]), float(r_w_b[0, 0]))

            fixes.append(
                VisualFix(
                    marker_id=mid,
                    indoor_id=fm.indoor_id,
                    building_id=fm.building_id,
                    building_name=fm.building_name or fm.building_id,
                    floor=fm.floor,
                    robot_x=float(t_w_b[0]),
                    robot_y=float(t_w_b[1]),
                    robot_yaw=robot_yaw,
                    distance_m=dist_m,
                    reproj_err_px=rmse,
                    label=str(mk.get("label") or f"标牌#{mid}"),
                    timestamp_wall=now,
                )
            )
        fixes.sort(key=lambda f: f.distance_m)
        return fixes


class VisualRelocalizerFilter:
    """Requires consecutive consistent observations of the same marker before confirming a fix."""

    def __init__(
        self,
        *,
        min_consecutive: int = 2,
        max_gap_s: float = 1.5,
        max_jump_m: float = 0.65,
    ) -> None:
        self.min_consecutive = max(1, int(min_consecutive))
        self.max_gap_s = float(max_gap_s)
        self.max_jump_m = float(max_jump_m)
        self._last: VisualFix | None = None
        self._streak: int = 0

    def reset(self) -> None:
        self._last = None
        self._streak = 0

    def push(self, fix: VisualFix) -> VisualFix | None:
        prev = self._last
        if (
            prev is not None
            and prev.marker_id == fix.marker_id
            and prev.indoor_id == fix.indoor_id
            and (fix.timestamp_wall - prev.timestamp_wall) <= self.max_gap_s
            and math.hypot(fix.robot_x - prev.robot_x, fix.robot_y - prev.robot_y) <= self.max_jump_m
        ):
            self._streak += 1
        else:
            self._streak = 1
        self._last = fix
        if self._streak >= self.min_consecutive:
            return fix
        return None
