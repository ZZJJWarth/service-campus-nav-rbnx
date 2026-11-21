# SPDX-License-Identifier: MulanPSL-2.0
"""Strict configuration validation for service-campus-nav-rbnx."""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOPIC_RE = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*$")

TOP_LEVEL_KEYS = {
    "rtk_fix_topic",
    "rtk_status_topic",
    "odom_topic",
    "local_cost_grid_topic",
    "rgb_topic",
    "intrinsics_topic",
    "extrinsics_topic",
    "cmd_vel_topic",
    "visual_reloc_enabled",
    "visual_reloc_hz",
    "fiducial_size_m",
    "require_global_lock",
    "auto_relocalize_on_boot",
    "boot_rtk_wait_s",
    "relocalize_max_hops",
    "use_sim_time",
    "map_dir",
    "map_file",
    "server_host",
    "server_port",
    "elevator_config_file",
    "local_mapping_url",
    "elevator_settle_s",
    "dimension",
    "rtk_max_range_m",
    "rtk_noise_enabled",
    "require_base_station_for_mapping",
    "lookahead_m",
    "advance_radius_m",
    "indoor_lookahead_m",
    "indoor_advance_radius_m",
    "corner_clamp_deg",
    "goal_tolerance_m",
    "subgoal_timeout_s",
    "blocked_edge_ttl_s",
    "webui_enabled",
    "webui_host",
    "webui_port",
}


class ConfigError(RuntimeError):
    """Raised when deployment configuration is invalid."""


@dataclass(frozen=True)
class CampusNavConfig:
    rtk_fix_topic: str = "/minecraft/gnss/rtk_fix"
    rtk_status_topic: str = "/minecraft/gnss/rtk_status"
    odom_topic: str = "/minecraft/chassis/odom"
    local_cost_grid_topic: str = "/robonix/local_mapping/cost_grid"
    rgb_topic: str = "/minecraft/camera/rgb"
    intrinsics_topic: str = "/minecraft/camera/intrinsics"
    extrinsics_topic: str = "/minecraft/camera/extrinsics"
    cmd_vel_topic: str = "/cmd_vel"
    visual_reloc_enabled: bool = True
    visual_reloc_hz: float = 4.0
    fiducial_size_m: float = 98.0 / 128.0
    require_global_lock: bool = True
    auto_relocalize_on_boot: bool = True
    boot_rtk_wait_s: float = 3.5
    relocalize_max_hops: int = 6
    use_sim_time: bool = True
    map_dir: Path = Path("maps")
    map_file: str = "campus_default.json"
    server_host: str = "127.0.0.1"
    server_port: int = 25577
    elevator_config_file: str = ""
    local_mapping_url: str = "http://127.0.0.1:8092"
    elevator_settle_s: float = 0.6
    dimension: str = "minecraft:overworld"
    rtk_max_range_m: float = 256.0
    rtk_noise_enabled: bool = True
    require_base_station_for_mapping: bool = True
    lookahead_m: float = 14.0
    advance_radius_m: float = 4.5
    indoor_lookahead_m: float = 5.0
    indoor_advance_radius_m: float = 2.0
    corner_clamp_deg: float = 35.0
    goal_tolerance_m: float = 2.0
    subgoal_timeout_s: float = 25.0
    blocked_edge_ttl_s: float = 120.0
    webui_enabled: bool = True
    webui_host: str = "127.0.0.1"
    webui_port: int = 8093


def _num(raw: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be a number")
    try:
        val = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not math.isfinite(val) or val < lo or val > hi:
        raise ConfigError(f"{name}={raw!r} must be within [{lo}, {hi}]")
    return val


def _int(raw: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool):
        raise ConfigError(f"{name} must be an integer")
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise ConfigError(f"{name} must be an exact integer")
    if val < lo or val > hi:
        raise ConfigError(f"{name}={val} must be within [{lo}, {hi}]")
    return val


def _bool(raw: Any, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.strip().lower() in (
        "1", "true", "yes", "on", "0", "false", "no", "off",
    ):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    raise ConfigError(f"{name} must be a boolean")


def _str(raw: Any, name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return raw.strip()


def _topic(raw: Any, name: str) -> str:
    val = _str(raw, name)
    if not TOPIC_RE.fullmatch(val):
        raise ConfigError(f"{name}={val!r} must be an absolute ROS topic")
    return val


def from_config(config: dict[str, Any], package_root: Path) -> CampusNavConfig:
    if not isinstance(config, dict):
        raise ConfigError("config must be a mapping")
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)}")

    raw_dir = Path(os.path.expanduser(_str(config.get("map_dir", "maps"), "map_dir")))
    map_dir = raw_dir if raw_dir.is_absolute() else (package_root / raw_dir)
    map_file = _str(config.get("map_file", "campus_default.json"), "map_file")
    if "/" in map_file or "\\" in map_file or ".." in map_file:
        raise ConfigError("map_file must be a plain filename without path separators")

    lookahead = _num(config.get("lookahead_m", 14.0), "lookahead_m", 4, 28)
    advance = _num(config.get("advance_radius_m", 4.5), "advance_radius_m", 1.5, 12)
    if advance >= lookahead:
        raise ConfigError("advance_radius_m must be smaller than lookahead_m")
    indoor_lookahead = _num(config.get("indoor_lookahead_m", 5.0), "indoor_lookahead_m", 2.0, 12.0)
    indoor_advance = _num(config.get("indoor_advance_radius_m", 2.0), "indoor_advance_radius_m", 0.8, 6.0)
    if indoor_advance >= indoor_lookahead:
        raise ConfigError("indoor_advance_radius_m must be smaller than indoor_lookahead_m")
    corner_clamp = _num(config.get("corner_clamp_deg", 35.0), "corner_clamp_deg", 15.0, 120.0)

    raw_elev = str(config.get("elevator_config_file", "") or "").strip()
    if raw_elev:
        elev_p = Path(os.path.expanduser(raw_elev))
        elev_cfg_file = str(elev_p if elev_p.is_absolute() else (package_root / elev_p).resolve())
    else:
        elev_cfg_file = ""
    raw_lm_url = str(config.get("local_mapping_url", "http://127.0.0.1:8092") or "").strip().rstrip("/")

    return CampusNavConfig(
        rtk_fix_topic=_topic(config.get("rtk_fix_topic", "/minecraft/gnss/rtk_fix"), "rtk_fix_topic"),
        rtk_status_topic=_topic(config.get("rtk_status_topic", "/minecraft/gnss/rtk_status"), "rtk_status_topic"),
        odom_topic=_topic(config.get("odom_topic", "/minecraft/chassis/odom"), "odom_topic"),
        local_cost_grid_topic=_topic(
            config.get("local_cost_grid_topic", "/robonix/local_mapping/cost_grid"),
            "local_cost_grid_topic",
        ),
        rgb_topic=_topic(config.get("rgb_topic", "/minecraft/camera/rgb"), "rgb_topic"),
        intrinsics_topic=_topic(config.get("intrinsics_topic", "/minecraft/camera/intrinsics"), "intrinsics_topic"),
        extrinsics_topic=_topic(config.get("extrinsics_topic", "/minecraft/camera/extrinsics"), "extrinsics_topic"),
        cmd_vel_topic=_topic(config.get("cmd_vel_topic", "/cmd_vel"), "cmd_vel_topic"),
        visual_reloc_enabled=_bool(config.get("visual_reloc_enabled", True), "visual_reloc_enabled"),
        visual_reloc_hz=_num(config.get("visual_reloc_hz", 4.0), "visual_reloc_hz", 0.5, 20.0),
        fiducial_size_m=_num(config.get("fiducial_size_m", 98.0 / 128.0), "fiducial_size_m", 0.1, 3.0),
        require_global_lock=_bool(config.get("require_global_lock", True), "require_global_lock"),
        auto_relocalize_on_boot=_bool(config.get("auto_relocalize_on_boot", True), "auto_relocalize_on_boot"),
        boot_rtk_wait_s=_num(config.get("boot_rtk_wait_s", 3.5), "boot_rtk_wait_s", 0.5, 30.0),
        relocalize_max_hops=_int(config.get("relocalize_max_hops", 6), "relocalize_max_hops", 0, 15),
        use_sim_time=_bool(config.get("use_sim_time", True), "use_sim_time"),
        map_dir=map_dir,
        map_file=map_file,
        server_host=_str(config.get("server_host", "127.0.0.1"), "server_host"),
        server_port=_int(config.get("server_port", 25577), "server_port", 0, 65535),
        elevator_config_file=elev_cfg_file,
        local_mapping_url=raw_lm_url,
        elevator_settle_s=_num(config.get("elevator_settle_s", 0.6), "elevator_settle_s", 0.0, 5.0),
        dimension=_str(config.get("dimension", "minecraft:overworld"), "dimension"),
        rtk_max_range_m=_num(config.get("rtk_max_range_m", 256.0), "rtk_max_range_m", 10, 4096),
        rtk_noise_enabled=_bool(config.get("rtk_noise_enabled", True), "rtk_noise_enabled"),
        require_base_station_for_mapping=_bool(
            config.get("require_base_station_for_mapping", True),
            "require_base_station_for_mapping",
        ),
        lookahead_m=lookahead,
        advance_radius_m=advance,
        indoor_lookahead_m=indoor_lookahead,
        indoor_advance_radius_m=indoor_advance,
        corner_clamp_deg=corner_clamp,
        goal_tolerance_m=_num(config.get("goal_tolerance_m", 2.0), "goal_tolerance_m", 0.5, 6),
        subgoal_timeout_s=_num(config.get("subgoal_timeout_s", 25.0), "subgoal_timeout_s", 5, 120),
        blocked_edge_ttl_s=_num(config.get("blocked_edge_ttl_s", 120.0), "blocked_edge_ttl_s", 10, 1800),
        webui_enabled=_bool(config.get("webui_enabled", True), "webui_enabled"),
        webui_host=_str(config.get("webui_host", "127.0.0.1"), "webui_host"),
        webui_port=_int(config.get("webui_port", 8093), "webui_port", 0, 65535),
    )
