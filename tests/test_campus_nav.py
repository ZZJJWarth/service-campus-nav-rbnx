# SPDX-License-Identifier: MulanPSL-2.0
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pytest

from campus_nav_rbnx import params
from campus_nav_rbnx.campus_map import CampusMap, IndoorSurveyRecorder, SurveyRecorder
from campus_nav_rbnx.elevator_client import ElevatorClient
from campus_nav_rbnx.indoor_map import IndoorFloorSurveyRecorder, IndoorMapStore
from campus_nav_rbnx.executive import (
    HierarchicalNavExecutive,
    LocalCostGridSnapshot,
    polyline_lookahead,
)
from campus_nav_rbnx.global_planner import GlobalPlanner
from campus_nav_rbnx.fiducial_localizer import (
    DEFAULT_MARKER_SIZE_M,
    VisualFiducialLocalizer,
    VisualRelocalizerFilter,
    default_base_to_camera,
    marker_world_rotation,
)
from campus_nav_rbnx.rtk_fusion import RtkOdomFusion
from campus_nav_rbnx.standalone import StandaloneCampusMapper
from campus_nav_rbnx.webui import CampusWebUI

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "campus_nav_rbnx"


def test_params_validation(tmp_path):
    cfg = params.from_config({}, tmp_path)
    assert cfg.webui_port == 8093
    assert cfg.lookahead_m == 14.0
    assert cfg.advance_radius_m == 4.5
    with pytest.raises(params.ConfigError):
        params.from_config({"unknown_key": 1}, tmp_path)
    with pytest.raises(params.ConfigError):
        params.from_config({"lookahead_m": 6.0, "advance_radius_m": 8.0}, tmp_path)


def test_campus_map_persistence_and_poi_lookup(tmp_path):
    cmap = CampusMap(campus_id="test_campus", name="测试园区")
    n1 = cmap.add_node(0.0, 0.0, kind="entrance")
    n2 = cmap.add_node(20.0, 0.0, kind="junction")
    cmap.add_edge(n1["id"], n2["id"], name="主干道", road_class="primary", width_m=5.0)
    cmap.add_area("实验楼", "building", [[0, 5], [10, 5], [10, 15], [0, 15]])
    cmap.add_poi("饭堂", 20.0, 0.0, category="canteen", aliases=["canteen", "食堂"])

    path = tmp_path / "campus.json"
    cmap.save(path)
    loaded = CampusMap.load(path)
    assert loaded.name == "测试园区"
    assert len(loaded.areas) == 1
    assert len(loaded.nodes) == 2
    assert len(loaded.edges) == 1
    assert loaded.find_poi("食堂")["name"] == "饭堂"
    assert loaded.find_poi("canteen")["node_id"] == n2["id"]


def test_survey_recorder_road_junction_snap_and_area_and_right_click():
    cmap = CampusMap()
    rec = SurveyRecorder(node_spacing_m=4.0, merge_radius_m=2.0)
    rec.start("road", road_class="primary", road_name="中轴路")

    # NO_FIX must not record when require_rtk=True.
    res = rec.ingest_sample(cmap, 0.0, 0.0, "NO_FIX", require_rtk=True)
    assert not res["changed"]
    assert len(cmap.nodes) == 0

    # Walk a loop and return near start -> must snap & form a junction.
    for x, y in [(0, 0), (5, 0), (10, 0), (10, 5), (5, 5), (0, 5), (0.5, 0.5)]:
        rec.ingest_sample(cmap, float(x), float(y), "RTK_FIXED", require_rtk=True)
    assert len(cmap.nodes) >= 5
    assert len(cmap.edges) >= 5
    assert any(n["kind"] == "junction" for n in cmap.nodes)

    # Area building corners via right-click + Shift+right-click (action=close).
    rec.start("area", area_kind="building", area_name="实验楼")
    for idx, (x, y) in enumerate([(20, 0), (25, 0), (25, 5), (20, 5)], start=1):
        rec.ingest_sample(
            cmap,
            float(x),
            float(y),
            "RTK_FIXED",
            mark_event={"seq": idx, "action": "mark", "fix": {"valid": True, "east": x, "north": y}},
        )
    res_close = rec.ingest_sample(
        cmap,
        20.0,
        5.0,
        "RTK_FIXED",
        mark_event={"seq": 5, "action": "close", "fix": {"valid": True, "east": 20.0, "north": 5.0}},
    )
    assert res_close["changed"] and res_close["auto_saved"]
    assert any(a["name"] == "实验楼" and a["kind"] == "building" and len(a["polygon"]) == 4 for a in cmap.areas)

    # Right-click in POI mode drops a named landmark.
    rec.start("poi", poi_category="canteen", poi_name="饭堂")
    mark = {"seq": 7, "action": "mark", "fix": {"valid": True, "east": 12.0, "north": -3.0}}
    rec.ingest_sample(cmap, 12.0, -3.0, "RTK_FIXED", mark_event=mark)
    assert any(p["name"] == "饭堂" and p["category"] == "canteen" for p in cmap.pois)


def test_rtk_fusion_handles_offset_and_no_fix_dead_reckoning():
    fusion = RtkOdomFusion()
    fusion.update_odom(10.0, 20.0, 0.5)
    # Verify numpy covariance array handling (mirrors ROS2 PoseWithCovarianceStamped.pose.covariance)
    cov = np.zeros(36, dtype=np.float64)
    cov[0] = 0.0025
    sigma_xy = float(np.sqrt(max(1e-6, float(cov[0])))) if (cov is not None and len(cov) > 0) else 0.05
    fusion.update_rtk_fix(110.0, 220.0, 0.5, sigma_xy=sigma_xy)
    pose = fusion.campus_pose()
    assert pose is not None
    assert pose[0] == pytest.approx(110.0)
    assert pose[1] == pytest.approx(220.0)

    # Enter building (NO_FIX): odom moves +3 m east; fused campus pose coasts smoothly!
    fusion.update_rtk_status({"valid": False, "fix_type": "NO_FIX"})
    fusion.update_odom(13.0, 20.0, 0.5)
    pose2 = fusion.campus_pose()
    assert pose2 is not None
    assert pose2[0] == pytest.approx(113.0)
    assert pose2[1] == pytest.approx(220.0)
    assert fusion.campus_to_odom(113.0, 220.0) == pytest.approx((13.0, 20.0))


def test_global_planner_astar_and_blocked_edge_rerouting():
    cmap = CampusMap()
    a = cmap.add_node(0.0, 0.0)
    b = cmap.add_node(10.0, 0.0)   # short direct top road
    c = cmap.add_node(20.0, 0.0)
    d = cmap.add_node(10.0, 15.0)  # longer detour road
    e_ab = cmap.add_edge(a["id"], b["id"])
    cmap.add_edge(b["id"], c["id"])
    cmap.add_edge(a["id"], d["id"])
    cmap.add_edge(d["id"], c["id"])

    planner = GlobalPlanner(blocked_edge_ttl_s=60.0)
    route1 = planner.plan(cmap, (0.0, 0.0), (20.0, 0.0))
    assert route1 is not None
    assert b["id"] in route1.node_ids

    # Block direct edge -> must detour through d!
    planner.mark_edge_blocked(e_ab["id"])
    route2 = planner.plan(cmap, (0.0, 0.0), (20.0, 0.0))
    assert route2 is not None
    assert d["id"] in route2.node_ids
    assert b["id"] not in route2.node_ids


def test_cost_grid_nudge_and_lookahead():
    data = np.zeros((20, 20), dtype=np.int8)
    data[10, 10] = 100  # lethal at world (0.1, 0.1)
    grid = LocalCostGridSnapshot(origin_x=-2.0, origin_y=-2.0, resolution=0.2, width=20, height=20, data=data)
    nx, ny = grid.nudge_to_passable(0.1, 0.1, search_radius_m=1.0)
    assert grid.cost_at(nx, ny) == 0

    pt, yaw, is_final, seg, rem = polyline_lookahead([(0.0, 0.0), (30.0, 0.0)], (0.0, 0.0), 14.0)
    assert pt[0] == pytest.approx(14.0)
    assert not is_final
    assert rem == pytest.approx(30.0)


def test_hierarchical_executive_piecewise_advance_and_reroute(tmp_path):
    cfg = params.from_config({"lookahead_m": 10.0, "advance_radius_m": 3.0, "goal_tolerance_m": 1.5}, tmp_path)
    cmap = CampusMap()
    a = cmap.add_node(0.0, 0.0)
    b = cmap.add_node(15.0, 0.0)
    c = cmap.add_node(30.0, 0.0)
    d = cmap.add_node(15.0, 12.0)
    cmap.add_edge(a["id"], b["id"])
    cmap.add_edge(b["id"], c["id"])
    cmap.add_edge(a["id"], d["id"])
    cmap.add_edge(d["id"], c["id"])
    cmap.add_poi("饭堂", 30.0, 0.0, node_id=c["id"])

    fusion = RtkOdomFusion()
    fusion.update_odom(0.0, 0.0, 0.0)
    fusion.update_rtk_fix(0.0, 0.0, 0.0, 0.05)
    planner = GlobalPlanner(blocked_edge_ttl_s=60.0)

    goals_sent: list[tuple[float, float]] = []
    run_counter = [0]

    def fake_rpc(op, payload, timeout_s=10.0):
        if op == "navigate":
            run_counter[0] += 1
            pos = payload["goal"]["pose"]["position"]
            goals_sent.append((pos["x"], pos["y"]))
            return {"accepted": True, "run_id": f"loc-{run_counter[0]}", "detail": "ok"}
        if op == "status":
            # Move robot directly toward last subgoal so it advances across segments!
            gx, gy = goals_sent[-1]
            fusion.update_odom(gx, gy, 0.0)
            return {"known": True, "state": "SUCCEEDED", "detail": "arrived"}
        if op == "cancel":
            return {"accepted": True, "detail": "canceled"}
        return {"ok": False}

    exe = HierarchicalNavExecutive(
        cfg=cfg,
        map_provider=lambda: cmap,
        fusion=fusion,
        planner=planner,
        cost_grid_provider=lambda: None,
        rpc_fn=fake_rpc,
        tick_s=0.02,
    )
    res = exe.start_run(place="饭堂", timeout_s=5.0)
    assert res["ok"] is True
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and exe.snapshot()["state"] == "RUNNING":
        time.sleep(0.02)
    snap = exe.snapshot()
    assert snap["state"] == "SUCCEEDED"
    assert len(goals_sent) >= 2  # Piecewise: at least 2 subgoals for a 30m route with 10m lookahead!
    exe.shutdown()


def test_webui_serves_google_maps_editor_and_endpoints():
    webui = CampusWebUI(
        "127.0.0.1",
        0,
        {
            "GET:api/state": lambda _: {"ok": True, "map": CampusMap().to_dict()},
            "POST:api/navigate": lambda body: {"ok": True, "place": body.get("place", "")},
        },
    )
    webui.start()
    try:
        with urlopen(f"{webui.url}/", timeout=3) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "Robonix Campus Map" in html
            assert "rtk_mapper" in html
        with urlopen(f"{webui.url}/api/state", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["ok"] is True
        with urlopen(
            Request(
                f"{webui.url}/api/navigate",
                data=json.dumps({"place": "饭堂"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=3,
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["place"] == "饭堂"
    finally:
        webui.stop()


def test_standalone_campus_mapper_serves_and_saves_without_rbnx(tmp_path):
    cfg = params.from_config(
        {
            "webui_host": "127.0.0.1",
            "webui_port": 0,
            "server_host": "127.0.0.1",
            "server_port": 1,  # unused unreachable port; poll error is caught gracefully
            "map_dir": str(tmp_path),
            "map_file": "campus_test.json",
        },
        tmp_path,
    )
    app = StandaloneCampusMapper(cfg)
    app.start()
    try:
        with urlopen(f"{app._webui.url}/api/state", timeout=3) as resp:
            state = json.loads(resp.read().decode("utf-8"))
            assert state["ok"] is True
            assert state["standalone"] is True
            assert state["active_file"] == "campus_test.json"
        # Save a building via /api/map/replace and verify persistence on disk
        cmap = CampusMap(campus_id="campus_test", name="测试校区")
        cmap.add_area("实验楼", "building", [[0, 0], [10, 0], [10, 10], [0, 10]])
        with urlopen(
            Request(
                f"{app._webui.url}/api/map/replace",
                data=json.dumps({"map": cmap.to_dict(), "save": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=3,
        ) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            assert out["ok"] is True
        saved = CampusMap.load(tmp_path / "campus_test.json")
        assert len(saved.areas) == 1 and saved.areas[0]["name"] == "实验楼"
        # Reset map via /api/map/reset and verify empty persistence
        with urlopen(
            Request(f"{app._webui.url}/api/map/reset", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"),
            timeout=3,
        ) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            assert out["ok"] is True
        cleared = CampusMap.load(tmp_path / "campus_test.json")
        assert cleared.areas == [] and cleared.nodes == [] and cleared.edges == [] and cleared.pois == []
        # Exercise /api/indoor/* multi-floor endpoints end-to-end (isolated from outdoor map!)
        for ep, payload in [
            ("api/indoor/select_floor", {"building_id": "b_lab", "building_name": "实验楼", "floor": "1F"}),
            ("api/indoor/anchor_entrance", {"building_id": "b_lab", "floor": "1F", "x": 80.0, "y": -8.0}),
            ("api/indoor/mark_room_door", {"building_id": "b_lab", "floor": "1F", "x": 80.0, "y": -14.0, "room_name": "101机房"}),
            ("api/indoor/mark_room_nav", {"building_id": "b_lab", "floor": "1F", "x": 75.0, "y": -14.0, "room_name": "101机房"}),
            ("api/indoor/corridor_toggle", {"action": "resume", "building_id": "b_lab", "floor": "1F", "x": 80.0, "y": -14.0}),
            ("api/indoor/select_floor", {"building_id": "b_lab", "building_name": "实验楼", "floor": "2F"}),
            ("api/indoor/mark_room_door", {"building_id": "b_lab", "floor": "2F", "x": 80.0, "y": -14.0, "room_name": "201会议室"}),
            ("api/indoor/mark_room_nav", {"building_id": "b_lab", "floor": "2F", "x": 75.0, "y": -14.0, "room_name": "201会议室"}),
        ]:
            with urlopen(
                Request(
                    f"{app._webui.url}/{ep}",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=3,
            ) as resp:
                out = json.loads(resp.read().decode("utf-8"))
                assert out["ok"] is True
        # Outdoor map remains completely clean (no indoor rooms mixed in)
        outdoor_reloaded = CampusMap.load(tmp_path / "campus_test.json")
        assert outdoor_reloaded.nodes == [] and outdoor_reloaded.pois == []
        # Indoor store holds separate 1F and 2F floor maps!
        store = IndoorMapStore(tmp_path / "indoor")
        assert len(store.all_maps()) == 2
        hit1 = store.find_room("101机房")
        hit2 = store.find_room("201会议室")
        assert hit1 is not None and hit1[0].floor == "1F" and hit1[1]["door_node_id"]
        assert hit2 is not None and hit2[0].floor == "2F" and hit2[1]["door_node_id"]
        # Resetting 2F leaves 1F untouched!
        with urlopen(
            Request(
                f"{app._webui.url}/api/indoor/reset_floor",
                data=json.dumps({"indoor_id": hit2[0].indoor_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=3,
        ) as resp:
            assert json.loads(resp.read().decode("utf-8"))["ok"] is True
        store.reload()
        assert store.find_room("101机房") is not None
        assert store.find_room("201会议室") is None
    finally:
        app.stop()


def test_multifloor_bridged_planning_outdoor_to_1f_and_intra_2f(tmp_path):
    outdoor = CampusMap(campus_id="campus_multi", name="多楼层园区")
    out1 = outdoor.add_node(80.0, 10.0, kind="waypoint")
    out2 = outdoor.add_node(80.0, -5.0, kind="waypoint")
    outdoor.add_edge(out1["id"], out2["id"], road_class="walkway")
    bldg = outdoor.add_area("实验楼", "building", [[50, -6], [100, -6], [100, -40], [50, -40]])

    store = IndoorMapStore(tmp_path / "indoor")
    f1 = store.get_or_create(bldg["id"], "实验楼", "1F", elevation_y=-60.0)
    f2 = store.get_or_create(bldg["id"], "实验楼", "2F", elevation_y=-55.0)
    rec = IndoorFloorSurveyRecorder(corridor_spacing_m=2.5, outdoor_snap_m=8.0)

    # Map 1F with outdoor-linked entrance + corridor + room 101
    rec.anchor_entrance(outdoor, f1, 80.0, -8.0, elevation_y=-60.0)
    assert f1.entrances[0]["outdoor_node_id"] == out2["id"]
    for x, y in [(80.0, -11.0), (80.0, -14.0)]:
        rec.ingest_sample(outdoor, f1, x, y, anchored=True)
    rec.mark_room_door(f1, 80.0, -17.0, room_name="101实验室")
    r101 = rec.mark_room_nav(f1, 74.0, -17.0, room_name="101实验室")
    store.save(f1)

    # Map 2F separately with 201 and 202 (same x/y footprint, zero overlap with 1F!)
    rec.reset()
    rec.anchor_entrance(outdoor, f2, 80.0, -8.0, elevation_y=-55.0)
    assert f2.entrances[0]["outdoor_node_id"] == ""  # 2F does NOT erroneously link to outdoor ground!
    rec.mark_room_door(f2, 80.0, -12.0, room_name="201室")
    r201 = rec.mark_room_nav(f2, 74.0, -12.0, room_name="201室")
    rec.resume_corridor(f2, 80.0, -12.0)
    rec.ingest_sample(outdoor, f2, 80.0, -16.0, anchored=True)
    rec.mark_room_door(f2, 80.0, -20.0, room_name="202室")
    r202 = rec.mark_room_nav(f2, 74.0, -20.0, room_name="202室")
    store.save(f2)

    planner = GlobalPlanner()
    # Outdoor (80, 10) -> 1F 101实验室 bridges cleanly across entrance!
    route_out_in = planner.plan_unified(
        outdoor,
        store.all_maps(),
        (80.0, 10.0),
        (74.0, -17.0),
        goal_indoor_id=f1.indoor_id,
        goal_node_id=r101["nav_node"]["id"],
    )
    assert route_out_in is not None
    assert "entrance" in route_out_in.point_kinds
    assert "room_door" in route_out_in.point_kinds
    assert "room_nav" in route_out_in.point_kinds

    # 2F 201室 -> 2F 202室 routes purely inside 2F!
    route_2f = planner.plan_unified(
        outdoor,
        store.all_maps(),
        (74.0, -12.0),
        (74.0, -20.0),
        start_indoor_id=f2.indoor_id,
        goal_indoor_id=f2.indoor_id,
        goal_node_id=r202["nav_node"]["id"],
    )
    assert route_2f is not None
    assert r201["door_node_id"] in route_2f.node_ids
    assert r202["door_node_id"] in route_2f.node_ids
    assert r202["nav_node"]["id"] in route_2f.node_ids

    # Regression: walking back from a corridor dead-end and marking a room_door MUST attach
    # to the nearest corridor node/segment, NOT jump 20m back to the dead-end!
    rec.reset()
    f3 = store.get_or_create(bldg["id"], "实验楼", "3F", elevation_y=-50.0)
    rec.anchor_entrance(outdoor, f3, 90.0, -20.0)
    for y in (-23.0, -26.0, -29.0, -32.0, -35.0, -40.0, -45.0, -50.0):
        rec.ingest_sample(outdoor, f3, 90.0, y, anchored=True)
    rec.pause_corridor()
    door_back = rec.mark_room_door(f3, 90.2, -24.5, room_name="304")
    f3_nodes = f3.node_map()
    attached_edges = [e for e in f3.edges if door_back["id"] in (e["from"], e["to"])]
    assert len(attached_edges) >= 1
    for e in attached_edges:
        other_id = e["to"] if e["from"] == door_back["id"] else e["from"]
        other = f3_nodes[other_id]
        assert np.hypot(other["x"] - door_back["x"], other["y"] - door_back["y"]) < 3.0


def test_fiducial_localizer_synthetic_projection_and_relocalization(tmp_path):
    import cv2

    # 1. Register ArUco marker #7 on a North-facing wall in 实验楼 2F
    store = IndoorMapStore(tmp_path / "indoor")
    f2 = store.get_or_create("b_lab", "实验楼", "2F", elevation_y=-55.0)
    mx, my, mz = 80.5, -17.9375, -53.5
    wall_yaw = np.pi / 2.0  # facing +North
    size_m = DEFAULT_MARKER_SIZE_M
    f2.upsert_marker(7, mx, my, mz, wall_yaw, label="201门牌", size_m=size_m)
    store.save(f2)

    # 2. Synthesize a 640x480 RGB camera frame from a known true robot pose facing the wall!
    true_rx, true_ry, true_rz = 81.1, -14.7, -55.0
    true_yaw = -np.pi / 2.0 + 0.12  # facing slightly rotated South toward the marker
    r_w_b = np.array([
        [np.cos(true_yaw), -np.sin(true_yaw), 0.0],
        [np.sin(true_yaw),  np.cos(true_yaw), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    t_w_b = np.array([true_rx, true_ry, true_rz], dtype=np.float64)
    r_b_c, t_b_c = default_base_to_camera()
    r_w_c = r_w_b @ r_b_c
    t_w_c = t_w_b + r_w_b @ t_b_c

    r_w_m = marker_world_rotation(wall_yaw)
    t_w_m = np.array([mx, my, mz], dtype=np.float64)
    r_c_m = r_w_c.T @ r_w_m
    t_c_m = r_w_c.T @ (t_w_m - t_w_c)

    K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    half = size_m * 0.5
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(r_c_m)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t_c_m, K, np.zeros((5, 1)))
    dst_pts = img_pts.reshape((4, 2)).astype(np.float32)

    # Render crisp ArUco #7 plaque with white border and warp onto synthetic 640x480 corridor wall
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    raw = cv2.aruco.generateImageMarker(aruco_dict, 7, 7)
    inner = cv2.resize(raw, (98, 98), interpolation=cv2.INTER_NEAREST)
    tex = np.full((128, 128), 248, dtype=np.uint8)
    tex[15:113, 15:113] = inner
    src_pts = np.array([[15, 15], [113, 15], [113, 113], [15, 113]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    bg = np.full((480, 640, 3), 130, dtype=np.uint8)
    warped = cv2.warpPerspective(cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB), H, (640, 480))
    mask = cv2.warpPerspective(np.full((128, 128), 255, dtype=np.uint8), H, (640, 480))
    bg[mask > 0] = warped[mask > 0]

    # 3. Run pure-sensor VisualFiducialLocalizer + 2-frame consistency filter + RtkOdomFusion!
    loc = VisualFiducialLocalizer(default_size_m=size_m)
    loc.update_intrinsics(K.flatten())
    filt = VisualRelocalizerFilter(min_consecutive=2)

    fixes1 = loc.detect_and_localize(bg, store)
    assert len(fixes1) == 1
    assert fixes1[0].marker_id == 7
    assert fixes1[0].floor == "2F"
    assert filt.push(fixes1[0]) is None  # 1st frame alone must NOT snap

    fixes2 = loc.detect_and_localize(bg, store)
    confirmed = filt.push(fixes2[0])
    assert confirmed is not None
    assert confirmed.robot_x == pytest.approx(true_rx, abs=0.08)
    assert confirmed.robot_y == pytest.approx(true_ry, abs=0.08)
    assert abs(((confirmed.robot_yaw - true_yaw + np.pi) % (2 * np.pi)) - np.pi) < np.radians(3.0)

    # 4. Inject 5 m indoor odom drift and verify visual fix relocalizes & locks floor!
    fusion = RtkOdomFusion()
    fusion.update_odom(true_rx - 5.0, true_ry + 4.0, true_yaw)
    fusion.update_rtk_status({"valid": False, "fix_type": "NO_FIX"})
    updated = fusion.update_visual_fix(
        confirmed.robot_x,
        confirmed.robot_y,
        confirmed.robot_yaw,
        indoor_id=confirmed.indoor_id,
        building_name=confirmed.building_name,
        floor=confirmed.floor,
        marker_id=confirmed.marker_id,
        distance_m=confirmed.distance_m,
    )
    assert updated is True
    pose = fusion.campus_pose()
    assert pose is not None
    assert pose[0] == pytest.approx(true_rx, abs=0.10)
    assert pose[1] == pytest.approx(true_ry, abs=0.10)
    assert fusion.current_indoor_id == f2.indoor_id
    assert fusion.current_floor == "2F"


def test_indoor_survey_recorder_entrance_corridor_and_paired_room_points(tmp_path):
    cmap = CampusMap(campus_id="hybrid_test", name="室内外测试园区")
    # Existing outdoor road right outside building door
    out_n = cmap.add_node(80.0, -5.0, kind="waypoint")
    out_n2 = cmap.add_node(80.0, 10.0, kind="waypoint")
    cmap.add_edge(out_n["id"], out_n2["id"], road_class="walkway")
    cmap.add_area("实验楼", "building", [[50, -6], [100, -6], [100, -40], [50, -40]])

    irec = IndoorSurveyRecorder(corridor_spacing_m=2.5, merge_radius_m=1.5, outdoor_snap_m=6.0)
    # 1. Anchor entrance at building doorway (80, -8) -> snaps to outdoor node (80, -5)!
    ent = irec.anchor_entrance(cmap, 80.0, -8.0, floor="1F")
    assert ent["kind"] == "entrance"
    assert cmap.has_edge(out_n["id"], ent["id"])
    assert irec.state == "corridor"

    # 2. Walk down corridor -> auto-drops corridor nodes every ~2.5 m
    for x, y in [(80.0, -11.0), (80.0, -14.0), (80.0, -17.0)]:
        irec.ingest_sample(cmap, x, y, anchored=True)
    corridor_nodes = [n for n in cmap.nodes if n["kind"] == "corridor"]
    assert len(corridor_nodes) >= 3

    # 3. At room doorway: mark room_door -> pauses corridor auto-drop
    door = irec.mark_room_door(cmap, 80.0, -18.0, room_name="101实验室")
    assert door["kind"] == "room_door"
    assert irec.state == "awaiting_room_nav"
    n_before_walk = len(cmap.nodes)

    # Walk around inside room -> MUST NOT drop stray corridor nodes!
    for x, y in [(77.0, -18.0), (74.0, -19.0), (72.0, -21.0)]:
        irec.ingest_sample(cmap, x, y, anchored=True)
    assert len(cmap.nodes) == n_before_walk

    # 4. Inside room: mark room_nav -> connects room_entry edge to door + creates room POI + stays paused!
    res = irec.mark_room_nav(cmap, 74.0, -20.0, room_name="101实验室")
    assert res["nav_node"]["kind"] == "room_nav"
    assert res["poi"]["category"] == "room"
    assert res["poi"]["door_node_id"] == door["id"]
    assert res["poi"]["node_id"] == res["nav_node"]["id"]
    assert any(e["road_class"] == "room_entry" and e["from"] == door["id"] for e in cmap.edges)
    assert irec.state == "in_room_paused"

    # Walking back toward corridor while paused still MUST NOT auto-drop nodes (manual resume only!)
    n_after_room = len(cmap.nodes)
    irec.ingest_sample(cmap, 80.0, -18.0, anchored=True)
    irec.ingest_sample(cmap, 80.0, -22.0, anchored=True)
    assert len(cmap.nodes) == n_after_room

    # 5. Manual resume -> continues corridor from doorway!
    irec.resume_corridor(cmap, 80.0, -18.0)
    assert irec.state == "corridor"
    irec.ingest_sample(cmap, 80.0, -21.5, anchored=True)
    assert len(cmap.nodes) == n_after_room + 1

    # 6. Seamless outdoor -> entrance -> corridor -> room_door -> room_nav A* planning & door clamping!
    planner = GlobalPlanner()
    route = planner.plan(cmap, (80.0, 10.0), (74.0, -20.0), goal_node_id=res["nav_node"]["id"])
    assert route is not None
    assert "entrance" in route.point_kinds
    assert "room_door" in route.point_kinds
    assert "room_nav" in route.point_kinds

    # Verify door clamping: approaching entrance from outdoor (80, -3) with 14m lookahead MUST clamp at entrance (80, -8)
    pt_ent, _, is_fin, _, _ = polyline_lookahead(
        route.points,
        (80.0, -3.0),
        14.0,
        point_kinds=route.point_kinds,
        seg_classes=route.seg_classes,
        indoor_lookahead_m=5.0,
        corner_clamp_deg=35.0,
    )
    assert not is_fin
    assert pt_ent == pytest.approx((80.0, -8.0))

    # Approaching room_door from corridor (80, -15) MUST clamp at room_door (80, -18) instead of cutting through wall to (74, -20)!
    pt_door, _, is_fin2, _, _ = polyline_lookahead(
        route.points,
        (80.0, -15.0),
        14.0,
        point_kinds=route.point_kinds,
        seg_classes=route.seg_classes,
        indoor_lookahead_m=5.0,
        corner_clamp_deg=35.0,
    )
    assert not is_fin2
    assert pt_door == pytest.approx((80.0, -18.0))

    # Persistence round-trip preserves indoor metadata + door_node_id
    path = tmp_path / "hybrid.json"
    cmap.save(path)
    loaded = CampusMap.load(path)
    poi_loaded = loaded.find_poi("101实验室")
    assert poi_loaded is not None
    assert poi_loaded["door_node_id"] == door["id"]
    assert any(n["kind"] == "room_door" for n in loaded.nodes)
    assert any(n["kind"] == "room_nav" for n in loaded.nodes)


class _FakeMovingCabinTransport:
    """Simulates the dedicated-server 25577 moving-cabin elevator protocol."""

    def __init__(self, cfg: dict, initial_floor: str = "lab_4f") -> None:
        self.cfg = cfg
        self.cabin_floor = initial_floor
        self.doors_open: dict[str, bool] = {fk: False for fk in cfg["floors"]}
        self.cabin_door_open: bool = False
        self.game_time = 100
        self.cmd_ticks: dict[tuple[int, int, int], int] = {}
        self.history: list[tuple[str, str]] = []
        self._pending_move: tuple[str, int] | None = None  # (target_floor, arrive_tick)
        # Map controller coordinates -> (floor_key, action)
        self.ctrl_map: dict[tuple[int, int, int], tuple[str, str]] = {}
        for fk, fcfg in cfg["floors"].items():
            for act in ("door_open_controller", "door_close_controller", "target_controller"):
                xyz = tuple(int(v) for v in fcfg[act].split("@")[1].split(","))
                self.ctrl_map[xyz] = (fk, act)

    def _advance(self, ticks: int = 1) -> None:
        self.game_time += ticks
        if self._pending_move is not None and self.game_time >= self._pending_move[1]:
            self.cabin_floor = self._pending_move[0]
            self._pending_move = None

    def request(self, payload: dict) -> dict:
        cmd = payload["cmd"]
        pos = (int(payload["x"]), int(payload["y"]), int(payload["z"]))
        self._advance(2)
        if cmd == "elevator_controller_set":
            if payload.get("powered"):
                fk, act = self.ctrl_map[pos]
                self.cmd_ticks[pos] = self.game_time
                if act == "door_open_controller":
                    assert self.cabin_floor == fk, f"open called on {fk} while cabin at {self.cabin_floor}"
                    self.doors_open[fk] = True
                    self.cabin_door_open = True
                    self.history.append(("open", fk))
                elif act == "door_close_controller":
                    self.doors_open[fk] = False
                    if self.cabin_floor == fk:
                        self.cabin_door_open = False
                    self.history.append(("close", fk))
                elif act == "target_controller":
                    assert not any(self.doors_open.values()) and not self.cabin_door_open
                    self.cabin_floor = ""  # in transit
                    self._pending_move = (fk, self.game_time + 6)
                    self.history.append(("move", fk))
            return {"game_time": self.game_time, "powered": bool(payload.get("powered"))}

        assert cmd == "elevator_controller_get"
        out: dict = {
            "game_time": self.game_time,
            "command_tick": self.cmd_ticks.get(pos, 0),
            "command_success": 1 if pos in self.cmd_ticks else 0,
        }
        dr = payload.get("door_region")
        if isinstance(dr, dict):
            mn, mx = dr["min"], dr["max"]
            vol = (mx[0] - mn[0] + 1) * (mx[1] - mn[1] + 1) * (mx[2] - mn[2] + 1)
            # Check if this is a cabin floor/ceiling plane probe (note: adjacent floors share a y-plane,
            # e.g., lab_3f ceiling y=-46 == lab_4f floor y=-46, so check whether the ACTIVE cabin owns this plane!)
            is_plane = any(
                mn[0] == fcfg["cabin_bounds"]["min"][0]
                and mx[0] == fcfg["cabin_bounds"]["max"][0]
                and mn[1] == mx[1]
                and mn[1] in (fcfg["cabin_bounds"]["min"][1] - 1, fcfg["cabin_bounds"]["max"][1] + 1)
                for fcfg in self.cfg["floors"].values()
            )
            if is_plane:
                cur_cfg = self.cfg["floors"].get(self.cabin_floor)
                present = bool(
                    cur_cfg is not None
                    and mn[1] in (cur_cfg["cabin_bounds"]["min"][1] - 1, cur_cfg["cabin_bounds"]["max"][1] + 1)
                )
                out["door_region"] = {
                    "total_blocks": vol,
                    "air_blocks": 0 if present else vol,
                    "closed_blocks": 0,
                }
            else:
                # Hall door or cabin door check
                is_cabin_door = dr.get("closed_block") == "minecraft:glass_pane"
                opened = False
                for fk, fcfg in self.cfg["floors"].items():
                    ref = fcfg["cabin_door_region"] if is_cabin_door else fcfg["door_region"]
                    if ref["min"] == mn and ref["max"] == mx:
                        opened = self.cabin_door_open if is_cabin_door else self.doors_open[fk]
                        break
                out["door_region"] = {
                    "total_blocks": vol,
                    "air_blocks": vol if opened else 0,
                    "closed_blocks": 0 if opened else vol,
                }
        return out


def test_elevator_client_moving_cabin_lifecycle():
    elev_cfg_path = ROOT.parent / "robot-syswonder-minecraft_bot" / "config" / "elevator.local.json"
    cfg = json.loads(elev_cfg_path.read_text(encoding="utf-8"))
    transport = _FakeMovingCabinTransport(cfg, initial_floor="lab_4f")
    client = ElevatorClient(
        cfg,
        transport=transport,
        pulse_ticks=2,
        settle_ticks=4,
        poll_interval_s=0.01,
    )
    assert client.resolve_floor("1F") == "lab_1f"
    assert client.resolve_floor("三楼") == "lab_3f"
    assert client.floor_elevation_y("2F") == pytest.approx(-55.0)
    # Safe cabin interior target must be deep inside usable interior (x in [85.75..87.25], clear of x=84 glass door)
    safe_xy = client.safe_cabin_interior_xy("3F")
    assert safe_xy is not None
    assert 85.7 <= safe_xy[0] <= 87.3 and -28.5 <= safe_xy[1] <= -26.5
    # Doorway threshold (x=84.4) must NOT count as safely inside cabin!
    assert client.is_xy_safely_inside_cabin("3F", 84.4, -27.5) is False
    assert client.is_xy_safely_inside_cabin("3F", safe_xy[0], safe_xy[1]) is True
    # Standing safely inside cabin off-center in Y (e.g. (86.57, -26.49), 1.01m from center) MUST still count as inside!
    assert client.is_xy_safely_inside_cabin("4F", 86.57, -26.49) is True
    # Hallway outside landing door (x=81.5) is safely outside; doorway (x=83.2) is NOT!
    assert client.is_xy_safely_outside_cabin("3F", 83.2, -27.5) is False
    assert client.is_xy_safely_outside_cabin("3F", 81.5, -27.5) is True
    assert client.current_floor() == "lab_4f"

    # Call cabin from 4F down to 1F and open both doors
    client.call_and_open("1F")
    assert client.current_floor() == "lab_1f"
    assert client.is_door_open("1F") is True

    # Close 1F door and ride to 3F
    client.close_door("1F")
    assert client.is_door_open("1F") is False
    client.move_cabin("3F")
    assert client.current_floor() == "lab_3f"
    client.call_and_open("3F")
    assert client.is_door_open("3F") is True
    client.close_door("3F")
    assert transport.history == [
        ("move", "lab_1f"),
        ("open", "lab_1f"),
        ("close", "lab_1f"),
        ("move", "lab_3f"),
        ("open", "lab_3f"),
        ("close", "lab_3f"),
    ]


def test_multifloor_elevator_planning_and_executive_end_to_end(tmp_path):
    """Verify Outdoor -> 实验室 3F 301 decomposes into walk -> elevator -> walk and executes cleanly."""
    # Load the real mapped campus + 4-floor lab maps!
    cmap = CampusMap.load(ROOT / "maps" / "campus_default.json")
    store = IndoorMapStore(ROOT / "maps" / "indoor")
    hit_301 = store.find_room("301")
    assert hit_301 is not None
    fm_3f, rm_301 = hit_301
    assert fm_3f.floor == "3F"

    planner = GlobalPlanner()
    # Start outdoors near campus entrance
    start_xy = (118.0, -30.0)
    route = planner.plan_unified(
        cmap,
        store.all_maps(),
        start_xy,
        (rm_301["x"], rm_301["y"]),
        start_indoor_id="",
        goal_indoor_id=fm_3f.indoor_id,
        goal_node_id=rm_301["node_id"],
    )
    assert route is not None
    assert len(route.legs) == 3
    assert [lg.kind for lg in route.legs] == ["walk", "elevator", "walk"]
    assert route.legs[1].from_floor == "1F"
    assert route.legs[1].to_floor == "3F"
    assert route.legs[1].from_door_xy is not None
    assert route.legs[1].from_cabin_xy is not None
    # Leg 1 must stop outside the 1F elevator door (x ~ 81.3), NOT inside the cabin (x ~ 86.0)
    assert route.legs[0].goal_xy == route.legs[1].from_door_xy
    assert route.legs[0].goal_xy[0] < 83.0

    # Also test 3F -> 4F direct single ride (no intermediate 2F/1F stop!)
    hit_401 = store.find_room("401")
    assert hit_401 is not None
    fm_4f, rm_401 = hit_401
    route_3_4 = planner.plan_unified(
        cmap,
        store.all_maps(),
        (rm_301["x"], rm_301["y"]),
        (rm_401["x"], rm_401["y"]),
        start_indoor_id=fm_3f.indoor_id,
        goal_indoor_id=fm_4f.indoor_id,
        goal_node_id=rm_401["node_id"],
    )
    assert route_3_4 is not None
    assert [lg.kind for lg in route_3_4.legs] == ["walk", "elevator", "walk"]
    assert route_3_4.legs[1].from_floor == "3F"
    assert route_3_4.legs[1].to_floor == "4F"

    # End-to-end executive execution with simulated Nav2 + FakeMovingCabinTransport!
    elev_cfg_path = ROOT.parent / "robot-syswonder-minecraft_bot" / "config" / "elevator.local.json"
    elev_cfg = json.loads(elev_cfg_path.read_text(encoding="utf-8"))
    transport = _FakeMovingCabinTransport(elev_cfg, initial_floor="lab_4f")
    elev_client = ElevatorClient(
        elev_cfg, transport=transport, pulse_ticks=2, settle_ticks=4, poll_interval_s=0.01
    )

    fusion = RtkOdomFusion()
    fusion.update_odom(start_xy[0], start_xy[1], 0.0, z=-60.0)
    fusion.update_rtk_fix(start_xy[0], start_xy[1], 0.0, sigma_xy=0.05)

    events: list[str] = []
    nav_counter = [0]

    def fake_rpc(op: str, payload: dict, timeout_s: float) -> dict:
        if op == "navigate":
            nav_counter[0] += 1
            pos = payload["goal"]["pose"]["position"]
            ori = payload["goal"]["pose"]["orientation"]
            yaw = 2.0 * np.arctan2(ori["z"], ori["w"])
            # Move simulated robot immediately to subgoal
            fusion.update_odom(float(pos["x"]), float(pos["y"]), float(yaw))
            return {"accepted": True, "run_id": f"loc-{nav_counter[0]}"}
        if op == "status":
            return {"state": "SUCCEEDED", "detail": "arrived"}
        if op == "cancel":
            return {"accepted": True}
        return {}

    # Simulate physical lift moving robot vertically along with the cabin
    orig_move = elev_client.move_cabin

    def move_cabin_with_robot(target_fl: str, **kwargs):
        res_fk = orig_move(target_fl, **kwargs)
        pose = fusion.campus_pose()
        if pose is not None and elev_client.is_xy_safely_inside_cabin(target_fl, pose[0], pose[1], margin_m=0.35):
            ey = elev_client.floor_elevation_y(target_fl)
            if ey is not None and fusion.odom_pose is not None:
                fusion.update_odom(fusion.odom_pose[0], fusion.odom_pose[1], fusion.odom_pose[2], z=ey)
        return res_fk

    elev_client.move_cabin = move_cabin_with_robot  # type: ignore[method-assign]

    def fake_reset_local_map() -> bool:
        events.append("reset_local_map")
        return True

    def fake_floor_changed(iid: str, bname: str, fl: str) -> None:
        fusion.set_indoor_floor(iid, bname, fl)
        events.append(f"floor:{fl}")

    cfg = params.from_config({"elevator_settle_s": 0.01}, tmp_path)
    exec_obj = HierarchicalNavExecutive(
        cfg=cfg,
        map_provider=lambda: cmap,
        fusion=fusion,
        planner=planner,
        cost_grid_provider=lambda: None,
        indoor_store_provider=lambda: store,
        rpc_fn=fake_rpc,
        elevator_client=elev_client,
        reset_local_map_fn=fake_reset_local_map,
        on_floor_changed=fake_floor_changed,
        tick_s=0.01,
    )
    res = exec_obj.start_run(place="301", start_indoor_id="", timeout_s=15.0)
    assert res["ok"] and res["accepted"]

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        snap = exec_obj.snapshot()
        if snap["state"] in ("SUCCEEDED", "FAILED", "CANCELED"):
            break
        time.sleep(0.02)

    snap = exec_obj.snapshot()
    assert snap["state"] == "SUCCEEDED", f"unexpected nav state: {snap}"
    assert fusion.current_indoor_id == fm_3f.indoor_id
    assert fusion.current_floor == "3F"
    # Must have reset local_mapping before board, on floor switch, and before exit (>= 3 resets)
    assert events.count("reset_local_map") >= 3
    assert "floor:3F" in events
    # Elevator physically moved 4F -> 1F -> open -> close -> 3F -> open -> close
    assert transport.history == [
        ("move", "lab_1f"),
        ("open", "lab_1f"),
        ("close", "lab_1f"),
        ("move", "lab_3f"),
        ("open", "lab_3f"),
        ("close", "lab_3f"),
    ]


def test_require_global_lock_cold_start_and_local_corridor_explore():
    # 1. Indoor cold boot with require_global_lock=True: odom alone MUST NOT expose campus_pose!
    f_in = RtkOdomFusion(require_global_lock=True)
    f_in.update_odom(72.8, -16.1, 0.0, z=-60.0)
    f_in.update_rtk_status({"valid": False, "fix_type": "NO_FIX"})
    assert f_in.campus_pose() is None
    assert f_in.snapshot()["globally_localized"] is False

    # Once an indoor ArUco visual marker locks -> unlocks global pose & floor!
    assert f_in.update_visual_fix(
        72.8, -16.1, 0.0,
        indoor_id="a_aa681194__1F",
        building_name="实验室",
        floor="1F",
        marker_id=21,
        distance_m=3.2,
    )
    assert f_in.campus_pose() == pytest.approx((72.8, -16.1, 0.0))
    assert f_in.snapshot()["globally_localized"] is True
    assert f_in.snapshot()["localization_source"] == "visual"

    # 2. Outdoor cold boot with require_global_lock=True: RTK_FIXED immediately unlocks!
    f_out = RtkOdomFusion(require_global_lock=True)
    f_out.update_odom(10.0, 20.0, 0.5)
    assert f_out.campus_pose() is None
    f_out.update_rtk_fix(110.0, 220.0, 0.5, sigma_xy=0.05)
    assert f_out.campus_pose() == pytest.approx((110.0, 220.0, 0.5))
    assert f_out.snapshot()["localization_source"] == "rtk"

    # Manual invalidate must drop localization completely until re-locked
    f_out.invalidate()
    assert f_out.campus_pose() is None
    assert f_out.snapshot()["invalidated"] is True
    assert f_out.snapshot()["globally_localized"] is False
    f_out.update_rtk_fix(110.0, 220.0, 0.5, sigma_xy=0.05)
    assert f_out.campus_pose() == pytest.approx((110.0, 220.0, 0.5))
    assert f_out.snapshot()["invalidated"] is False

    # 3. LocalCostGridSnapshot.pick_safe_explore_target selects open corridor cell away from walls
    # Build a 20x20 m costmap (0.5 m res = 40x40) with walls everywhere except an eastbound corridor y in [-1.5, 1.5]
    data = np.full((40, 40), 100, dtype=np.int8)
    # Free corridor along y in [-1.5, 1.5] (rows 17..23), x from -1.5 to +8.0 (cols 17..36)
    data[17:23, 17:36] = 0
    grid = LocalCostGridSnapshot(origin_x=-10.0, origin_y=-10.0, resolution=0.5, width=40, height=40, data=data)
    cand = grid.pick_safe_explore_target(0.0, 0.0, [(0.0, 0.0)])
    assert cand is not None
    tx, ty, _ = cand
    assert tx >= 3.0 and abs(ty) <= 1.0


def test_inflight_visual_floor_mismatch_triggers_automatic_replan(tmp_path):
    """If robot started nav thinking it was outdoors/1F, spotting a 4F marker mid-way must
    automatically replan to 4F -> Elevator(4F->3F) -> 3F 303 instead of calling 1F elevator!
    """
    cmap = CampusMap.load(ROOT / "maps" / "campus_default.json")
    store = IndoorMapStore(ROOT / "maps" / "indoor")
    elev_cfg_path = ROOT.parent / "robot-syswonder-minecraft_bot" / "config" / "elevator.local.json"
    elev_cfg = json.loads(elev_cfg_path.read_text(encoding="utf-8"))
    transport = _FakeMovingCabinTransport(elev_cfg, initial_floor="lab_4f")
    elev_client = ElevatorClient(
        elev_cfg, transport=transport, pulse_ticks=2, settle_ticks=4, poll_interval_s=0.01
    )

    # 1. Verify leaked RTK inside multi-floor building does NOT unlock cold start
    f_leak = RtkOdomFusion(require_global_lock=True)
    f_leak.update_odom(90.5, -31.7, 0.0, z=-45.0)
    f_leak.update_rtk_fix(90.5, -31.7, 0.0, sigma_xy=0.05, inside_multifloor_building=True, inside_building=True)
    assert f_leak.campus_pose() is None

    # 2. Simulate robot on 4F mistakenly starting nav with start_indoor_id="" toward 3F 303
    fusion = RtkOdomFusion(require_global_lock=True)
    fusion.update_odom(90.5, -31.7, 0.0, z=-45.0)
    fusion.update_rtk_fix(90.5, -31.7, 0.0, sigma_xy=0.05, inside_multifloor_building=False, inside_building=False)
    assert fusion.current_indoor_id == ""

    planner = GlobalPlanner()
    nav_calls = [0]
    replan_injected = [False]

    orig_move = elev_client.move_cabin

    def move_cabin_with_robot(target_fl: str, **kwargs):
        res_fk = orig_move(target_fl, **kwargs)
        ey = elev_client.floor_elevation_y(target_fl)
        if ey is not None and fusion.odom_pose is not None:
            fusion.update_odom(fusion.odom_pose[0], fusion.odom_pose[1], fusion.odom_pose[2], z=ey)
        return res_fk

    elev_client.move_cabin = move_cabin_with_robot  # type: ignore[method-assign]

    exec_holder: list[HierarchicalNavExecutive] = []

    def fake_rpc(op: str, payload: dict, timeout_s: float) -> dict:
        if op == "navigate":
            nav_calls[0] += 1
            pos = payload["goal"]["pose"]["position"]
            ori = payload["goal"]["pose"]["orientation"]
            yaw = 2.0 * np.arctan2(ori["z"], ori["w"])
            fusion.update_odom(float(pos["x"]), float(pos["y"]), float(yaw))
            # On the 1st subgoal, robot's camera spots a 4F ArUco marker!
            if not replan_injected[0] and exec_holder:
                replan_injected[0] = True
                old_p = fusion.campus_pose()
                fusion.update_visual_fix(
                    90.5, -31.0, float(yaw),
                    indoor_id="a_aa681194__4F",
                    building_name="实验室",
                    floor="4F",
                    marker_id=16,
                    distance_m=4.5,
                )
                triggered = exec_holder[0].notify_visual_fix(
                    "a_aa681194__4F",
                    "实验室4F",
                    old_xy=(old_p[0], old_p[1]) if old_p else None,
                    new_xy=(90.5, -31.0),
                )
                assert triggered is True
            return {"accepted": True, "run_id": f"loc-{nav_calls[0]}"}
        if op == "status":
            return {"state": "SUCCEEDED", "detail": "arrived"}
        if op == "cancel":
            return {"accepted": True}
        return {}

    cfg = params.from_config({"elevator_settle_s": 0.01}, tmp_path)
    exec_obj = HierarchicalNavExecutive(
        cfg=cfg,
        map_provider=lambda: cmap,
        fusion=fusion,
        planner=planner,
        cost_grid_provider=lambda: None,
        indoor_store_provider=lambda: store,
        rpc_fn=fake_rpc,
        elevator_client=elev_client,
        reset_local_map_fn=lambda: True,
        on_floor_changed=lambda iid, bn, fl: fusion.set_indoor_floor(iid, bn, fl),
        tick_s=0.01,
    )
    exec_holder.append(exec_obj)

    res = exec_obj.start_run(place="303", start_indoor_id="", timeout_s=15.0)
    assert res["ok"] and res["accepted"]

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        snap = exec_obj.snapshot()
        if snap["state"] in ("SUCCEEDED", "FAILED", "CANCELED"):
            break
        time.sleep(0.02)

    snap = exec_obj.snapshot()
    assert snap["state"] == "SUCCEEDED", f"unexpected state: {snap}"
    assert fusion.current_floor == "3F"
    # Critically: elevator MUST have boarded on lab_4f (NOT dispatched to lab_1f!) and ridden lab_4f -> lab_3f!
    assert ("move", "lab_1f") not in transport.history
    assert ("open", "lab_4f") in transport.history
    assert ("move", "lab_3f") in transport.history


def test_visual_coverage_frontier_explorer_escapes_enclosed_room():
    """After a 360-degree scan inside an enclosed room with an open doorway,
    `pick_frontier_explore_target` MUST escape through the open doorway into the unseen hallway
    rather than circling inside the already-seen room!
    """
    # 24m x 24m local map at 0.2m res = 120x120 cells, origin (-12, -12)
    res = 0.2
    grid_data = np.full((120, 120), -1, dtype=np.int8)  # unknown outside

    def fill_box(x0, x1, y0, y1, val):
        c0 = max(0, int(round((x0 + 12.0) / res)))
        c1 = min(120, int(round((x1 + 12.0) / res)))
        r0 = max(0, int(round((y0 + 12.0) / res)))
        r1 = min(120, int(round((y1 + 12.0) / res)))
        grid_data[r0:r1, c0:c1] = val

    # Room interior: x in [-4, 2], y in [-3, 3] (free = 0)
    fill_box(-4.0, 2.0, -3.0, 3.0, 0)
    # Walls around room (cost = 100)
    fill_box(-4.4, -4.0, -3.4, 3.4, 100)
    fill_box(2.0, 2.4, -3.4, 3.4, 100)
    fill_box(-4.4, 2.4, -3.4, -3.0, 100)
    fill_box(-4.4, 2.4, 3.0, 3.4, 100)
    # Open doorway on east wall at x in [2.0, 2.4], y in [-1.0, 1.0] (with wall-aversion 45 near edges, 0 center)
    fill_box(2.0, 2.4, -1.0, 1.0, 45)
    fill_box(2.0, 2.4, -0.6, 0.6, 0)
    # Partially observed hallway outside doorway: x in [2.4, 7.0], y in [-1.5, 4.0] (bordering unknown -1 beyond!)
    fill_box(2.4, 7.0, -1.5, 4.0, 0)

    grid = LocalCostGridSnapshot(origin_x=-12.0, origin_y=-12.0, resolution=res, width=120, height=120, data=grid_data)

    # Robot stands in room corner (-1.5, -1.5) and spins 360 deg (marking the whole room as visually seen!)
    seen: set[tuple[int, int]] = set()
    for deg in range(0, 360, 30):
        grid.mark_camera_fov_seen(-1.5, -1.5, np.radians(deg), seen)

    # Pick next exploration target: MUST escape through doorway (x > 2.4) into the unseen hallway/frontier!
    target = grid.pick_frontier_explore_target(-1.5, -1.5, seen, [(-1.5, -1.5)])
    assert target is not None
    tx, ty, _, _ = target
    assert tx > 2.4, f"expected explorer to exit room through doorway x>2.4, got ({tx:.2f}, {ty:.2f})"


def test_no_game_state_references_in_src():
    forbidden = (
        "ground" + "_truth",
        "perception" + "/objects",
        "get" + "_state",
        "SCENE" + "_",
    )
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pat in forbidden:
            if re.search(pat, text, flags=re.IGNORECASE):
                offenders.append(f"{path.name}: {pat}")
    assert offenders == []
