# service-campus-nav-rbnx deployment configuration.
specVersion: 1
description: >-
  Google Maps-style campus vector map with handheld RTK surveyor mapping mode
  and hierarchical piecewise outdoor navigation backed by rolling local Nav2.

properties:
  rtk_fix_topic:
    type: string
    default: /minecraft/gnss/rtk_fix
    description: ROS2 PoseWithCovarianceStamped topic from onboard RTK receiver.

  rtk_status_topic:
    type: string
    default: /minecraft/gnss/rtk_status
    description: ROS2 String JSON status topic from onboard RTK receiver.

  odom_topic:
    type: string
    default: /minecraft/chassis/odom
    description: ROS2 Odometry topic for RTK-odom fusion and NO_FIX dead reckoning.

  local_cost_grid_topic:
    type: string
    default: /robonix/local_mapping/cost_grid
    description: ROS2 OccupancyGrid from local_mapping for subgoal safety nudging.

  rgb_topic:
    type: string
    default: /minecraft/camera/rgb
    description: ROS2 RGB Image topic for pure-sensor indoor ArUco visual relocalization.

  intrinsics_topic:
    type: string
    default: /minecraft/camera/intrinsics
    description: ROS2 CameraInfo topic supplying pinhole K matrix for solvePnP.

  extrinsics_topic:
    type: string
    default: /minecraft/camera/extrinsics
    description: ROS2 TransformStamped topic from base frame to camera optical frame.

  cmd_vel_topic:
    type: string
    default: /cmd_vel
    description: ROS2 Twist topic used only during explicit in-place active relocalization spin.

  visual_reloc_enabled:
    type: boolean
    default: true
    description: Enable continuous indoor ArUco 5x5 visual fiducial relocalization from RGB frames.

  visual_reloc_hz:
    type: number
    minimum: 0.5
    maximum: 20.0
    default: 4.0
    description: Maximum RGB detection rate for ArUco visual relocalization.

  fiducial_size_m:
    type: number
    minimum: 0.1
    maximum: 3.0
    default: 0.765625
    description: Default physical side length (metres) of wall-mounted ArUco marker squares.

  require_global_lock:
    type: boolean
    default: true
    description: Require an outdoor RTK fix or indoor ArUco visual marker lock before exposing global pose (realistic cold-start).

  auto_relocalize_on_boot:
    type: boolean
    default: true
    description: Automatically search for an ArUco wall marker on boot when no outdoor RTK fix arrives indoors.

  boot_rtk_wait_s:
    type: number
    minimum: 0.5
    maximum: 30.0
    default: 3.5
    description: Seconds to wait for outdoor RTK fix on boot before starting indoor visual marker search.

  relocalize_max_hops:
    type: integer
    minimum: 0
    maximum: 15
    default: 6
    description: Maximum frontier/unseen-space exploration hops if initial 360-degree marker scan sees no plaque.

  use_sim_time:
    type: boolean
    default: true
    description: Use /clock for ROS2 timestamps.

  map_dir:
    type: string
    default: maps
    description: Directory storing campus vector map JSON files (relative to package root or absolute).

  map_file:
    type: string
    default: campus_default.json
    description: Default campus map filename inside map_dir.

  server_host:
    type: string
    default: 127.0.0.1
    description: Minecraft dedicated server loopback host (used ONLY in manual mapping mode).

  server_port:
    type: integer
    minimum: 0
    maximum: 65535
    default: 25577
    description: Minecraft dedicated server loopback port for rtk_mapper_poll / rtk_stations_list; 0 disables.

  elevator_config_file:
    type: string
    default: ""
    description: Path to elevator.local.json controller bindings (empty disables elevator control).

  local_mapping_url:
    type: string
    default: http://127.0.0.1:8092
    description: Base HTTP URL of service-local-mapping-rbnx for rolling local map reset across floors.

  elevator_settle_s:
    type: number
    minimum: 0.0
    maximum: 5.0
    default: 0.6
    description: Pause after resetting local_mapping on elevator door open so live LiDAR refreshes.

  dimension:
    type: string
    default: minecraft:overworld
    description: Minecraft dimension surveyed in manual mapping mode.

  rtk_max_range_m:
    type: number
    minimum: 10
    maximum: 4096
    default: 256
    description: Maximum RTK base station radio range in metres.

  rtk_noise_enabled:
    type: boolean
    default: true
    description: Enable realistic RTK noise during manual surveyor polling.

  require_base_station_for_mapping:
    type: boolean
    default: true
    description: Require an active RTK base station fix (RTK_FIXED/RTK_FLOAT) to record survey geometry.

  lookahead_m:
    type: number
    minimum: 4
    maximum: 28
    default: 14
    description: Forward lookahead distance along the global route for each local Nav2 subgoal.

  advance_radius_m:
    type: number
    minimum: 1.5
    maximum: 12
    default: 4.5
    description: Distance from an intermediate subgoal at which the executive seamlessly advances to the next segment.

  indoor_lookahead_m:
    type: number
    minimum: 2.0
    maximum: 12.0
    default: 5.0
    description: Shorter forward lookahead distance along indoor corridor / room_entry segments.

  indoor_advance_radius_m:
    type: number
    minimum: 0.8
    maximum: 6.0
    default: 2.0
    description: Seamless subgoal advance radius on indoor corridor / room_entry segments.

  corner_clamp_deg:
    type: number
    minimum: 15.0
    maximum: 120.0
    default: 35.0
    description: Indoor corridor corner angle threshold above which lookahead clamps at the corner vertex.

  goal_tolerance_m:
    type: number
    minimum: 0.5
    maximum: 6
    default: 2.0
    description: Arrival radius for the final campus destination.

  subgoal_timeout_s:
    type: number
    minimum: 5
    maximum: 120
    default: 25
    description: Maximum seconds without progress toward a subgoal before retrying or blocking the road edge.

  blocked_edge_ttl_s:
    type: number
    minimum: 10
    maximum: 1800
    default: 120
    description: Seconds a road edge remains penalized/blocked after repeated local navigation failures.

  webui_enabled:
    type: boolean
    default: true
    description: Serve the Google Maps-style campus map and manual editor Web UI.

  webui_host:
    type: string
    default: 127.0.0.1
    description: Bind host for the campus map Web UI.

  webui_port:
    type: integer
    minimum: 0
    maximum: 65535
    default: 8093
    description: Bind port for the campus map Web UI; 0 disables.
