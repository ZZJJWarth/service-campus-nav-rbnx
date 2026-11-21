#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# Start the Google Maps-style campus mapper Web UI standalone (no rbnx/ROS2 required).
# Usage: bash scripts/start-mapper-standalone.sh [--host 0.0.0.0] [--port 8093]
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PKG/src:${PYTHONPATH:-}"
exec env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
    no_proxy='*' NO_PROXY='*' \
    python3 -u -m campus_nav_rbnx.standalone "$@"
