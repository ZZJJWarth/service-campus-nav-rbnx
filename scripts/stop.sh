#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
set -euo pipefail
PKG="${RBNX_PACKAGE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PATTERN="(^|/)campus_nav_rbnx$|(-m campus_nav_rbnx)$|${PKG}/src/campus_nav_rbnx"
pkill -TERM -f "$PATTERN" 2>/dev/null || true
sleep 1
pkill -KILL -f "$PATTERN" 2>/dev/null || true
