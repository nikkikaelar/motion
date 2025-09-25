#!/usr/bin/env bash
set -euo pipefail

# Windsurf: mark executable after creation:
#   chmod +x video_tracker/run.sh
#
# Usage examples:
#   ./video_tracker/run.sh input.mp4 out.mp4
#   ./video_tracker/run.sh input.mp4 out.mp4 --only-moving --static-camera --preset quality --resolution-cap 1280 --speed-threshold 25

python3 video_tracker/app.py \
  --input "${1:-input.mp4}" \
  --output "${2:-out.mp4}" \
  "${@:3}"
