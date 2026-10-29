#!/usr/bin/env bash
# Build the Jetson column's 8 TensorRT engines on the board (PLATFORMS.md step 5). jetson/wizard.sh starts it under
# nohup; it also runs by hand, from anywhere:
#
#   bash ~/uav/study/jetson/export.sh [workspace_gib]
#
# Each engine is one studyexport.py call, so a failed build does not stop the others, and an engine with a manifest is
# skipped: run it again to build only what is missing. The order puts the first real test first (YOLO11s FP16 at 640)
# and INT8 at 960, the build most likely to run out of memory, last. A workspace (in GiB) caps TensorRT's builder; the
# wizard offers --workspace 1 for a build that was killed for memory, and the manifest records it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="$HOME/uav-venv/bin/python"
WS=()
[ -n "${1:-}" ] && WS=(--workspace "$1")

# run:imgsz:variant, in build order
ENGINES=(
  yolo11s-640-s0:640:tensorrt-fp16 yolo26s-640-s0:640:tensorrt-fp16
  yolo11s-640-s0:640:tensorrt-int8 yolo26s-640-s0:640:tensorrt-int8
  yolo11s-960-s0:960:tensorrt-fp16 yolo26s-960-s0:960:tensorrt-fp16
  yolo11s-960-s0:960:tensorrt-int8 yolo26s-960-s0:960:tensorrt-int8
)
failed=()
for e in "${ENGINES[@]}"; do
  IFS=: read -r run sz v <<< "$e"
  echo "=== $(date -Is) $run $sz $v ${WS[*]}"
  if [ -f "exports/$run/$sz/$v/manifest.json" ]; then
    echo "already built"
    continue
  fi
  "$PY" -u scripts/studyexport.py --weights "$HOME/uav-ckpt/$run/best.pt" --imgsz "$sz" --only "$v" ${WS[@]+"${WS[@]}"} \
    || failed+=("$run/$sz/$v")
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "=== $(date -Is) FAILED: ${failed[*]}"
  exit 1
fi
echo "=== $(date -Is) all 8 engines built"
