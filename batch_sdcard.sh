#!/bin/bash
# Process SD-card clips through the ASL pipeline: copy locally, track, fill, render.
# Usage: ./batch_sdcard.sh <stem> [<stem> ...]
cd "$(dirname "$0")"
PY=${PY:-/usr/bin/python3}   # needs OpenCV + MediaPipe; override with PY=...
for stem in "$@"; do
  src=/Volumes/Untitled/DCIM/Camera01/$stem.mp4; dst=recordings/sdcard/$stem.mp4
  [ -s "$dst" ] || cp "$src" "$dst"
  rot=$($PY -c "import cv2;c=cv2.VideoCapture('$dst');print('cw' if c.get(3)>c.get(4) else 'none')")
  t0=$(date +%s)
  $PY asl_stream.py --video "$dst" --rotate $rot --fisheye-hfov 78 --hfov 70 --no-preview --thorough \
      --out recordings/sdcard/${stem}_world.jsonl > recordings/sdcard/${stem}_log.txt 2>&1
  $PY fill_tracks.py recordings/sdcard/${stem}_world.jsonl recordings/sdcard/${stem}_world_filled.jsonl >> recordings/sdcard/${stem}_log.txt 2>&1
  $PY render_overlay.py "$dst" recordings/sdcard/${stem}_world_filled.jsonl recordings/sdcard/${stem}_overlay.mp4 --rotate $rot >> recordings/sdcard/${stem}_log.txt 2>&1
  echo "DONE $stem rotate=$rot $(( $(date +%s) - t0 ))s $(grep -c Traceback recordings/sdcard/${stem}_log.txt) errors"
done
