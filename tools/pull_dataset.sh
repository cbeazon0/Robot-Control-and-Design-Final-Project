#!/usr/bin/env bash
# Pull YOLO dataset frames from the Raspberry Pi to this laptop.
#
# Usage (run on the LAPTOP, not on the Pi):
#
#   tools/pull_dataset.sh                       # pull everything
#   tools/pull_dataset.sh Stop                  # pull only the 'Stop' label
#   tools/pull_dataset.sh Stop run_20260423_181500   # pull one specific run
#
# Environment overrides:
#   PI_USER   (default: cdslab)
#   PI_HOST   (default: cdslab.local)
#   PI_DIR    (default: ~/yolo_dataset)           # remote dataset root
#   LOCAL_DIR (default: ~/yolo_dataset)           # local dataset root
#
# Examples:
#   PI_HOST=192.168.1.42 tools/pull_dataset.sh
#   LOCAL_DIR=./datasets tools/pull_dataset.sh Right_Turn

set -euo pipefail

PI_USER="${PI_USER:-cdslab}"
PI_HOST="${PI_HOST:-cdslab.local}"
PI_DIR="${PI_DIR:-~/yolo_dataset}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/yolo_dataset}"

LABEL="${1:-}"
RUN="${2:-}"

if [[ -n "$RUN" && -z "$LABEL" ]]; then
    echo "Error: supplied a run id but no label." >&2
    echo "Usage: $0 [label] [run_id]" >&2
    exit 2
fi

if [[ -n "$LABEL" && -n "$RUN" ]]; then
    REMOTE_PATH="${PI_DIR%/}/${LABEL}/${RUN}"
    LOCAL_PATH="${LOCAL_DIR%/}/${LABEL}/"
elif [[ -n "$LABEL" ]]; then
    REMOTE_PATH="${PI_DIR%/}/${LABEL}"
    LOCAL_PATH="${LOCAL_DIR%/}/"
else
    REMOTE_PATH="${PI_DIR%/}/"
    LOCAL_PATH="${LOCAL_DIR%/}/"
fi

mkdir -p "$LOCAL_PATH"

echo "[pull] src : ${PI_USER}@${PI_HOST}:${REMOTE_PATH}"
echo "[pull] dst : ${LOCAL_PATH}"
echo

if command -v rsync >/dev/null 2>&1; then
    rsync -avh --info=progress2 \
        "${PI_USER}@${PI_HOST}:${REMOTE_PATH}" \
        "${LOCAL_PATH}"
else
    echo "[pull] rsync not found, falling back to scp -r"
    scp -r "${PI_USER}@${PI_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}"
fi

echo
echo "[pull] done -> ${LOCAL_PATH}"
