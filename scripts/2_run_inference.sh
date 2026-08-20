#!/usr/bin/env bash
# Convenience wrapper around infer.py.
#
#   bash scripts/2_run_inference.sh assets/data/clip.mp4          # one clip
#   bash scripts/2_run_inference.sh assets/data                   # a directory
#   DEVICE=cuda:1 OUT=runs/x bash scripts/2_run_inference.sh assets/data
#
# Env: DEVICE (default cuda:0) · OUT (default runs/<timestamp>) · CONFIG · EXTRA
# Each can also be set in .env (see .env.example); the shell value wins.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Load .env as defaults — a variable already set in the shell wins.
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ -z "${!key:-}" ]] && export "$key=$value"
  done < .env
fi

TARGET="${1:-assets/data}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-config/tracking.yaml}"
OUT="${OUT:-runs/$(date +%Y%m%d-%H%M%S)}"

if [[ -d "$TARGET" ]]; then SRC=(--videos-dir "$TARGET"); else SRC=(--video "$TARGET"); fi

echo "==> input  : $TARGET"
echo "==> config : $CONFIG"
echo "==> device : $DEVICE"
echo "==> output : $OUT"
python infer.py "${SRC[@]}" --out "$OUT" --config "$CONFIG" --device "$DEVICE" ${EXTRA:-}
