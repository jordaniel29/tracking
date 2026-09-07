#!/usr/bin/env bash
# Convenience wrapper around infer.py.
#
#   bash scripts/2_run_inference.sh assets/data/03_scenarios/scenario_01   # camera directory → multi-camera (default)
#   MODE=single bash scripts/2_run_inference.sh assets/data               # each clip on its own
#   bash scripts/2_run_inference.sh assets/data/clip.mp4                  # one file
#   for d in assets/data/03_scenarios/scenario_*; do bash scripts/2_run_inference.sh "$d"; done
#   MODE=render OUT=runs/compare/trace_ft/scenario_01 bash scripts/2_run_inference.sh assets/data/03_scenarios/scenario_01
#
# Env: MODE (multi | single | render, default multi) · DEVICE (default cuda:0)
#      OUT (default runs/<input name>) · CONFIG · EXTRA (extra infer.py flags, e.g. "--final-labels")
# Each can also be set in .env (see .env.example); the shell value wins.
# Files named grid_* are skipped by default (infer.py --exclude).
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
MODE="${MODE:-multi}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-config/tracking.yaml}"
name="$(basename "${TARGET%.*}")"
OUT="${OUT:-runs/$name}"

if [[ -d "$TARGET" ]]; then SRC=(--videos-dir "$TARGET"); else SRC=(--video "$TARGET"); fi

echo "==> input  : $TARGET"
echo "==> mode   : $MODE"
echo "==> config : $CONFIG"
echo "==> device : $DEVICE"
echo "==> output : $OUT"
python infer.py --mode "$MODE" "${SRC[@]}" --out "$OUT" --config "$CONFIG" --device "$DEVICE" ${EXTRA:-}
