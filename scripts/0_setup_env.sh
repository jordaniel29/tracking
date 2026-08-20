#!/usr/bin/env bash
# Environment setup for single-camera person tracking.
#
#   bash scripts/0_setup_env.sh                 # venv + CPU/GPU deps + model packages
#   bash scripts/0_setup_env.sh --trt           # also install TensorRT (needed for FP16 engines)
#   bash scripts/0_setup_env.sh --venv .myenv   # custom venv path
#   PYTHON=/path/to/python3.12 bash scripts/0_setup_env.sh --trt   # explicit interpreter
#
# Needs Python 3.12 EXACTLY — the pinned stack (numpy<2, torch cu124 wheels)
# has no 3.13+/3.14 builds. Also assumes an NVIDIA GPU with a recent driver.
set -euo pipefail

VENV=".venv"
WITH_TRT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trt)  WITH_TRT=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."   # repo root

# ── Interpreter: $PYTHON override → python3.12 on PATH → python3 if it is 3.12
is_py312() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; }
PYBIN=""
for cand in "${PYTHON:-}" python3.12 python3; do
  [[ -n "$cand" ]] || continue
  command -v "$cand" >/dev/null 2>&1 || continue
  if is_py312 "$cand"; then PYBIN="$(command -v "$cand")"; break; fi
done
if [[ -z "$PYBIN" ]]; then
  echo "ERROR: Python 3.12 is required and was not found (python3 is $(python3 --version 2>&1 || echo missing))." >&2
  echo "  Install one and point the script at it, e.g. with conda:" >&2
  echo "    conda create -n py312 python=3.12" >&2
  echo "    PYTHON=\$(conda run -n py312 which python) bash scripts/0_setup_env.sh --trt" >&2
  exit 1
fi
echo "==> Using $PYBIN ($("$PYBIN" --version 2>&1))"

if [[ -e "$VENV" ]]; then
  if ! is_py312 "$VENV/bin/python"; then
    echo "ERROR: $VENV exists but is not a Python 3.12 venv. Remove it and re-run:" >&2
    echo "    rm -rf $VENV" >&2
    exit 1
  fi
  echo "==> Reusing venv at $VENV"
else
  echo "==> Creating venv at $VENV"
  "$PYBIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch (CUDA build)"
# Pinned to a CUDA index so a CPU-only wheel is not silently selected — a CPU
# torch will import fine and then fail at engine load with a confusing error.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing bundled model packages"
# Order matters: both others depend on trt-runtime.
pip install -e ./src/models/piaspace-trt-runtime
pip install -e ./src/models/piaspace-clip-reid
pip install -e ./src/models/piaspace-yolo26

echo "==> Installing this package"
EXTRAS="hf"
if [[ "$WITH_TRT" == "1" ]]; then EXTRAS="hf,trt"; fi
pip install -e ".[${EXTRAS}]"

if [[ "$WITH_TRT" == "1" ]]; then
  echo "==> Verifying TensorRT"
  python - <<'PY'
try:
    import tensorrt as trt
    print(f"    TensorRT {trt.__version__} OK")
except Exception as e:
    print(f"    WARNING: TensorRT import failed: {e}")
    print("    FP16 .engine inference will not work. Options:")
    print("      - install the TensorRT matching your CUDA version, or")
    print("      - point the config at a .pt/.onnx detector instead of an .engine")
PY
fi

echo "==> Verifying GPU visibility"
python - <<'PY'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"    cuda:{i}  {torch.cuda.get_device_name(i)}")
else:
    print("    WARNING: no CUDA device visible — inference will not run.")
PY

cat <<EOF

==> Done. Next steps:

  source $VENV/bin/activate
  bash scripts/1_download_models.sh      # fetch model files into assets/models
  python infer.py --video <clip.mp4> --out runs/demo --device cuda:0

EOF
