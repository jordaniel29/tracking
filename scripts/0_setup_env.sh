#!/usr/bin/env bash
# Environment setup for single-camera person tracking.
#
#   bash scripts/0_setup_env.sh                # conda env "tracking" + deps + model packages
#   bash scripts/0_setup_env.sh --trt          # also install TensorRT (needed for FP16 engines)
#   bash scripts/0_setup_env.sh --env myenv    # use/create a differently named conda env
#
# Creates the conda env if it is missing and installs into it, so there is
# nothing to activate beforehand. Requires conda (miniforge/miniconda) on PATH.
#
# Pins Python 3.12 EXACTLY — the pinned stack (numpy<2, torch cu124 wheels) has
# no 3.13+/3.14 builds. Also assumes an NVIDIA GPU with a recent driver.
set -euo pipefail

ENV_NAME="tracking"
PY_VERSION="3.12"
WITH_TRT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trt) WITH_TRT=1; shift ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."   # repo root

# Some images (Backend.AI) export PYTHONPATH=~/.local/lib/python3.12/site-packages.
# Left set, that directory shadows the env's own packages in every python/pip
# call below — the env then works here and breaks in a clean shell, or vice versa.
unset PYTHONPATH

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH." >&2
  echo "  Install miniforge, or re-attach an existing install for this shell:" >&2
  echo "    source /path/to/miniforge3/etc/profile.d/conda.sh" >&2
  exit 1
fi

# `conda activate` is a shell function, not the conda binary — it only exists
# after this is sourced, which `conda init` normally does from ~/.bashrc.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# First column of `conda env list`, minus the header comments and the `*`/`+`
# markers conda puts on the active and frozen envs.
env_exists() { conda env list | sed 's/\*//' | awk 'NF>0 && $1 !~ /^#/ {print $1}' | grep -qxF "$1"; }

if env_exists "$ENV_NAME"; then
  echo "==> Reusing conda env $ENV_NAME"
else
  echo "==> Creating conda env $ENV_NAME (python=$PY_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

# A pre-existing env may be on the wrong Python; the pinned wheels need 3.12.
if ! python -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
  echo "ERROR: conda env $ENV_NAME is $(python --version 2>&1), not Python $PY_VERSION." >&2
  echo "  Recreate it, or use a different name with --env:" >&2
  echo "    conda env remove -n $ENV_NAME && bash scripts/0_setup_env.sh --env $ENV_NAME --trt" >&2
  exit 1
fi
echo "==> Installing into $ENV_NAME ($(python --version 2>&1) at $(command -v python))"

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
# dev = pytest, so `python -m pytest tests` works straight after setup.
EXTRAS="hf,dev"
if [[ "$WITH_TRT" == "1" ]]; then EXTRAS="hf,trt,dev"; fi
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

  conda activate $ENV_NAME
  bash scripts/1_download_models.sh      # fetch model files into assets/models
  python infer.py --videos-dir <camera dir> --out runs/demo --device cuda:0     # multi-camera (default)
  python infer.py --mode single --video <clip.mp4> --out runs/demo             # one clip on its own

EOF
