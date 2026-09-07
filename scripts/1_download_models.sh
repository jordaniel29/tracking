#!/usr/bin/env bash
# Provision the model files into assets/models/.
#
#   cp .env.example .env                  # put HF_TOKEN there, if you were issued one
#   bash scripts/1_download_models.sh     # (or: export HF_TOKEN=<token> instead)
#   bash scripts/1_download_models.sh --ft  # also the fine-tuned ReID engine (config/tracking_ft.yaml)
#
# Provisions what config/tracking_general.yaml (the default) needs: the YOLO26
# detector and the stock CLIP-ReID person model. `--ft` adds the fine-tuned
# ReID engine that config/tracking_ft.yaml points at — a separate ~340 MB ONNX
# download plus its own engine build, so it is opt-in.
#
# One step per model: download the ONNX (if absent) and build a TensorRT FP16
# engine from it. Both are done by each model package's own
# `ensure_engine_by_filename`, which carries the VALIDATED optimization profile
# for that model. Do not hand-write a profile here — the input shapes a model is
# probed with (Ultralytics warms up at a smaller size than it infers at) must all
# fall inside it, and an engine whose profile is too narrow fails at load with an
# opaque CUDA "illegal memory access" rather than a clear error.
#
# A TRT engine is COMPILED FOR ONE GPU ARCHITECTURE and is not portable. An
# engine built elsewhere either refuses to load or runs measurably slower (13-22%
# measured across two GPUs of the SAME model), which is why this runs here rather
# than shipping prebuilt engines. Budget ~5-6 minutes the first time; TensorRT
# autotunes kernels per layer against your actual GPU. Results are cached, so
# re-running is instant.
set -euo pipefail

WITH_FT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ft) WITH_FT=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."   # repo root

# Load .env as defaults — a variable already set in the shell wins.
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ -z "${!key:-}" ]] && export "$key=$value"
  done < .env
fi

MODELS_DIR="assets/models"
mkdir -p "$MODELS_DIR"

echo "==> Provisioning into $MODELS_DIR$([[ "$WITH_FT" == "1" ]] && echo " (including the fine-tuned ReID engine)")"
python - "$MODELS_DIR" "$WITH_FT" <<'PY'
import os
import sys
from pathlib import Path

models_dir = sys.argv[1]
with_ft = sys.argv[2] == "1"
token = os.environ.get("HF_TOKEN") or None

try:
    from piaspace_clip_reid.engine import ensure_engine_by_filename as reid_engine
    from piaspace_yolo26.engine import ensure_engine_by_filename as det_engine
except ModuleNotFoundError as exc:
    sys.exit(f"{exc}. Run scripts/0_setup_env.sh first.")

TARGETS = [
    ("yolo26l_v6.3.fp16.engine", det_engine, "detector (YOLO26-L v6.3, person-only)"),
    ("clipreid_person.fp16.engine", reid_engine, "person ReID (CLIP-ReID ViT-B/16)"),
]
if with_ft:
    TARGETS.append(
        (
            "combined+scenario1-5_clipreid_ViT-B-16_20.fp16.engine",
            reid_engine,
            "person ReID, fine-tuned (config/tracking_ft.yaml)",
        )
    )

missing_source = []
for engine_name, provision, desc in TARGETS:
    out = Path(models_dir) / engine_name
    if out.is_file():
        print(f"    have  {out}")
        continue
    print(f"    provisioning {engine_name}  — {desc}", flush=True)
    try:
        built = provision(engine_name, weights_dir=models_dir, token=token)
    except Exception as exc:  # noqa: BLE001 — report the actionable cause
        name = f"{type(exc).__name__}: {exc}"
        if not token and ("401" in name or "403" in name or "gated" in name.lower()):
            missing_source.append(engine_name)
            print(f"    FAILED {engine_name}: source ONNX unavailable without HF_TOKEN")
            continue
        sys.exit(f"    FAILED {engine_name}: {name}")
    size_mb = Path(built).stat().st_size / (1024 * 1024)
    print(f"    built {built}  ({size_mb:.1f} MiB)")

if missing_source:
    onnx = ["yolo26l_v6.3.onnx", "clipreid_person.onnx"]
    if with_ft:
        onnx.append("combined+scenario1-5_clipreid_ViT-B-16_20.onnx")
    sys.exit(
        "\n==> Could not obtain the source ONNX for: "
        + ", ".join(missing_source)
        + "\n\nTwo ways to proceed:\n"
        "  A) If you were issued a Hugging Face token for the PIASPACE model repo,\n"
        "     put it in .env (cp .env.example .env) or export HF_TOKEN, then:\n"
        "       bash scripts/1_download_models.sh\n"
        "  B) Otherwise request the .onnx files from your PIASPACE contact and copy\n"
        f"     them into {models_dir}/ , then re-run this script to build the engines:\n"
        + "".join(f"       {n}\n" for n in onnx)
    )
PY

echo
echo "==> Verifying the engines load and infer"
python - "$MODELS_DIR" "$WITH_FT" <<'PY'
import sys
from pathlib import Path

import numpy as np

models_dir = Path(sys.argv[1])
with_ft = sys.argv[2] == "1"
device = "cuda:0"

from piaspace_yolo26 import YOLO26Detector

det = YOLO26Detector(
    {
        "backend": "yolo26",
        "model": "yolo26l_v6.3.fp16.engine",
        "weights_dir": str(models_dir),
        "device": device,
        "conf": 0.4,
        "iou": 0.5,
        "imgsz": 640,
        "target_classes": {"person": [0]},
        "min_box_size": 16,
    }
)
det.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
print("    detector OK")

from piaspace_clip_reid import CLIPReIDEmbedder

engines = [("clipreid_person.fp16.engine", "reid")]
if with_ft:
    engines.append(("combined+scenario1-5_clipreid_ViT-B-16_20.fp16.engine", "reid (fine-tuned)"))
for engine_name, label in engines:
    reid = CLIPReIDEmbedder(
        {
            "engine_path": str(models_dir / engine_name),
            "backbone": "ViT-B-16",
            "device": device,
            "input_size": [256, 128],
            "stride": 12,
        }
    )
    feats = reid.embed([np.zeros((256, 128, 3), dtype=np.uint8)])
    print(f"    {label} OK (embedding dim {feats.shape[-1]})")
PY

echo
echo "==> $MODELS_DIR"
ls -la "$MODELS_DIR"
cat <<'EOF'

==> Ready:

    bash scripts/2_run_inference.sh assets/data

EOF
