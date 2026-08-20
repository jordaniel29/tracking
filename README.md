# Single-camera person tracking

Person detection, re-identification and tracking for one camera at a time —
a self-contained package by PIASPACE.

```
video → YOLO26 detect → CLIP-ReID embed → BoostTrack++ associate → track ids
```

Outputs, per clip: an annotated MP4 with colour-coded ID labels, a MOTChallenge
prediction file, and a JSON run summary.

## Quick start

Requires Python 3.12 (exactly — the pinned wheels have no 3.13+ builds), an
NVIDIA GPU and a CUDA driver. If your default `python3` is not 3.12, point the
setup script at one: `PYTHON=/path/to/python3.12 bash scripts/0_setup_env.sh --trt`.

```bash
bash scripts/0_setup_env.sh --trt      # venv, PyTorch (CUDA), TensorRT, model packages
source .venv/bin/activate
cp .env.example .env                 # fill in HF_TOKEN if you were issued one
bash scripts/1_download_models.sh      # fetch the ONNX files, build the TRT engines
bash scripts/2_run_inference.sh assets/data          # process every clip in assets/data
```

`HF_TOKEN` (see `.env.example`) is only needed to *download* the `.onnx` files.
If they are already in `assets/models/` (e.g. delivered directly), leave it
empty — the script skips the download and just builds the engines.

## Layout

```
assets/
  models/       model artefacts (.onnx delivered, .engine built here) — not committed
  data/         input videos — not committed
config/
  tracking.yaml detector + reid + tracker settings, commented per knob
.env.example    template for .env — HF_TOKEN + optional runtime defaults
scripts/
  0_setup_env.sh          venv + dependencies + GPU/TensorRT verification
  1_download_models.sh    fetch the ONNX files, build the TensorRT engines
  2_run_inference.sh      convenience wrapper around infer.py
src/
  pia_tracking/ pipeline, tracker, schemas, visualisation
  models/       model packages (piaspace-yolo26, -clip-reid, -trt-runtime)
infer.py        CLI entry point
```

## Usage

```bash
# one clip
python infer.py --video assets/data/clip.mp4 --out runs/demo --device cuda:0

# a directory (models load once, reused across clips)
python infer.py --videos-dir assets/data --out runs/demo --device cuda:0

# MOT files only, no video rendering
python infer.py --videos-dir assets/data --out runs/demo --no-video

# also dump raw pre-tracking detections
python infer.py --video assets/data/clip.mp4 --out runs/demo --show-all-dets

# geometry-only tracking (no ReID) — faster, more ID switches in crowds
python infer.py --video assets/data/clip.mp4 --out runs/demo --no-reid
```

The wrapper takes the same options via env vars (or `.env`):

```bash
DEVICE=cuda:1 OUT=runs/x EXTRA="--no-video" bash scripts/2_run_inference.sh assets/data
```

To pin a GPU, use `--device cuda:N` — not `CUDA_VISIBLE_DEVICES`, which
Ultralytics rewrites internally.

Output, per clip:

```
<out>/<stem>.mp4                annotated video (box + id, colour per id)
<out>/preds/<stem>.txt          frame,id,x,y,w,h,conf,-1,-1,-1   (MOTChallenge)
<out>/preds/<stem>_dets.txt     raw detections, id = -1          (--show-all-dets)
<out>/run_summary.json          config used + per-clip frames/fps/coverage
```

## Configuration

All settings live in [`config/tracking.yaml`](config/tracking.yaml), commented
per knob. Two things to know before changing it:

- `detector.imgsz` cannot be raised against a built TRT engine — rebuild the
  engine with a wider profile instead.
- The tracker gates are tuned to this detector's confidence distribution; if
  you swap the detector or change its `conf`, re-tune them.
