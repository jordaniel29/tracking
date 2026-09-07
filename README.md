# Person tracking — single camera, or several with shared ids

Person detection, re-identification and tracking — a self-contained package by
PIASPACE. One CLI, `infer.py`: a set of simultaneously recorded cameras linked by
cross-camera global ids (`--mode multi`, the default), or each clip on its own
(`--mode single`).

```
per camera:  video → YOLO26 detect → CLIP-ReID embed → BoostTrack++ associate → local track ids
all cameras: track's mean embedding → Global ID service (shared) → same person = same G-<n> everywhere
```

Outputs, per clip: an annotated MP4 with colour-coded ID labels, a MOTChallenge
prediction file, and a JSON run summary.

## Quick start

Requires conda (miniforge or miniconda), an NVIDIA GPU and a CUDA driver. The
setup script creates a conda env named `tracking` on Python 3.12 — exactly 3.12,
the pinned wheels have no 3.13+ builds — so there is nothing to create or
activate first.

```bash
bash scripts/0_setup_env.sh --trt      # conda env, PyTorch (CUDA), TensorRT, model packages
conda activate tracking
cp .env.example .env                 # fill in HF_TOKEN if you were issued one
bash scripts/1_download_models.sh      # fetch the ONNX files, build the TRT engines
bash scripts/2_run_inference.sh assets/data/03_scenarios/scenario_01   # one set of cameras → global ids
```

Pass `--env <name>` to use a different env name. Re-running the script is safe:
an existing env is reused, not recreated.

`HF_TOKEN` (see `.env.example`) is only needed to *download* the `.onnx` files.
If they are already in `assets/models/` (e.g. delivered directly), leave it
empty — the script skips the download and just builds the engines.

## Layout

```
assets/
  models/       model artefacts (.onnx delivered, .engine built here) — not committed
  data/         input videos — not committed
config/
  tracking_general.yaml   default — detector + reid + tracker + global-id settings, commented per knob
  tracking_ft.yaml        same profile with the fine-tuned ReID model (only the reid block differs)
.env.example    template for .env — HF_TOKEN + optional runtime defaults
scripts/
  0_setup_env.sh          conda env + dependencies + GPU/TensorRT verification
  1_download_models.sh    fetch the ONNX files, build the TensorRT engines
  2_run_inference.sh      convenience wrapper around infer.py (MODE=multi|single)
src/
  pia_tracking/           one folder per stage — data flows camera → detection → tracking → fusion → utils
    schemas.py            Detection / Track — the data contracts everything passes around
    config.py             config yaml → dict
    camera/               video sources (discovery, opening, shared clock) + round-robin frame sync
    detection/            detector factory (YOLO26) + output normalisation
    reid/                 ReIDBackend interface + embedder factory (CLIP-ReID)
    tracking/             single-camera: Tracker contracts, BoostTrack++, matching math, TrackingPipeline
    fusion/               cross-camera: Tracklet + accumulator, GlobalIDService, CameraWorker
    utils/                crops, drawing, MOT / MP4 / JSON writers
    runners/              run_single / run_multi — the end-to-end runs behind infer.py
  models/                 model packages (piaspace-yolo26, -clip-reid, -trt-runtime)
infer.py        CLI — --mode multi (default: cameras linked by global id) or --mode single
tests/          pytest — global-id decision rules and worker policy, no models needed
```

## Usage

One CLI, two modes. `--mode multi` is the default.

```bash
# multi (default): every video in the directory is a camera; tracks linked across cameras
python infer.py --videos-dir assets/data/03_scenarios/scenario_01 --out runs/scenario_01 --device cuda:0
python infer.py --video cam8.mp4 cam9.mp4 --out runs/pair                 # explicit camera files

# single: each clip tracked on its own with local ids (the original behaviour)
python infer.py --mode single --video assets/data/clip.mp4 --out runs/demo
python infer.py --mode single --videos-dir assets/data --out runs/demo   # models load once, reused
python infer.py --mode single --video clip.mp4 --out runs/demo --no-reid # geometry only: faster, more ID switches

# render: draw an existing run's predictions onto its source videos (no models, CPU only) —
# e.g. to get MP4s for a run made with --no-video. --out is that run's directory.
python infer.py --mode render --videos-dir assets/data/03_scenarios/scenario_01 --out runs/scenario_01

# grid: one <out>/grid.mp4 with every camera tiled and time-aligned — the clearest way to see
# cross-camera identity (same person = same colour + G-<n> in every cell at the same instant)
python infer.py --mode grid --videos-dir assets/data/03_scenarios/scenario_01 --out runs/scenario_01
python infer.py --mode grid --videos-dir DIR --out OUT --grid-cols 5 --grid-width 2560   # one row, larger

# both tracking modes
python infer.py --videos-dir DIR --out OUT --no-video          # MOT files only
python infer.py --videos-dir DIR --out OUT --show-all-dets     # also dump raw pre-tracking detections
python infer.py --videos-dir DIR --out OUT --exclude 'grid_*' 'test_*'   # skip files (default: grid_*)
```

The wrapper takes the same options via env vars (or `.env`); `MODE` picks the mode:

```bash
bash scripts/2_run_inference.sh assets/data/03_scenarios/scenario_01                   # multi
MODE=single DEVICE=cuda:1 EXTRA="--no-video" bash scripts/2_run_inference.sh assets/data
for d in assets/data/03_scenarios/scenario_*; do bash scripts/2_run_inference.sh "$d"; done
```

To pin a GPU, use `--device cuda:N` — not `CUDA_VISIBLE_DEVICES`, which
Ultralytics rewrites internally.

Output (both modes):

```
<out>/<stem>.mp4                annotated video (box + id label, colour per id)
<out>/preds/<stem>.txt          frame,id,x,y,w,h,conf,-1,-1,-1   (MOTChallenge, local id)
<out>/preds/<stem>_dets.txt     raw detections, id = -1          (--show-all-dets)
<out>/run_summary.json          config used + per-clip frames/fps/coverage
```

## Multi-camera (cross-camera global ids)

In `--mode multi` the per-camera pipeline runs on every video at once and one
`GlobalIDService` is shared by all cameras, so a person carries the same `G-<n>`
label on every camera they appear on. One directory = one set of simultaneously
recorded cameras (frame *k* of each is the same instant); `grid_*` files are
skipped by default. A single video is also valid — re-entries on that camera are
then linked.

How a track gets its id (knobs in the `global_id:` block of the config):

1. Per camera, the ReID embeddings the tracker already computed are averaged
   per local track (`pia_tracking.fusion.TrackAccumulator`).
2. After `checkpoint_frames` (40) embedded frames a still-visible track asks the
   service for an id. A track that ends earlier asks once, at loss, if it has at
   least `min_frames_before_id_assign` (10) frames — shorter tracks stay unlabelled.
3. The service (`pia_tracking.fusion.GlobalIDService`) compares the mean
   embedding with every identity seen within `reidentify_within_sec` (600 s) on
   any camera, skipping identities that were on screen on the *same* camera at
   the *same* time (two boxes at once are two people). Cosine ≥
   `similarity_threshold` (0.45) reuses that id; otherwise a new one is minted.
4. When the track ends, its remaining frames are folded into the identity and
   its full evidence is re-scored; if another identity wins by
   `revise_at_loss.margin` the track moves there — the early guess is provisional.

Frames are interleaved across cameras (frame *k* of every camera, then *k*+1),
so an id decision only ever sees the past — the same causal order as one live
worker per camera. Timestamps are synthetic, `frame_idx / fps` from a shared
t = 0, which assumes the cameras started recording together.

The video labels are what was known *at that frame*, like a live view — a
track that a loss-time revision moves to another identity wears its provisional
id in the earlier frames. `--final-labels` re-renders after tracking with the
final ids instead, so the MP4 matches `preds/<cam>_global.txt` exactly (one
extra decode/encode pass, no GPU).

Additional output in multi mode:

```
<out>/<cam>.mp4                 boxes labelled G-<gid>, one colour per global id on every camera;
                                grey + local id until the id is known (or final ids with --final-labels)
<out>/preds/<cam>_global.txt    MOT rows with the global id (unlabelled tracks omitted)
<out>/global_ids.json           identities → (camera, local id) members; local→global map per camera
<out>/run_summary.json          also: per-camera fps + identity counts
<out>/grid.mp4                  all cameras tiled into one video (`--mode grid`)
```

Before matching, embeddings are mean-centred per camera (`global_id.percam_norm`,
on by default): different people on one camera share its colour balance,
exposure and "everyone wears dark clothes" bias, and subtracting the camera's
running mean removes that shared component so the gate separates people rather
than cameras. `similarity_threshold: 0.45` is paired with centering on; with it
off, ~0.40 is the better starting point.

## Configuration

All settings live in [`config/tracking_general.yaml`](config/tracking_general.yaml)
(the default), commented per knob; [`config/tracking_ft.yaml`](config/tracking_ft.yaml)
is the same profile with the fine-tuned ReID model — pass it with `--config`
(or `CONFIG=` for the wrapper). Two things to know before changing them:

- `detector.imgsz` cannot be raised against a built TRT engine — rebuild the
  engine with a wider profile instead.
- The tracker gates are tuned to this detector's confidence distribution; if
  you swap the detector or change its `conf`, re-tune them.
