# piaspace-clip-reid

PIASPACE CLIP-ReID image encoder for cross-camera Re-Identification.
ViT-B/16 with overlapping stride-12 patches + Side Information Embedding +
BN-neck on the CLS token → **768-D L2-normalized** embedding.

Image-only at inference (no text branch). TRT-preferred at runtime; PyTorch
fallback when no engine is available.

Self-contained: depends only on external packages plus `piaspace-trt-runtime`.

## Usage

```python
from piaspace_clip_reid import CLIPReIDEmbedder

emb = CLIPReIDEmbedder({
    "kind": "person",                  # or "vehicle"
    "weights_path": "weights/MSMT17_clipreid_12x12sie_ViT-B-16_60.pth",
    # OR: "engine_path": "weights/clipreid_person.fp16.engine"
    "device": "cuda:0",
})

# BGR crops, varying sizes; resized internally to (256, 128) for person
# and (256, 256) for vehicle.
import numpy as np
crops = [np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8) for _ in range(4)]
embeddings = emb.embed(crops)        # (4, 768) float32, L2-normalized
emb.embed_dim                         # 768
```

## TRT engine auto-provisioning

If `engine_path` points to a `.engine` file that doesn't exist on disk, the
package downloads the matching ONNX from `PIA-SPACE-LAB/SSAVE` (gated HF repo;
requires `HF_TOKEN`) and builds the TRT engine on the fly via
`piaspace_trt_runtime`. The registry of provisionable engines lives in
[`engine.py`](src/piaspace_clip_reid/engine.py).

```python
from piaspace_clip_reid import ensure_engine

engine_path = ensure_engine("clipreid_person", weights_dir="./weights")
```

## Consumed via

In this repository the embedder is consumed by the tracker's appearance
channel — its `embed(crops_bgr)` + `embed_dim` surface satisfies
`pia_tracking.reid.ReIDBackend` directly (see `config/tracking.yaml`
at the repository root for the config actually used).
