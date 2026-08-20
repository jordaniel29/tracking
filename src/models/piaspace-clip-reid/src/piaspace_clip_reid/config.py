"""Constants for CLIP-ReID (Syliz517/CLIP-ReID).

CLIP-ReID is a CLIP ViT-B/16 image encoder modified for re-identification:
overlapping stride-12 patches + Side Information Embedding (SIE, training-only)
+ a BN neck on the CLS token → **768-dim** L2-normalized embedding.

Deployment: the portable artifact is an ONNX hosted on the gated HF repo
``PIA-SPACE-LAB/SSAVE``; a TensorRT engine is built from it on-demand (see
``engine.py``). A PyTorch fallback (``vision_encoder.py``) loads the original
``.pth`` directly when no engine is available.

Image-only: there is no text branch at inference.
"""

from __future__ import annotations

# CLIP ViT-B/16 width == ReID embedding dim (post BN-neck CLS).
EMBED_DIM = 768

# Person ReID default geometry (H, W). Vehicle (VeRi) is square 256×256 and is
# auto-detected from the checkpoint / engine binding at load time.
PERSON_INPUT_SIZE = (256, 128)
VEHICLE_INPUT_SIZE = (256, 256)
DEFAULT_INPUT_SIZE = PERSON_INPUT_SIZE

# Overlapping-patch stride (CLIP-ReID uses 12, vs 16 for vanilla CLIP).
DEFAULT_STRIDE = 12

# CLIP-standard normalization (CLIP-ReID inherits CLIP's preprocessing).
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_DEVICE = "cuda:0"
DEFAULT_BACKBONE = "ViT-B-16"

# Gated HF repo hosting the ONNX artifacts (same repo as YOLO26; one HF_TOKEN
# with access covers both).
HF_REPO = "PIA-SPACE-LAB/SSAVE"
