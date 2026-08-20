"""PIASPACE CLIP-ReID image encoder — ViT-B/16, ONNX→TRT (PyTorch fallback).

A self-contained CLIP-ReID embedder: the framework-neutral public surface is
`embed(crops_bgr) -> (N, 768)` + `embed_dim`.

Deployment: the portable artifact is an ONNX on the gated HF repo
`PIA-SPACE-LAB/SSAVE`; a TensorRT engine is built from it on demand
(`engine.ensure_engine`). A PyTorch fallback runs the custom CLIP ViT directly
from a `.pth` when no engine is available.

Image-only — no text branch at inference. Heavy deps (`tensorrt`,
`huggingface_hub`) are lazy-imported, so importing this package and running the
stubbed tests needs neither.
"""

from .config import (
    CLIP_MEAN,
    CLIP_STD,
    DEFAULT_BACKBONE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_STRIDE,
    EMBED_DIM,
    HF_REPO,
    PERSON_INPUT_SIZE,
    VEHICLE_INPUT_SIZE,
)
from .embedder import CLIPReIDEmbedder
from .engine import MODELS, ModelSpec, ensure_engine, ensure_engine_by_filename
from .vision_encoder import CLIPVisionEncoder

__all__ = [
    # Public API
    "CLIPReIDEmbedder",
    # Provisioning
    "ensure_engine",
    "ensure_engine_by_filename",
    "ModelSpec",
    "MODELS",
    # Constants
    "EMBED_DIM",
    "PERSON_INPUT_SIZE",
    "VEHICLE_INPUT_SIZE",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_STRIDE",
    "DEFAULT_BACKBONE",
    "CLIP_MEAN",
    "CLIP_STD",
    "HF_REPO",
    # Lower-level building block (PyTorch path)
    "CLIPVisionEncoder",
]
__version__ = "0.2.0"
