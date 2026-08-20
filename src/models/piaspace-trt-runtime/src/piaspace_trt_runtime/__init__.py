"""piaspace-trt-runtime — shared TRT runner + engine provisioning helpers.

Consumed by ``piaspace-clip-reid`` and ``piaspace-yolo26`` (and any future
model package). The per-family ``MODELS`` registry is supplied by the caller,
so this package stays model-agnostic.

Public API:
    from piaspace_trt_runtime import (
        TRTRunner,            # torch.cuda-based TensorRT engine runner
        ModelSpec,            # dataclass describing one provisionable artifact
        download_onnx,        # HF Hub → local .onnx
        build_engine,         # local .onnx → local .engine via TRT Python API
        ensure_engine,        # idempotent end-to-end (disk → onnx → engine)
        ensure_engine_by_filename,
    )
"""

from __future__ import annotations

from .provision import (
    ModelSpec,
    build_engine,
    download_onnx,
    ensure_engine,
    ensure_engine_by_filename,
)
from .runner import TRTRunner

__all__ = [
    "ModelSpec",
    "TRTRunner",
    "build_engine",
    "download_onnx",
    "ensure_engine",
    "ensure_engine_by_filename",
]
