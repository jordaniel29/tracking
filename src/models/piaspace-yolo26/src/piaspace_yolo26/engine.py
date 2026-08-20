"""YOLO26 engine registry + thin provisioning wrappers.

The provisioning pipeline (HF download → TRT build → cache) lives in
``piaspace-trt-runtime``; this module just owns the per-family ``MODELS``
registry and exposes ``ensure_engine`` / ``ensure_engine_by_filename`` with the
registry pre-bound for ergonomic callers.

Mirrors the pattern used by ``piaspace-clip-reid.engine``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from piaspace_trt_runtime import ModelSpec
from piaspace_trt_runtime import ensure_engine as _ensure_engine
from piaspace_trt_runtime import ensure_engine_by_filename as _ensure_engine_by_filename

from .config import HF_REPO

# Registry — keep in sync with the ONNX artifacts published in PIA-SPACE-LAB/SSAVE.
# All YOLO26 variants share the same graph I/O (input tensor "images", NCHW) and
# only differ in weights, so the optimization profile is identical across sizes.
MODELS: dict[str, ModelSpec] = {
    "yolo26l": ModelSpec(
        onnx_filename="yolo26l.onnx",
        engine_filename="yolo26l.fp16.engine",
        input_name="images",
        min_shape=(1, 3, 320, 320),
        opt_shape=(4, 3, 640, 640),
        max_shape=(16, 3, 640, 640),
        hub_repo=HF_REPO,
    ),
    # Person-only fine-tune of yolo26l. Same graph I/O as stock yolo26l.
    # hub_path differs from onnx_filename: the hub hosts this artifact under a
    # versioned name, while local files keep the yolo26l_ft.* names.
    "yolo26l_ft": ModelSpec(
        onnx_filename="yolo26l_ft.onnx",
        engine_filename="yolo26l_ft.fp16.engine",
        input_name="images",
        min_shape=(1, 3, 320, 320),
        opt_shape=(4, 3, 640, 640),
        max_shape=(16, 3, 640, 640),
        hub_repo=HF_REPO,
        hub_path="yolo26l_v5.1.onnx",
    ),
    # Person-only fine-tune of yolo26l, v6.3. Same graph I/O as stock yolo26l.
    "yolo26l_v6.3": ModelSpec(
        onnx_filename="yolo26l_v6.3.onnx",
        engine_filename="yolo26l_v6.3.fp16.engine",
        input_name="images",
        min_shape=(1, 3, 320, 320),
        opt_shape=(4, 3, 640, 640),
        max_shape=(16, 3, 640, 640),
        hub_repo=HF_REPO,
    ),
    "yolo26n": ModelSpec(
        onnx_filename="yolo26n.onnx",
        engine_filename="yolo26n.fp16.engine",
        input_name="images",
        min_shape=(1, 3, 320, 320),
        opt_shape=(4, 3, 640, 640),
        max_shape=(16, 3, 640, 640),
        hub_repo=HF_REPO,
    ),
}


def ensure_engine(
    model_key: str,
    weights_dir: str | Path = "./weights",
    precision: str = "fp16",
    token: str | None = None,
) -> Path:
    """Return a ready-to-load TRT engine for ``model_key`` (YOLO26 family)."""
    return _ensure_engine(model_key, MODELS, weights_dir, precision=precision, token=token)


def ensure_engine_by_filename(
    engine_filename: str, weights_dir: str | Path = "./weights", **kwargs: Any
) -> Path:
    """Resolve a YOLO26 ``ModelSpec`` by ``engine_filename`` and provision it."""
    return _ensure_engine_by_filename(engine_filename, MODELS, weights_dir=weights_dir, **kwargs)
