"""CLIP-ReID engine registry + thin provisioning wrappers.

The provisioning pipeline (HF download → TRT build → cache) lives in
``piaspace-trt-runtime``; this module just owns the per-family ``MODELS``
registry and exposes ``ensure_engine`` / ``ensure_engine_by_filename`` with the
registry pre-bound for ergonomic callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from piaspace_trt_runtime import ModelSpec
from piaspace_trt_runtime import ensure_engine as _ensure_engine
from piaspace_trt_runtime import ensure_engine_by_filename as _ensure_engine_by_filename

from .config import HF_REPO

# Registry — keep in sync with the ONNX artifacts published in PIA-SPACE-LAB/SSAVE.
MODELS: dict[str, ModelSpec] = {
    "clipreid_person": ModelSpec(
        onnx_filename="clipreid_person.onnx",
        engine_filename="clipreid_person.fp16.engine",
        input_name="images",
        min_shape=(1, 3, 256, 128),
        opt_shape=(8, 3, 256, 128),
        max_shape=(32, 3, 256, 128),
        hub_repo=HF_REPO,
    ),
    "clipreid_vehicle": ModelSpec(
        onnx_filename="clipreid_vehicle.onnx",
        engine_filename="clipreid_vehicle.fp16.engine",
        input_name="images",
        # VeRi CLIP-ReID trained on 256×256 square crops.
        min_shape=(1, 3, 256, 256),
        opt_shape=(8, 3, 256, 256),
        max_shape=(32, 3, 256, 256),
        hub_repo=HF_REPO,
    ),
}


def ensure_engine(
    model_key: str,
    weights_dir: str | Path = "./weights",
    precision: str = "fp16",
    token: str | None = None,
) -> Path:
    """Return a ready-to-load TRT engine for ``model_key`` (CLIP-ReID family)."""
    return _ensure_engine(model_key, MODELS, weights_dir, precision=precision, token=token)


def ensure_engine_by_filename(
    engine_filename: str, weights_dir: str | Path = "./weights", **kwargs: Any
) -> Path:
    """Resolve a CLIP-ReID ``ModelSpec`` by ``engine_filename`` and provision it."""
    return _ensure_engine_by_filename(engine_filename, MODELS, weights_dir=weights_dir, **kwargs)
