"""Shared HF→ONNX→TensorRT engine provisioning helpers.

Resolution order for ``weights/<name>.fp16.engine``:

  1. Engine file on disk → use it.
  2. ONNX file ``weights/<name>.onnx`` on disk → build engine, cache to disk.
  3. Neither → download ONNX from the spec's ``hub_repo`` → build the engine.

Auth: ``HF_TOKEN`` env var preferred; falls back to
``~/.cache/huggingface/token``. Engines are built with the TensorRT Python API
(no ``trtexec`` binary needed inside the container).

``tensorrt`` / ``huggingface_hub`` are lazy-imported inside the functions that
need them, so importing this module is cheap and dependency-free.

The ``MODELS`` registry is supplied by the caller (each per-family model
package keeps its own), so the same code path serves CLIP-ReID, YOLO26, and
any future model package.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to materialize one engine on demand.

    - ``onnx_filename``  : local filename under ``weights/`` (final landing spot)
    - ``engine_filename``: local filename for the cached TRT engine
    - ``input_name``     : ONNX input tensor name (must match the graph)
    - ``min/opt/max_shape``: trtexec-style profile. Element order follows the
      ONNX graph (e.g. ``(B, C, H, W)``).
    - ``hub_repo``       : HF repo id, e.g. ``"PIA-SPACE-LAB/SSAVE"``.
    - ``hub_path``       : repo-relative path of the ONNX. Defaults to
      ``onnx_filename`` (file at the repo root).
    - ``workspace_mib``  : TRT builder workspace pool size.
    """

    onnx_filename: str
    engine_filename: str
    input_name: str
    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    hub_repo: str
    hub_path: str | None = None
    workspace_mib: int = 4096


def _resolve_hf_token() -> str | None:
    """HF token: env first, then the legacy cache file."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    try:
        if cache_path.is_file():
            return cache_path.read_text().strip() or None
    except OSError:
        # The legacy cache may live under another user's $HOME (split-ownership
        # NAS layout) and be unreadable by the account the server runs as.
        # Treat as "no token": HF_TOKEN (checked above) is the supported
        # override, and download_onnx() raises a clear error if still missing.
        pass
    return None


def download_onnx(spec: ModelSpec, weights_dir: Path, token: str | None = None) -> Path:
    """Download ``spec``'s ONNX from the Hub into ``weights_dir``.

    The local filename is always ``spec.onnx_filename`` regardless of where the
    file lives in the remote repo (``spec.hub_path``) — callers don't need to
    track the remote-vs-local naming difference.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / spec.onnx_filename
    if dest.is_file():
        return dest

    token = token or _resolve_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set and no token at ~/.cache/huggingface/token. Cannot "
            f"download {spec.hub_repo}/{spec.hub_path or spec.onnx_filename} (gated repo). "
            "Put HF_TOKEN in .env or run `huggingface-cli login`."
        )

    hub_path = spec.hub_path or spec.onnx_filename
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for auto-provisioning. "
            "`pip install huggingface_hub` (or install your encoder package's [hf] extra)."
        ) from e

    logger.info("downloading hf://%s/%s -> %s", spec.hub_repo, hub_path, dest)
    downloaded = Path(
        hf_hub_download(
            repo_id=spec.hub_repo,
            filename=hub_path,
            local_dir=str(weights_dir),
            token=token,
        )
    )
    if downloaded != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded.rename(dest)
    return dest


def build_engine(
    onnx_path: Path, engine_path: Path, spec: ModelSpec, precision: str = "fp16"
) -> Path:
    """Build a TRT engine from ``onnx_path`` using the TensorRT Python API."""
    import tensorrt as trt

    logger.info("building TRT engine: %s -> %s", onnx_path.name, engine_path.name)
    t0 = time.time()
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    # Explicit-batch network flag. TensorRT >=10 removed
    # NetworkDefinitionCreationFlag.EXPLICIT_BATCH (explicit batch is the only
    # mode now), so create_network(0). On TRT <=9 the flag is still required.
    # Guard on attribute presence to support both (pyproject pins tensorrt>=10.7).
    _net_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        _net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(_net_flags)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, spec.workspace_mib * (1 << 20))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision != "fp32":
        raise ValueError(f"unsupported precision '{precision}' (fp16|fp32)")

    profile = builder.create_optimization_profile()
    profile.set_shape(spec.input_name, spec.min_shape, spec.opt_shape, spec.max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TRT failed to build engine for {onnx_path.name}.")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path.with_suffix(engine_path.suffix + ".part")
    with open(tmp, "wb") as f:
        f.write(serialized)
    tmp.rename(engine_path)
    logger.info(
        "built engine %s (%.1f MiB) in %.1fs",
        engine_path.name,
        engine_path.stat().st_size / (1024 * 1024),
        time.time() - t0,
    )
    return engine_path


def ensure_engine(
    model_key: str,
    registry: Mapping[str, ModelSpec],
    weights_dir: str | Path = "./weights",
    precision: str = "fp16",
    token: str | None = None,
) -> Path:
    """Return a ready-to-load TRT engine for ``model_key``, building/downloading
    on demand. ``registry`` is the per-package ``MODELS`` dict."""
    if model_key not in registry:
        raise KeyError(f"unknown model '{model_key}'. Known: {list(registry)}")
    spec = registry[model_key]
    weights_dir = Path(weights_dir)

    engine_path = weights_dir / spec.engine_filename
    if engine_path.is_file():
        return engine_path

    onnx_path = weights_dir / spec.onnx_filename
    if not onnx_path.is_file():
        onnx_path = download_onnx(spec, weights_dir, token=token)

    return build_engine(onnx_path, engine_path, spec, precision=precision)


def ensure_engine_by_filename(
    engine_filename: str,
    registry: Mapping[str, ModelSpec],
    weights_dir: str | Path = "./weights",
    **kwargs: Any,
) -> Path:
    """Resolve a ModelSpec by its ``engine_filename`` and call ``ensure_engine``."""
    for key, spec in registry.items():
        if spec.engine_filename == engine_filename:
            return ensure_engine(key, registry, weights_dir=weights_dir, **kwargs)
    raise KeyError(
        f"no ModelSpec for engine '{engine_filename}'. "
        f"Known: {[s.engine_filename for s in registry.values()]}"
    )
