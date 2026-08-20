"""CLIPReIDEmbedder — CLIP-ReID image embedder (TRT-preferred, PyTorch fallback).

Self-contained, with no dependency on the rest of the repository: the public
surface is just ``embed(crops_bgr) -> (N, D)`` + ``embed_dim``.

Extracts ``image_encoder.*`` and ``bottleneck.*`` from the ``.pth`` for the
PyTorch path; or runs a prebuilt/auto-provisioned TRT engine. Image-only — the
text encoder / classifier / prompt learner are not needed at inference.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

if TYPE_CHECKING:
    from piaspace_trt_runtime import TRTRunner

from .config import (
    CLIP_MEAN,
    CLIP_STD,
    DEFAULT_BACKBONE,
    DEFAULT_DEVICE,
    DEFAULT_INPUT_SIZE,
    DEFAULT_STRIDE,
    EMBED_DIM,
)
from .vision_encoder import CLIPVisionEncoder

logger = logging.getLogger(__name__)


class CLIPReIDEmbedder:
    """CLIP-ReID embedder. Construct with a config dict.

    Keys: ``device``, ``backbone``, ``input_size`` [H, W], ``stride``,
    ``weights_path`` (PyTorch ``.pth``), ``engine_path`` (TRT engine — preferred;
    auto-provisioned if missing), ``batch_size``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.device = config.get("device", DEFAULT_DEVICE)
        self.backbone_key = config.get("backbone", DEFAULT_BACKBONE)
        h, w = config.get("input_size", DEFAULT_INPUT_SIZE)
        self.input_size = (int(h), int(w))
        self.weights_path: str | None = config.get("weights_path")
        self.stride = int(config.get("stride", DEFAULT_STRIDE))
        self.batch_size = int(config.get("batch_size", 32))
        self._dim = EMBED_DIM  # ViT-B/16: width=768

        engine_path = config.get("engine_path")
        self._trt: TRTRunner | None = None
        self._model: CLIPVisionEncoder | None = None
        if engine_path:
            engine_path = self._ensure_engine_on_disk(engine_path)
        if engine_path and Path(engine_path).is_file():
            self._trt = self._build_trt(engine_path)
        else:
            if config.get("engine_path"):
                logger.warning(
                    "CLIP-ReID engine '%s' unavailable; falling back to PyTorch %s",
                    config.get("engine_path"),
                    self.weights_path,
                )
            self._model = self._build()

    # ------------------------------------------------------------------
    # Construction paths
    # ------------------------------------------------------------------
    def _ensure_engine_on_disk(self, engine_path: str) -> str | None:
        """If the engine is missing, try auto-provision (download ONNX + build).
        Returns the path on success, or None so the caller falls back to PyTorch."""
        p = Path(engine_path)
        if p.is_file():
            return engine_path
        try:
            from .engine import ensure_engine_by_filename

            built = ensure_engine_by_filename(
                p.name, weights_dir=p.parent if str(p.parent) else "./weights"
            )
            return str(built)
        except Exception as e:
            logger.warning("CLIP-ReID auto-provision failed for '%s': %s", engine_path, e)
            return None

    def _build(self) -> CLIPVisionEncoder:
        if not self.weights_path or not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"CLIP-ReID weights required at '{self.weights_path}' (no TRT engine "
                "available). Provide an engine_path (auto-provisions from "
                "PIA-SPACE-LAB/SSAVE) or a .pth from https://github.com/Syliz517/CLIP-ReID."
            )

        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)

        # Keep only image_encoder.* (strip prefix) + bottleneck.* (BN neck).
        model_state = {}
        for k, v in state.items():
            if k.startswith("image_encoder."):
                model_state[k[len("image_encoder.") :]] = v
            elif k.startswith("bottleneck.") and not k.startswith("bottleneck_proj"):
                model_state[k] = v

        # Auto-detect input resolution from positional_embedding (handles VeRi's
        # square 256×256 even when the caller passes 256×128).
        pos_embed = model_state.get("positional_embedding")
        if pos_embed is not None:
            num_patches = pos_embed.shape[0] - 1
            h, w = self.input_size
            h_patches = (h - 16) // self.stride + 1
            w_patches = (w - 16) // self.stride + 1
            if h_patches * w_patches != num_patches:
                side = int(math.sqrt(num_patches))
                if side * side == num_patches:
                    res = (side - 1) * self.stride + 16
                    self.input_size = (res, res)
                    logger.info(
                        "CLIP-ReID: auto-detected input_size=(%d,%d) from pos_embed "
                        "(%d patches, stride=%d)",
                        res,
                        res,
                        num_patches,
                        self.stride,
                    )

        model = CLIPVisionEncoder(
            patch_size=16,
            stride=self.stride,
            width=768,
            layers=12,
            heads=12,
            input_resolution=self.input_size,
        )
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        logger.info(
            "CLIP-ReID loaded from %s (missing=%d: %s, unexpected=%d)",
            self.weights_path,
            len(missing),
            missing[:5],
            len(unexpected),
        )
        model.eval().to(self.device)
        return model

    def _build_trt(self, engine_path: str) -> TRTRunner:
        """Load the TRT engine; sync input_size from the engine's binding."""
        from piaspace_trt_runtime import TRTRunner

        runner = TRTRunner(engine_path, device=self.device)
        if runner.num_inputs != 1 or runner.num_outputs != 1:
            raise RuntimeError(
                f"Expected 1-in/1-out engine; got {runner.num_inputs}-in/"
                f"{runner.num_outputs}-out at {engine_path}"
            )
        in_shape = runner.get_binding_shape(runner.input_names[0])  # (-1, 3, H, W)
        if len(in_shape) == 4 and in_shape[2] > 0 and in_shape[3] > 0:
            engine_size = (int(in_shape[2]), int(in_shape[3]))
            if engine_size != self.input_size:
                logger.info(
                    "CLIP-ReID TRT binding is %s; overriding input_size %s -> %s",
                    in_shape,
                    self.input_size,
                    engine_size,
                )
                self.input_size = engine_size
        logger.info("CLIP-ReID using TRT engine %s (size=%s)", engine_path, self.input_size)
        return runner

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def embed_dim(self) -> int:
        return self._dim

    def _preprocess(self, crops_bgr: list[npt.NDArray[np.uint8]]) -> torch.Tensor:
        """Resize + BGR→RGB + CLIP-standard normalize → (N, 3, H, W) float32."""
        h, w = self.input_size
        mean = torch.tensor(CLIP_MEAN)
        std = torch.tensor(CLIP_STD)
        tensors = []
        for crop in crops_bgr:
            img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(img.transpose(2, 0, 1))
            t = (t - mean.unsqueeze(-1).unsqueeze(-1)) / std.unsqueeze(-1).unsqueeze(-1)
            tensors.append(t)
        return torch.stack(tensors).to(self.device)

    def embed(self, crops_bgr: list[npt.NDArray[np.uint8]]) -> npt.NDArray[np.float32]:
        """Embed a batch of BGR uint8 crops → (N, 768) float32 L2-normalized."""
        if not crops_bgr:
            return np.empty((0, self._dim), dtype=np.float32)

        all_feats = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch = self._preprocess(crops_bgr[i : i + self.batch_size])
            if self._trt is not None:
                feats = self._trt.infer_single(batch)  # (B, 768), raw
            else:
                assert self._model is not None  # one of _trt/_model is always set
                with torch.no_grad():
                    feats = self._model(batch)  # (B, 768), raw
            feats = F.normalize(feats.float(), p=2, dim=1)
            all_feats.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_feats, axis=0)
