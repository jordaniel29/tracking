"""Fixtures: a real (random-weight) .pth for the PyTorch path + a stub TRT runner,
so tests run without GPU / tensorrt / real weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from piaspace_clip_reid.config import EMBED_DIM, PERSON_INPUT_SIZE
from piaspace_clip_reid.vision_encoder import CLIPVisionEncoder


@pytest.fixture
def person_pth(tmp_path: Path) -> Path:
    """A checkpoint shaped like CLIP-ReID's: image_encoder.* + bottleneck.*
    (random weights — enough to exercise the load + forward path)."""
    enc = CLIPVisionEncoder(input_resolution=PERSON_INPUT_SIZE)
    state = {}
    for k, v in enc.state_dict().items():
        state[k if k.startswith("bottleneck.") else f"image_encoder.{k}"] = v
    path = tmp_path / "MSMT17_clipreid_12x12sie_ViT-B-16_60.pth"
    torch.save(state, path)
    return path


class StubTRTRunner:
    """Drop-in for TRTRunner: deterministic content-dependent (N, 768) output."""

    def __init__(self, *args, embed_dim: int = EMBED_DIM, size=PERSON_INPUT_SIZE, **kwargs):
        self.input_names = ["images"]
        self.output_names = ["features"]
        self.num_inputs = 1
        self.num_outputs = 1
        self._embed_dim = embed_dim
        self._shape = (-1, 3, size[0], size[1])
        self._proj: dict[int, torch.Tensor] = {}

    def get_binding_shape(self, name: str):
        return self._shape

    def infer_single(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        flat = x.reshape(n, -1).float()
        d_in = flat.shape[1]
        if d_in not in self._proj:
            gen = torch.Generator().manual_seed(42)
            self._proj[d_in] = torch.randn(d_in, self._embed_dim, generator=gen)
        return flat @ self._proj[d_in]


@pytest.fixture
def stub_trt(monkeypatch):
    """Patch the lazily-imported TRTRunner so the engine path needs no GPU."""
    # Patch both the source location and the re-exported binding so the
    # lazy ``from piaspace_trt_runtime import TRTRunner`` in embedder.py
    # picks up the stub regardless of which name it resolves through.
    monkeypatch.setattr("piaspace_trt_runtime.runner.TRTRunner", StubTRTRunner)
    monkeypatch.setattr("piaspace_trt_runtime.TRTRunner", StubTRTRunner)
    return StubTRTRunner


@pytest.fixture
def sample_crops() -> list[np.ndarray]:
    """4 random BGR uint8 crops of varying sizes (resized internally)."""
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (120 + i * 30, 60 + i * 10, 3), dtype=np.uint8) for i in range(4)]
