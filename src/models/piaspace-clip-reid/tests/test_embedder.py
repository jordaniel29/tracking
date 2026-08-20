"""Unit tests for CLIPReIDEmbedder — PyTorch path (real .pth) + stubbed TRT path."""

from __future__ import annotations

import numpy as np
import pytest
from piaspace_clip_reid import CLIPReIDEmbedder
from piaspace_clip_reid.config import EMBED_DIM

# ---- PyTorch fallback path (loads the random-weight .pth) ----


def test_pytorch_path_embed_shape(person_pth, sample_crops):
    emb = CLIPReIDEmbedder({"device": "cpu", "weights_path": str(person_pth)})
    out = emb.embed(sample_crops)
    assert out.shape == (4, EMBED_DIM)
    assert out.dtype == np.float32


def test_pytorch_path_l2_normalized(person_pth, sample_crops):
    emb = CLIPReIDEmbedder({"device": "cpu", "weights_path": str(person_pth)})
    out = emb.embed(sample_crops)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_embed_dim_is_768(person_pth):
    emb = CLIPReIDEmbedder({"device": "cpu", "weights_path": str(person_pth)})
    assert emb.embed_dim == 768


def test_embed_empty(person_pth):
    emb = CLIPReIDEmbedder({"device": "cpu", "weights_path": str(person_pth)})
    out = emb.embed([])
    assert out.shape == (0, EMBED_DIM)
    assert out.dtype == np.float32


def test_missing_weights_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="CLIP-ReID weights required"):
        CLIPReIDEmbedder({"device": "cpu", "weights_path": str(tmp_path / "nope.pth")})


# ---- TRT engine path (stubbed runner; engine file just needs to exist) ----


def test_trt_path_used_when_engine_present(stub_trt, tmp_path, sample_crops):
    engine = tmp_path / "clipreid_person.fp16.engine"
    engine.write_bytes(b"\x00" * 16)  # _ensure_engine_on_disk sees it as present
    emb = CLIPReIDEmbedder({"device": "cpu", "engine_path": str(engine)})
    assert emb._trt is not None and emb._model is None
    out = emb.embed(sample_crops)
    assert out.shape == (4, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_trt_binding_overrides_input_size(stub_trt, tmp_path):
    engine = tmp_path / "clipreid_person.fp16.engine"
    engine.write_bytes(b"\x00" * 16)
    # Pass a bogus input_size; the stub binding (256x128) should override it.
    emb = CLIPReIDEmbedder({"device": "cpu", "engine_path": str(engine), "input_size": [999, 999]})
    assert emb.input_size == (256, 128)
