"""Unit tests for the CLIP ViT vision encoder (PyTorch architecture)."""

from __future__ import annotations

import torch
from piaspace_clip_reid.config import EMBED_DIM, PERSON_INPUT_SIZE
from piaspace_clip_reid.vision_encoder import CLIPVisionEncoder


def test_forward_shape_person():
    enc = CLIPVisionEncoder(input_resolution=PERSON_INPUT_SIZE).eval()
    x = torch.randn(3, 3, *PERSON_INPUT_SIZE)
    out = enc(x)
    assert out.shape == (3, EMBED_DIM)  # 768-dim


def test_forward_shape_square_vehicle():
    enc = CLIPVisionEncoder(input_resolution=(256, 256)).eval()
    out = enc(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, EMBED_DIM)


def test_overlapping_patch_count():
    """Stride-12 overlapping patches: (256-16)/12+1=21 by (128-16)/12+1=10 → 210."""
    enc = CLIPVisionEncoder(input_resolution=(256, 128), stride=12)
    # positional_embedding is (num_patches + 1, width)
    assert enc.positional_embedding.shape[0] == 210 + 1


def test_content_dependent():
    enc = CLIPVisionEncoder(input_resolution=PERSON_INPUT_SIZE).eval()
    with torch.no_grad():
        a = enc(torch.randn(1, 3, *PERSON_INPUT_SIZE))
        b = enc(torch.randn(1, 3, *PERSON_INPUT_SIZE))
    assert not torch.allclose(a, b)
