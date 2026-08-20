"""CLIP ViT image encoder for CLIP-ReID (PyTorch fallback path).

Matches the architecture baked into the ONNX/TRT graph. CLIP ViT-B/16 (OpenAI
structure) modified for ReID:

  - Overlapping patches: conv1 kernel=16, stride=12 → e.g. 210 patches for (256,128)
  - Side Information Embedding (SIE): training-only; not applied at inference here
  - BN neck on the CLS output → 768-dim embedding (raw; the embedder L2-normalizes)
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.resblocks(x)
        return out


class CLIPVisionEncoder(nn.Module):
    """CLIP ViT image encoder with overlapping-patch embedding + BN neck for ReID."""

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 12,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        input_resolution: tuple[int, int] = (256, 128),
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.stride = stride

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=stride, bias=False)

        h_patches = (input_resolution[0] - patch_size) // stride + 1
        w_patches = (input_resolution[1] - patch_size) // stride + 1
        num_patches = h_patches * w_patches

        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = nn.LayerNorm(width)

        # BN neck (CLIP-ReID specific)
        self.bottleneck = nn.BatchNorm1d(width)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.conv1(x)  # (B, width, H', W')
        x = x.flatten(2).permute(0, 2, 1)  # (B, num_patches, width)

        cls = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+num_patches, width)
        x = x + self.positional_embedding

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq, B, width) for nn.MultiheadAttention
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, seq, width)

        x = self.ln_post(x[:, 0, :])  # CLS token → (B, width)
        x = self.bottleneck(x)  # (B, width)
        return x
