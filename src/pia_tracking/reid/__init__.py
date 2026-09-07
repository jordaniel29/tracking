"""Appearance embeddings (ReID).

    base.py     ``ReIDBackend`` — the embedder interface the tracker and fusion consume
    factory.py  ``build_reid(cfg)`` — config block → embedder (CLIP-ReID via piaspace_clip_reid)
"""

from .base import ReIDBackend
from .factory import build_reid

__all__ = ["ReIDBackend", "build_reid"]
