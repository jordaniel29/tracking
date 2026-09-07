"""Frame sync across cameras: round-robin interleaving.

Yields frame k of every open source, then k+1, … so a consumer that links
identities across cameras only ever sees the past — the causal order of one
live worker per camera, in a single process with one copy of each model.

A source that runs out of frames (or reaches ``max_frames``) is reported once
as :class:`Exhausted` and closed; the others keep going.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator

import numpy as np

from .sources import VideoSource


@dataclass(frozen=True)
class Frame:
    source: VideoSource
    frame_idx: int
    image: np.ndarray
    ts: datetime


@dataclass(frozen=True)
class Exhausted:
    """``source`` has no more frames; ``ts`` is the moment after its last one."""

    source: VideoSource
    ts: datetime


def _next(src: VideoSource, max_frames: int | None) -> Frame | Exhausted:
    item = None if (max_frames is not None and src.frames_read >= max_frames) else src.read()
    if item is None:
        src.close()
        return Exhausted(src, src.ts())
    idx, image = item
    return Frame(src, idx, image, src.ts(idx))


def round_robin(sources: Iterable[VideoSource], *, max_frames: int | None = None) -> Iterator[Frame | Exhausted]:
    open_sources = list(sources)
    while open_sources:
        for src in list(open_sources):
            event = _next(src, max_frames)
            if isinstance(event, Exhausted):
                open_sources.remove(src)
            yield event
