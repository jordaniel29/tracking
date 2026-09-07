"""End-to-end runs over video files — what ``infer.py`` dispatches to.

    common.py  ``RunOptions`` + the per-clip output sinks both modes share
    single.py  ``run_single`` — each clip on its own, local ids
    multi.py   ``run_multi``  — all cameras at once, one Global ID service
    render.py  ``run_render`` — draw an existing run's predictions onto the videos (no models)
"""

from .common import RunOptions
from .multi import run_multi
from .render import run_render
from .single import run_single

__all__ = ["RunOptions", "run_multi", "run_render", "run_single"]
