"""Config block → ReID embedder instance."""

from __future__ import annotations

from typing import Any


def build_reid(reid_cfg: dict[str, Any] | None) -> Any | None:
    """The ``reid:`` block. None when it is absent — the tracker then associates
    on geometry alone (and multi-camera mode cannot run)."""
    person_cfg = (reid_cfg or {}).get("person")
    if not person_cfg:
        return None
    backend = person_cfg.get("backend", "clip_reid")
    if backend != "clip_reid":
        raise ValueError(
            f"unsupported ReID backend '{backend}'; this package ships with 'clip_reid' only. "
            "Remove the `reid:` block to track on geometry alone."
        )
    from piaspace_clip_reid import CLIPReIDEmbedder

    return CLIPReIDEmbedder(person_cfg["clip_reid"])
