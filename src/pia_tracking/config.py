"""Loading a config yaml (config/tracking_general.yaml by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path, device: str | None = None) -> dict[str, Any]:
    """Parse the YAML config; ``device`` overrides every model's device."""
    config = yaml.safe_load(Path(path).read_text())
    if "detector" not in config or "tracker" not in config:
        raise ValueError(f"missing 'detector' or 'tracker' block in {path}")

    if device:
        config["detector"]["device"] = device
        reid_cfg = config.get("reid", {}).get("person", {})
        backend = reid_cfg.get("backend")
        if backend and backend in reid_cfg:
            reid_cfg[backend]["device"] = device

    return config
