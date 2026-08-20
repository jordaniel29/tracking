"""Unit tests for the provisioning registry (no network)."""

from __future__ import annotations

import pytest
from piaspace_clip_reid.engine import (
    MODELS,
    ensure_engine,
    ensure_engine_by_filename,
)


def test_registry_has_person_and_vehicle():
    assert set(MODELS) == {"clipreid_person", "clipreid_vehicle"}


def test_person_spec_geometry():
    spec = MODELS["clipreid_person"]
    assert spec.input_name == "images"
    assert spec.engine_filename == "clipreid_person.fp16.engine"
    assert spec.opt_shape == (8, 3, 256, 128)
    assert spec.hub_repo == "PIA-SPACE-LAB/SSAVE"


def test_vehicle_spec_is_square():
    spec = MODELS["clipreid_vehicle"]
    assert spec.opt_shape == (8, 3, 256, 256)


def test_ensure_engine_unknown_key():
    with pytest.raises(KeyError, match="unknown model"):
        ensure_engine("not_a_model")


def test_ensure_engine_by_filename_unknown():
    with pytest.raises(KeyError, match="no ModelSpec"):
        ensure_engine_by_filename("mystery.engine")
