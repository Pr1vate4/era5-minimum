from __future__ import annotations

import json

import numpy as np

from scripts.prepare_globe_assets import (
    generate_demo_field,
    normalize_loaded_array,
    normalize_timestamp,
    update_manifest,
)


def test_demo_field_is_deterministic_and_grid_shaped() -> None:
    first = generate_demo_field("t2m", "0p5", "original", None)
    second = generate_demo_field("t2m", "0p5", "original", None)

    assert first.values.shape == (361, 720)
    assert first.values.dtype == np.float32
    np.testing.assert_array_equal(first.values, second.values)


def test_demo_absolute_error_is_non_negative() -> None:
    field = generate_demo_field("t2m", "0p5", "absolute-error", None)

    assert np.isfinite(field.values).all()
    assert float(field.values.min()) >= 0
    assert float(field.values.max()) > 0


def test_normalize_loaded_array_rejects_non_spatial_input() -> None:
    with np.testing.assert_raises_regex(ValueError, "two-dimensional"):
        normalize_loaded_array(np.zeros((2, 3, 4), dtype=np.float32), "K")


def test_manifest_update_replaces_same_frame_id(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    frame = {"id": "frame-1", "timestamp": "2021-01-01T00:00:00Z", "mode": "original", "channel": "t2m"}
    update_manifest(manifest_path, {**frame, "mean": 1})
    update_manifest(manifest_path, {**frame, "mean": 2})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["frames"]) == 1
    assert manifest["frames"][0]["mean"] == 2


def test_timestamp_is_normalized_to_utc() -> None:
    assert normalize_timestamp("2021-06-15T15:00:00+03:00") == "2021-06-15T12:00:00Z"
