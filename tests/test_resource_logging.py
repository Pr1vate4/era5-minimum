from __future__ import annotations

import json
from pathlib import Path

from era5_minimum.codec.resources import build_resource_usage_record, write_resource_usage


def test_resource_usage_record_marks_compliance_and_serializes(tmp_path: Path) -> None:
    record = build_resource_usage_record(
        run_id="run-1",
        operation="train_codec",
        grid_resolution="0.5deg",
        dataset_size=12,
        started_at="2026-07-24T00:00:00Z",
        finished_at="2026-07-24T00:00:10Z",
        wall_clock_seconds=10.0,
        gpu_hours=None,
        gpu_name=None,
        visible_gpu_count=0,
        peak_vram_bytes=None,
        peak_ram_bytes=1234,
        trainable_parameter_count=1_000,
        total_parameter_count=2_000,
        optimizer_steps=5,
        examples_seen=12,
        unique_train_timestamps=3,
        patches_seen=12,
        encode_seconds=1.0,
        decode_seconds=2.0,
        bitstream_bytes=500,
        parameter_limit=2_000,
        max_vram_bytes=None,
        max_gpu_hours=None,
    )

    assert record["resource_compliance"] is True

    path = tmp_path / "resource_usage.json"
    write_resource_usage(path, record)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["peak_vram_bytes"] is None
    assert payload["optimizer_steps"] == 5


def test_resource_usage_record_flags_parameter_limit() -> None:
    record = build_resource_usage_record(
        run_id="run-2",
        operation="train_codec",
        grid_resolution="0.5deg",
        dataset_size=12,
        started_at="2026-07-24T00:00:00Z",
        finished_at="2026-07-24T00:00:10Z",
        wall_clock_seconds=10.0,
        gpu_hours=None,
        gpu_name=None,
        visible_gpu_count=0,
        peak_vram_bytes=None,
        peak_ram_bytes=1234,
        trainable_parameter_count=3_000,
        total_parameter_count=4_000,
        optimizer_steps=5,
        examples_seen=12,
        unique_train_timestamps=3,
        patches_seen=12,
        encode_seconds=1.0,
        decode_seconds=2.0,
        bitstream_bytes=500,
        parameter_limit=2_000,
        max_vram_bytes=None,
        max_gpu_hours=None,
    )

    assert record["resource_compliance"] is False
