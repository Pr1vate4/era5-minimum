"""Integration smoke test for ERA5 demo Zarr dataset and layer cache."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import xarray
import zarr
import era5_minimum.data.channel_spec

from prometheus_client import REGISTRY

from era5_minimum.api.config import CacheSettings
from era5_minimum.api.weather import ZarrWeatherDataProvider
from era5_minimum.api.weather_data_service import WeatherDataService


def metric_sample(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else float(value)


@pytest.fixture(scope="module")
def zarr_provider() -> ZarrWeatherDataProvider:
    root = Path(os.getenv("ERA5_DATASET_ROOT", "data/era5_28ch_demo/era5_28ch_demo")).resolve()
    split = os.getenv("ERA5_DATASET_SPLIT", "validation")

    required_paths = [
        root / "manifest.json",
        root / f"{split}.zarr",
        root / "static.zarr",
    ]

    if not all(path.exists() for path in required_paths):
        pytest.skip("era5_28ch_demo dataset is not available on disk")

    try:
        return ZarrWeatherDataProvider(root=root, split=split)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot open era5_28ch_demo dataset: {exc}")


def test_era5_demo_smoke_cache_miss_then_hit(
    zarr_provider: ZarrWeatherDataProvider,
) -> None:
    variables = {item["logical_name"]: item for item in zarr_provider.variables()}
    if "t2m" not in variables:
        pytest.skip("t2m is not present in era5_28ch_demo")

    timestamps = zarr_provider.timestamps()
    if not timestamps:
        pytest.skip("era5_28ch_demo has no timestamps")

    timestamp = timestamps[0]
    level = variables["t2m"].get("default_level")
    variable_group = "surface" if level is None else "pressure"

    service = WeatherDataService(
        zarr_provider,
        CacheSettings(enabled=True, max_entries=8, max_bytes=None),
    )

    read_metric_labels = {
        "provider": "zarr",
        "variable_group": variable_group,
    }

    before_zarr_reads = metric_sample(
        "era5_zarr_read_duration_seconds_count",
        read_metric_labels,
    )
    before_entries = metric_sample("era5_layer_cache_entries")

    payload_1, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp=timestamp,
        level=level,
        mode="original",
        target_width=32,
        target_height=16,
        stride=1,
        response_format="json",
    )

    assert status_1 == "miss"
    assert payload_1["shape"] == [16, 32]

    after_first_zarr_reads = metric_sample(
        "era5_zarr_read_duration_seconds_count",
        read_metric_labels,
    )
    assert after_first_zarr_reads == before_zarr_reads + 1
    assert metric_sample("era5_layer_cache_entries") == before_entries + 1

    payload_2, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp=timestamp,
        level=level,
        mode="original",
        target_width=32,
        target_height=16,
        stride=1,
        response_format="json",
    )

    assert status_2 == "hit"

    # Cache hit must not trigger another provider/Zarr read.
    after_second_zarr_reads = metric_sample(
        "era5_zarr_read_duration_seconds_count",
        read_metric_labels,
    )
    assert after_second_zarr_reads == after_first_zarr_reads

    # Cache entry count must remain unchanged after hit.
    assert metric_sample("era5_layer_cache_entries") == before_entries + 1

    # The second response should be the exact cached object.
    assert payload_2 is payload_1

    # Explicit SSoT smoke assertions.
    assert payload_1["shape"] == payload_2["shape"]
    assert payload_1["dataset_id"] == payload_2["dataset_id"]
    assert payload_1["variable"] == payload_2["variable"]
    assert payload_1["timestamp"] == payload_2["timestamp"]
    assert payload_1["level"] == payload_2["level"]
    assert payload_1["mode"] == payload_2["mode"]
    assert payload_1["unit"] == payload_2["unit"]

    service.clear_cache()