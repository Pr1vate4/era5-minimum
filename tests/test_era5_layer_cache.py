"""pytests for ERA5 in-memory LRU layer cache."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from era5_minimum.api.config import CacheSettings
from era5_minimum.api.weather_data_service import WeatherDataService


def metric_sample(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels or {})
    return 0.0 if value is None else float(value)


def base_labels(
    provider_name: str = "mock",
    mode: str = "original",
    response_format: str = "json",
    variable_group: str = "surface",
) -> dict[str, str]:
    return {
        "provider": provider_name,
        "mode": mode,
        "format": response_format,
        "variable_group": variable_group,
    }


def read_labels(
    provider_name: str = "mock",
    variable_group: str = "surface",
) -> dict[str, str]:
    return {
        "provider": provider_name,
        "variable_group": variable_group,
    }


class FakeProvider:
    is_mock = True

    def __init__(
        self,
        dataset_id: str = "mock",
        payload: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.payload = payload or {
            "values": [[1.0, 2.0], [3.0, 4.0]],
            "minimum": 1.0,
            "maximum": 4.0,
            "unit": "K",
        }
        self.error = error
        self.calls: list[tuple] = []

    def dataset_metadata(self) -> dict:
        return {"id": self.dataset_id}

    def layer(
        self,
        variable: str,
        timestamp: str,
        level: int | None,
        target_width: int,
        target_height: int,
    ) -> dict:
        self.calls.append(
            (
                variable,
                timestamp,
                level,
                target_width,
                target_height,
            )
        )

        if self.error is not None:
            raise self.error

        payload = dict(self.payload)
        payload.update(
            {
                "dataset_id": self.dataset_id,
                "variable": variable,
                "timestamp": timestamp,
                "level": level,
                "mode": "original",
                "shape": [target_height, target_width],
                "is_mock": True,
            }
        )
        return payload


def make_service(
    provider: FakeProvider | None = None,
    *,
    enabled: bool = True,
    max_entries: int | None = 32,
    max_bytes: int | None = 10_000_000,
) -> tuple[WeatherDataService, FakeProvider]:
    provider = provider or FakeProvider()
    settings = CacheSettings(
        enabled=enabled,
        max_entries=max_entries,
        max_bytes=max_bytes,
    )
    return WeatherDataService(provider, settings), provider


def test_first_request_miss_second_hit() -> None:
    service, provider = make_service()

    before_hits = metric_sample("era5_layer_cache_hits_total", base_labels())
    before_misses = metric_sample("era5_layer_cache_misses_total", base_labels())

    payload_1, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    payload_2, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert status_1 == "miss"
    assert status_2 == "hit"
    assert payload_1 == payload_2

    assert metric_sample("era5_layer_cache_hits_total", base_labels()) == before_hits + 1
    assert metric_sample("era5_layer_cache_misses_total", base_labels()) == before_misses + 1

    service.clear_cache()


def test_provider_called_once_for_identical_requests() -> None:
    service, provider = make_service()

    service.get_layer_with_status(variable="t2m", timestamp="2020-01-01T00:00:00Z")
    service.get_layer_with_status(variable="t2m", timestamp="2020-01-01T00:00:00Z")

    assert len(provider.calls) == 1

    service.clear_cache()


def test_change_timestamp_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-02T00:00:00Z",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_variable_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    _, status_2 = service.get_layer_with_status(
        variable="msl",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_level_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t",
        timestamp="2020-01-01T00:00:00Z",
        level=None,
    )
    _, status_2 = service.get_layer_with_status(
        variable="t",
        timestamp="2020-01-01T00:00:00Z",
        level=500,
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_target_width_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        target_width=8,
        target_height=8,
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        target_width=16,
        target_height=8,
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_target_height_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        target_width=8,
        target_height=8,
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        target_width=8,
        target_height=16,
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_mode_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        mode="original",
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        mode="reconstructed",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_stride_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        stride=1,
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        stride=2,
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_change_format_causes_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        response_format="json",
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
        response_format="binary",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_dataset_id_is_part_of_cache_key() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    # Simulate the same service seeing a different dataset identity.
    service._dataset_id = "other-dataset"

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_max_entries_eviction() -> None:
    service, provider = make_service(max_entries=1)

    before_evictions = metric_sample("era5_layer_cache_evictions_total")
    before_entries = metric_sample("era5_layer_cache_entries")

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_1 == "miss"
    assert metric_sample("era5_layer_cache_entries") == before_entries + 1

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-02T00:00:00Z",
    )
    assert status_2 == "miss"

    assert metric_sample("era5_layer_cache_evictions_total") == before_evictions + 1
    assert metric_sample("era5_layer_cache_entries") == before_entries + 1

    # The first timestamp should have been evicted.
    _, status_3 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert status_3 == "miss"
    assert len(provider.calls) == 3

    service.clear_cache()


def test_max_bytes_eviction() -> None:
    large_payload = {
        "values": [[0.0] * 32 for _ in range(32)],
        "minimum": 0.0,
        "maximum": 0.0,
        "unit": "K",
    }
    provider = FakeProvider(payload=large_payload)
    service, provider = make_service(provider, max_bytes=1)

    before_evictions = metric_sample("era5_layer_cache_evictions_total")
    before_entries = metric_sample("era5_layer_cache_entries")

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_1 == "miss"

    # Entry is stored and immediately evicted because it exceeds max_bytes.
    assert metric_sample("era5_layer_cache_evictions_total") >= before_evictions + 1
    assert metric_sample("era5_layer_cache_entries") == before_entries

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()


def test_disabled_cache_always_calls_provider() -> None:
    service, provider = make_service(enabled=False)

    before_entries = metric_sample("era5_layer_cache_entries")

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert status_1 == "miss"
    assert status_2 == "miss"
    assert len(provider.calls) == 2
    assert metric_sample("era5_layer_cache_entries") == before_entries

    service.clear_cache()


def test_provider_errors_are_not_cached() -> None:
    provider = FakeProvider(error=ValueError("boom"))
    service, provider = make_service(provider)

    with pytest.raises(ValueError):
        service.get_layer_with_status(
            variable="t2m",
            timestamp="2020-01-01T00:00:00Z",
        )

    with pytest.raises(ValueError):
        service.get_layer_with_status(
            variable="t2m",
            timestamp="2020-01-01T00:00:00Z",
        )

    assert len(provider.calls) == 2

    service.clear_cache()


def test_mock_and_zarr_providers_are_isolated() -> None:
    mock_provider = FakeProvider(dataset_id="mock")
    mock_provider.is_mock = True

    zarr_provider = FakeProvider(dataset_id="zarr")
    zarr_provider.is_mock = False

    settings = CacheSettings(enabled=True, max_entries=8, max_bytes=10_000_000)

    mock_service = WeatherDataService(mock_provider, settings)
    zarr_service = WeatherDataService(zarr_provider, settings)

    _, mock_status = mock_service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    _, zarr_status = zarr_service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )

    assert mock_status == "miss"
    assert zarr_status == "miss"
    assert len(mock_provider.calls) == 1
    assert len(zarr_provider.calls) == 1

    mock_service.clear_cache()
    zarr_service.clear_cache()


def test_gauges_and_counters_update() -> None:
    service, provider = make_service()

    before_entries = metric_sample("era5_layer_cache_entries")
    before_bytes = metric_sample("era5_layer_cache_bytes")
    before_hits = metric_sample("era5_layer_cache_hits_total", base_labels())
    before_misses = metric_sample("era5_layer_cache_misses_total", base_labels())

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_1 == "miss"

    after_first_entries = metric_sample("era5_layer_cache_entries")
    after_first_bytes = metric_sample("era5_layer_cache_bytes")

    assert after_first_entries == before_entries + 1
    assert after_first_bytes > before_bytes

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_2 == "hit"

    assert metric_sample("era5_layer_cache_hits_total", base_labels()) == before_hits + 1
    assert metric_sample("era5_layer_cache_misses_total", base_labels()) == before_misses + 1
    assert metric_sample("era5_layer_cache_entries") == after_first_entries

    service.clear_cache()

    assert metric_sample("era5_layer_cache_entries") == before_entries
    assert metric_sample("era5_layer_cache_bytes") == before_bytes


def test_cache_hit_does_not_increase_zarr_read_duration() -> None:
    service, provider = make_service()

    labels = read_labels(provider_name="mock", variable_group="surface")
    before = metric_sample("era5_zarr_read_duration_seconds_count", labels)

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_1 == "miss"

    after_miss = metric_sample("era5_zarr_read_duration_seconds_count", labels)
    assert after_miss == before + 1

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_2 == "hit"

    after_hit = metric_sample("era5_zarr_read_duration_seconds_count", labels)
    assert after_hit == after_miss

    service.clear_cache()


def test_clear_cache_makes_next_request_miss() -> None:
    service, provider = make_service()

    _, status_1 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_1 == "miss"

    service.clear_cache()

    _, status_2 = service.get_layer_with_status(
        variable="t2m",
        timestamp="2020-01-01T00:00:00Z",
    )
    assert status_2 == "miss"
    assert len(provider.calls) == 2

    service.clear_cache()