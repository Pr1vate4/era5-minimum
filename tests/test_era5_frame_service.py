"""Unit tests for exact timestamp-to-model ERA5 frame preparation."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from era5_minimum.api.codec_service import _read_npz
from era5_minimum.api.era5_frame_service import (
    Era5FrameService,
    FrameValidationError,
    InvalidCoordinateOrderError,
    MissingChannelError,
    NonFiniteValuesError,
    TimestampNotAvailable,
    prepare_native_model_tensor,
    serialize_frame_npz,
    validate_prepared_frame,
)
from era5_minimum.data.channel_spec import (
    CHANNEL_NAMES,
    CHANNEL_SPEC,
    PRESSURE_LEVELS,
)
from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.model_input import CANONICAL_FRAME_SHAPE


TIMESTAMP = datetime(2020, 1, 1, 18, tzinfo=UTC)


def _manifest() -> dict:
    return {
        "schema_version": "weatherbench2-28ch-0p5-v1",
        "source_uri": "fixture://weatherbench2",
        "channel_order": list(CHANNEL_NAMES),
        "channels": [
            {
                "name": channel.name,
                "source_name": channel.source_name,
                "units": channel.units,
                "level_hpa": channel.level,
            }
            for channel in CHANNEL_SPEC
        ],
    }


def _prepared_service() -> Era5FrameService:
    target = get_target_grid_05()
    shuffled = tuple(reversed(CHANNEL_NAMES))
    values = np.empty(
        (1, len(CHANNEL_NAMES), *CANONICAL_FRAME_SHAPE[-2:]),
        dtype=np.float64,
    )
    for position, name in enumerate(shuffled):
        values[:, position] = float(CHANNEL_NAMES.index(name))
    sst_position = shuffled.index("sst")
    values[0, sst_position, 0, 0] = np.nan

    ocean_mask = np.ones(CANONICAL_FRAME_SHAPE[-2:], dtype=np.uint8)
    ocean_mask[0, 0] = 0
    dataset = xr.Dataset(
        {
            "data": (
                ("time", "channel", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "time": [np.datetime64("2020-01-01T18:00:00")],
            "channel": list(shuffled),
            "latitude": target.latitude.values,
            "longitude": target.longitude.values,
        },
    )
    static = xr.Dataset(
        {
            "ocean_mask": (
                ("latitude", "longitude"),
                ocean_mask,
            )
        },
        coords={
            "latitude": target.latitude.values,
            "longitude": target.longitude.values,
        },
    )
    return Era5FrameService(
        dataset=dataset,
        static=static,
        manifest=_manifest(),
        dataset_id="fixture",
        split="validation",
    )


@pytest.fixture(scope="module")
def prepared_frame():
    return _prepared_service().prepare_frame(TIMESTAMP)


def test_prepare_frame_enforces_shape_dtype_order_and_sst_policy(
    prepared_frame,
) -> None:
    assert prepared_frame.tensor.shape == (1, 28, 360, 720)
    assert prepared_frame.tensor.dtype == np.float32
    assert tuple(channel.name for channel in prepared_frame.channels) == CHANNEL_NAMES
    assert prepared_frame.validation.valid is True
    assert prepared_frame.validation.missing_values == 1
    assert not prepared_frame.ocean_mask[0, 0]
    assert np.isnan(
        prepared_frame.tensor[0, CHANNEL_NAMES.index("sst"), 0, 0]
    )
    for index in range(len(CHANNEL_NAMES)):
        if index != CHANNEL_NAMES.index("sst"):
            assert prepared_frame.tensor[0, index, 0, 0] == float(index)


def test_prepare_frame_uses_canonical_coordinate_orientation(prepared_frame) -> None:
    assert prepared_frame.latitude[0] == -89.75
    assert prepared_frame.latitude[-1] == 89.75
    assert np.all(np.diff(prepared_frame.latitude) > 0)
    assert prepared_frame.longitude[0] == 0.25
    assert prepared_frame.longitude[-1] == 359.75
    assert np.all(np.diff(prepared_frame.longitude) > 0)


def test_timestamp_resolution_is_exact_and_reports_neighbours() -> None:
    service = _prepared_service()
    with pytest.raises(TimestampNotAvailable) as captured:
        service.prepare_frame(datetime(2020, 1, 1, 17, tzinfo=UTC))
    assert captured.value.details == {
        "requested": "2020-01-01T17:00:00Z",
        "nearest_before": None,
        "nearest_after": "2020-01-01T18:00:00Z",
    }


def test_npz_export_is_accepted_by_existing_codec_loader(prepared_frame) -> None:
    payload = serialize_frame_npz(prepared_frame)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert set(archive.files) == {"data", "channel_order"}
        assert archive["data"].shape == (1, 28, 360, 720)
        assert archive["data"].dtype == np.float32
    loaded, order = _read_npz(payload)
    np.testing.assert_array_equal(loaded, prepared_frame.tensor)
    assert order == CHANNEL_NAMES


def test_validation_rejects_non_sst_nan(prepared_frame) -> None:
    invalid = prepared_frame.tensor.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(NonFiniteValuesError, match="t2m"):
        validate_prepared_frame(
            tensor=invalid,
            ocean_mask=prepared_frame.ocean_mask,
            latitude=prepared_frame.latitude,
            longitude=prepared_frame.longitude,
            channel_order=CHANNEL_NAMES,
        )


def test_validation_rejects_infinity_even_on_masked_sst(prepared_frame) -> None:
    invalid = prepared_frame.tensor.copy()
    invalid[0, CHANNEL_NAMES.index("sst"), 0, 0] = np.inf
    with pytest.raises(NonFiniteValuesError, match="infinity"):
        validate_prepared_frame(
            tensor=invalid,
            ocean_mask=prepared_frame.ocean_mask,
            latitude=prepared_frame.latitude,
            longitude=prepared_frame.longitude,
            channel_order=CHANNEL_NAMES,
        )


def test_validation_preserves_existing_sst_finite_mask_policy(
    prepared_frame,
) -> None:
    values = prepared_frame.tensor.copy()
    values[0, CHANNEL_NAMES.index("sst"), 0, 1] = np.nan
    summary = validate_prepared_frame(
        tensor=values,
        ocean_mask=prepared_frame.ocean_mask,
        latitude=prepared_frame.latitude,
        longitude=prepared_frame.longitude,
        channel_order=CHANNEL_NAMES,
    )
    assert summary.valid is True
    assert summary.sst_ocean_missing_values == 1


def test_validation_rejects_wrong_shape_and_coordinate_order(prepared_frame) -> None:
    with pytest.raises(FrameValidationError, match="shape"):
        validate_prepared_frame(
            tensor=prepared_frame.tensor[:, :, :-1],
            ocean_mask=prepared_frame.ocean_mask,
            latitude=prepared_frame.latitude[:-1],
            longitude=prepared_frame.longitude,
            channel_order=CHANNEL_NAMES,
        )
    with pytest.raises(InvalidCoordinateOrderError):
        validate_prepared_frame(
            tensor=prepared_frame.tensor,
            ocean_mask=prepared_frame.ocean_mask,
            latitude=prepared_frame.latitude[::-1],
            longitude=prepared_frame.longitude,
            channel_order=CHANNEL_NAMES,
        )


def test_prepared_store_rejects_missing_and_duplicate_channels() -> None:
    service = _prepared_service()
    missing = service.dataset.isel(channel=slice(0, -1))
    with pytest.raises(MissingChannelError):
        Era5FrameService(
            dataset=missing,
            static=service.static,
            manifest=service.manifest,
            dataset_id="fixture",
            split="validation",
        )

    labels = list(service.dataset.channel.values)
    labels[-1] = labels[0]
    duplicate = service.dataset.assign_coords(channel=labels)
    with pytest.raises(FrameValidationError, match="duplicate"):
        Era5FrameService(
            dataset=duplicate,
            static=service.static,
            manifest=service.manifest,
            dataset_id="fixture",
            split="validation",
        )


def _native_fixture() -> tuple[xr.Dataset, xr.Dataset]:
    latitude = np.array([90.0, 30.0, -30.0, -90.0])
    longitude = np.arange(0.0, 360.0, 45.0)
    time = [np.datetime64("2020-01-01T18:00:00")]
    values: dict[str, tuple] = {}
    for channel in CHANNEL_SPEC:
        if channel.level is None:
            values[channel.source_name] = (
                ("time", "latitude", "longitude"),
                np.full(
                    (1, latitude.size, longitude.size),
                    channel.index,
                    dtype=np.float64,
                ),
            )
    for source_name in {
        channel.source_name for channel in CHANNEL_SPEC if channel.level is not None
    }:
        data = np.empty(
            (1, len(PRESSURE_LEVELS), latitude.size, longitude.size),
            dtype=np.float64,
        )
        for level_index, level in enumerate(PRESSURE_LEVELS):
            channel = next(
                item
                for item in CHANNEL_SPEC
                if item.source_name == source_name and item.level == level
            )
            data[:, level_index] = channel.index
        values[source_name] = (
            ("time", "level", "latitude", "longitude"),
            data,
        )
    source = xr.Dataset(
        values,
        coords={
            "time": time,
            "level": list(PRESSURE_LEVELS),
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    target = xr.Dataset(
        coords={
            "latitude": [-67.5, -22.5, 22.5, 67.5],
            "longitude": np.arange(22.5, 360.0, 45.0),
        }
    )
    return source, target


def test_native_adapter_matches_existing_channel_and_remapping_pipeline() -> None:
    source, target = _native_fixture()
    tensor = prepare_native_model_tensor(
        source,
        np.datetime64("2020-01-01T18:00:00"),
        target_grid=target,
    )
    assert tensor.dims == ("time", "channel", "latitude", "longitude")
    assert tuple(str(value) for value in tensor.channel.values) == CHANNEL_NAMES
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor.latitude.values, target.latitude.values)
    for channel in CHANNEL_SPEC:
        np.testing.assert_allclose(
            tensor.sel(channel=channel.name).values,
            float(channel.index),
            rtol=0.0,
            atol=1e-6,
        )
