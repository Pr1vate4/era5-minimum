"""Prepare exact real ERA5 timestamps for the accepted 28-channel codec."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from era5_minimum.data.channel_spec import (
    CHANNEL_NAMES,
    CHANNEL_SPEC,
    Channel,
    validate_wb2_compatibility,
)
from era5_minimum.data.era5_28ch import (
    assemble_model_tensor,
    select_dynamic,
)
from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.model_input import (
    CANONICAL_FRAME_SHAPE,
    NPZ_CHANNEL_ORDER_KEY,
    NPZ_DATA_KEY,
    ModelInputContractError,
    validate_model_input,
)
from era5_minimum.data.remapping import conservative_remap


class Era5FrameError(RuntimeError):
    """Base class for stable ERA5 frame API errors."""

    code = "era5_frame_error"
    status_code = 422

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.details}


class DatasetUnavailable(Era5FrameError):
    code = "dataset_unavailable"
    status_code = 503


class TimestampNotAvailable(Era5FrameError):
    code = "timestamp_not_available"
    status_code = 404


class FrameValidationError(Era5FrameError):
    code = "invalid_frame"


class UnsupportedGridError(FrameValidationError):
    code = "unsupported_grid"


class MissingChannelError(FrameValidationError):
    code = "missing_channel"


class InvalidCoordinateOrderError(FrameValidationError):
    code = "invalid_coordinate_order"


class InvalidUnitError(FrameValidationError):
    code = "invalid_unit"


class NonFiniteValuesError(FrameValidationError):
    code = "non_finite_values"


@dataclass(frozen=True)
class FrameValidationSummary:
    """Central validation result for one physical model input."""

    valid: bool
    missing_values: int
    sst_masked_values: int
    sst_ocean_missing_values: int
    invalid_values: int


@dataclass(frozen=True)
class PreparedEra5Frame:
    """One immutable, independently prepared physical ERA5 frame."""

    timestamp: datetime
    tensor: np.ndarray
    ocean_mask: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    channels: tuple[Channel, ...]
    dataset_id: str
    dataset_source: str
    dataset_split: str
    schema_version: str
    validation: FrameValidationSummary


def _iso_timestamp(value: np.datetime64) -> str:
    return np.datetime_as_string(value.astype("datetime64[ns]"), unit="s") + "Z"


def _normalise_unit(value: str) -> str:
    return (
        value.replace("**", "")
        .replace(" ", "")
        .replace("–", "-")
        .replace("(", "")
        .replace(")", "")
    )


def _as_utc_datetime64(value: datetime, *, timezone_required: bool) -> np.datetime64:
    if value.tzinfo is None or value.utcoffset() is None:
        if timezone_required:
            raise FrameValidationError(
                "timestamp must include an explicit UTC offset",
                reason="timezone_required",
            )
        value = value.replace(tzinfo=UTC)
    normalized = value.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(normalized, "ns")


def _datetime_from_np(value: np.datetime64) -> datetime:
    return datetime.fromisoformat(_iso_timestamp(value).replace("Z", "+00:00"))


def _validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    order = tuple(str(value) for value in manifest.get("channel_order", ()))
    if order != CHANNEL_NAMES:
        raise FrameValidationError(
            "dataset manifest channel_order does not match the accepted model",
            reason="channel_order_mismatch",
        )

    entries = manifest.get("channels")
    if not isinstance(entries, list):
        raise FrameValidationError(
            "dataset manifest does not contain channel metadata",
            reason="missing_channel_metadata",
        )
    by_name = {
        str(entry.get("name")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    for channel in CHANNEL_SPEC:
        entry = by_name.get(channel.name)
        if entry is None:
            raise MissingChannelError(
                f"dataset manifest is missing channel {channel.name}",
                channel=channel.name,
            )
        source_name = str(entry.get("source_name", ""))
        if source_name != channel.source_name:
            raise MissingChannelError(
                f"channel {channel.name} maps to unexpected source {source_name!r}",
                channel=channel.name,
                expected_source=channel.source_name,
            )
        level = entry.get("pressure_level_hpa", entry.get("level_hpa"))
        if level != channel.level:
            raise MissingChannelError(
                f"channel {channel.name} has unexpected pressure level {level!r}",
                channel=channel.name,
                expected_level=channel.level,
            )
        actual_unit = str(entry.get("units", ""))
        if _normalise_unit(actual_unit) != _normalise_unit(channel.units):
            raise InvalidUnitError(
                f"channel {channel.name} has unexpected unit {actual_unit!r}",
                channel=channel.name,
                expected_unit=channel.units,
            )


def _validate_axis(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    axis: str,
) -> None:
    values = np.asarray(actual, dtype=np.float64)
    if values.shape != expected.shape or not np.allclose(
        values, expected, rtol=0.0, atol=1e-10
    ):
        raise InvalidCoordinateOrderError(
            f"{axis} does not match the canonical model grid",
            axis=axis,
            expected_size=int(expected.size),
        )


def validate_prepared_frame(
    *,
    tensor: np.ndarray,
    ocean_mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    channel_order: tuple[str, ...],
) -> FrameValidationSummary:
    """Validate shape, semantics, coordinates and the established SST policy."""

    try:
        values = validate_model_input(
            tensor,
            channel_order=channel_order,
            allow_unbatched=False,
        )
    except ModelInputContractError as exc:
        raise FrameValidationError(str(exc), reason="model_input_contract") from exc

    target = get_target_grid_05()
    _validate_axis(
        latitude,
        np.asarray(target.latitude.values),
        axis="latitude",
    )
    _validate_axis(
        longitude,
        np.asarray(target.longitude.values),
        axis="longitude",
    )

    mask = np.asarray(ocean_mask)
    if mask.shape != CANONICAL_FRAME_SHAPE[-2:]:
        raise FrameValidationError(
            "ocean mask does not match the model grid",
            reason="ocean_mask_shape",
            actual_shape=list(mask.shape),
        )
    mask = mask.astype(bool, copy=False)

    for index, channel in enumerate(CHANNEL_SPEC):
        field = values[0, index]
        infinity_count = int(np.isinf(field).sum())
        if infinity_count:
            raise NonFiniteValuesError(
                f"channel {channel.name} contains infinity",
                channel=channel.name,
                count=infinity_count,
                kind="infinity",
            )
        nan = np.isnan(field)
        if channel.name != "sst" and nan.any():
            raise NonFiniteValuesError(
                f"channel {channel.name} contains missing values",
                channel=channel.name,
                count=int(nan.sum()),
                kind="nan",
            )

    sst = values[0, CHANNEL_NAMES.index("sst")]
    missing_values = int(np.isnan(sst).sum())
    masked_values = int((~mask).sum())
    ocean_missing_values = int((np.isnan(sst) & mask).sum())
    return FrameValidationSummary(
        valid=True,
        missing_values=missing_values,
        sst_masked_values=masked_values,
        sst_ocean_missing_values=ocean_missing_values,
        invalid_values=0,
    )


def serialize_frame_npz(frame: PreparedEra5Frame) -> bytes:
    """Serialize using the exact keys accepted by the existing NPZ codec path."""

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{
            NPZ_DATA_KEY: frame.tensor,
            NPZ_CHANNEL_ORDER_KEY: np.asarray(CHANNEL_NAMES),
        },
    )
    return buffer.getvalue()


def prepare_native_model_tensor(
    dataset: xr.Dataset,
    timestamp: np.datetime64,
    *,
    target_grid: xr.Dataset | None = None,
) -> xr.DataArray:
    """Run the established selective-load and conservative-remap pipeline."""

    selected = select_dynamic(
        dataset,
        pd.DatetimeIndex([timestamp]),
    ).load()
    remapped = conservative_remap(
        selected,
        target_grid if target_grid is not None else get_target_grid_05(),
    )
    return assemble_model_tensor(remapped)


class Era5FrameService:
    """Lazy adapter from a configured WeatherBench/Zarr split to model input."""

    def __init__(
        self,
        *,
        dataset: xr.Dataset,
        static: xr.Dataset,
        manifest: Mapping[str, Any],
        dataset_id: str,
        split: str,
    ) -> None:
        self.dataset = dataset
        self.static = static
        self.manifest = dict(manifest)
        self.dataset_id = dataset_id
        self.split = split
        self.dataset_source = str(self.manifest.get("source_uri", ""))
        self.schema_version = str(
            self.manifest.get(
                "schema_version",
                self.manifest.get("format_version", "unversioned"),
            )
        )
        self._mask_lock = Lock()
        self._cached_ocean_mask: np.ndarray | None = None
        self._source_kind = (
            "prepared" if "data" in self.dataset.data_vars else "native"
        )
        self._validate_dataset_contract()

    @classmethod
    def from_root(cls, root: str | Path, split: str) -> "Era5FrameService":
        """Open the configured local dataset without downloading any data."""

        path = Path(root)
        manifest_path = path / "manifest.json"
        store = path / f"{split}.zarr"
        static_store = path / "static.zarr"
        try:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"missing manifest: {manifest_path}")
            if not store.is_dir():
                raise FileNotFoundError(f"missing split: {store}")
            if not static_store.is_dir():
                raise FileNotFoundError(f"missing static fields: {static_store}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset = xr.open_zarr(store, consolidated=True)
            static = xr.open_zarr(static_store, consolidated=True)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise DatasetUnavailable(
                f"configured ERA5 dataset cannot be opened: {exc}",
                dataset_root=str(path),
                split=split,
            ) from exc
        return cls(
            dataset=dataset,
            static=static,
            manifest=manifest,
            dataset_id=str(manifest.get("dataset_id", path.name)),
            split=split,
        )

    def _validate_dataset_contract(self) -> None:
        _validate_manifest_contract(self.manifest)
        required_coordinates = {"time", "latitude", "longitude"}
        missing_coordinates = sorted(required_coordinates - set(self.dataset.coords))
        if missing_coordinates:
            raise DatasetUnavailable(
                "configured ERA5 split is missing required coordinates",
                missing_coordinates=missing_coordinates,
            )
        times = np.asarray(self.dataset.time.values, dtype="datetime64[ns]")
        if times.ndim != 1 or times.size == 0 or np.isnat(times).any():
            raise DatasetUnavailable("configured ERA5 split has invalid timestamps")
        if times.size > 1 and not np.all(times[:-1] < times[1:]):
            raise DatasetUnavailable(
                "configured ERA5 timestamps must be unique and increasing"
            )
        self._timestamps = times

        if self._source_kind == "prepared":
            self._validate_prepared_store()
        else:
            self._validate_native_store()

    def _validate_prepared_store(self) -> None:
        field = self.dataset["data"]
        expected_dims = ("time", "channel", "latitude", "longitude")
        if set(field.dims) != set(expected_dims) or "channel" not in self.dataset.coords:
            raise DatasetUnavailable(
                "prepared Zarr data must expose time/channel/latitude/longitude"
            )
        channel_values = tuple(str(value) for value in self.dataset.channel.values)
        if len(channel_values) != len(set(channel_values)):
            raise FrameValidationError(
                "prepared Zarr contains duplicate channel labels",
                reason="duplicate_channel",
            )
        missing = [name for name in CHANNEL_NAMES if name not in channel_values]
        if missing:
            raise MissingChannelError(
                "prepared Zarr is missing canonical channels",
                channels=missing,
            )
        target = get_target_grid_05()
        _validate_axis(
            self.dataset.latitude.values,
            np.asarray(target.latitude.values),
            axis="latitude",
        )
        _validate_axis(
            self.dataset.longitude.values,
            np.asarray(target.longitude.values),
            axis="longitude",
        )

    def _validate_native_store(self) -> None:
        try:
            validate_wb2_compatibility(self.dataset)
        except ValueError as exc:
            raise DatasetUnavailable(str(exc)) from exc
        if self.dataset.sizes.get("latitude") != 721:
            raise UnsupportedGridError(
                "native WeatherBench2 latitude size must be 721",
                actual_size=self.dataset.sizes.get("latitude"),
            )
        if self.dataset.sizes.get("longitude") != 1440:
            raise UnsupportedGridError(
                "native WeatherBench2 longitude size must be 1440",
                actual_size=self.dataset.sizes.get("longitude"),
            )
        expected_latitude = np.linspace(90.0, -90.0, 721, dtype=np.float64)
        expected_longitude = np.arange(1440, dtype=np.float64) * 0.25
        _validate_axis(
            self.dataset.latitude.values,
            expected_latitude,
            axis="source latitude",
        )
        _validate_axis(
            self.dataset.longitude.values,
            expected_longitude,
            axis="source longitude",
        )

    def list_timestamps(
        self,
        *,
        start: datetime | None,
        end: datetime | None,
        offset: int,
        limit: int,
    ) -> tuple[tuple[datetime, ...], int]:
        """Return a bounded exact timestamp page from the configured split."""

        selected = self._timestamps
        if start is not None:
            selected = selected[
                selected >= _as_utc_datetime64(start, timezone_required=False)
            ]
        if end is not None:
            selected = selected[
                selected <= _as_utc_datetime64(end, timezone_required=False)
            ]
        if start is not None and end is not None:
            if _as_utc_datetime64(start, timezone_required=False) > _as_utc_datetime64(
                end, timezone_required=False
            ):
                raise FrameValidationError(
                    "start must not be later than end",
                    reason="invalid_timestamp_range",
                )
        total = int(selected.size)
        page = selected[offset : offset + limit]
        return tuple(_datetime_from_np(value) for value in page), total

    def _resolve_timestamp(self, requested: datetime) -> np.datetime64:
        target = _as_utc_datetime64(requested, timezone_required=True)
        index = int(np.searchsorted(self._timestamps, target))
        if index < self._timestamps.size and self._timestamps[index] == target:
            return self._timestamps[index]
        before = (
            _iso_timestamp(self._timestamps[index - 1]) if index > 0 else None
        )
        after = (
            _iso_timestamp(self._timestamps[index])
            if index < self._timestamps.size
            else None
        )
        raise TimestampNotAvailable(
            "the requested timestamp is not present in the configured split",
            requested=_iso_timestamp(target),
            nearest_before=before,
            nearest_after=after,
        )

    def _load_tensor(self, timestamp: np.datetime64) -> np.ndarray:
        if self._source_kind == "prepared":
            frame = (
                self.dataset["data"]
                .sel(time=[timestamp], channel=list(CHANNEL_NAMES))
                .transpose("time", "channel", "latitude", "longitude")
                .load()
            )
        else:
            frame = prepare_native_model_tensor(
                self.dataset,
                timestamp,
            )
        return np.asarray(frame.values, dtype=np.float32)

    def _load_ocean_mask(self) -> np.ndarray:
        with self._mask_lock:
            if self._cached_ocean_mask is not None:
                return self._cached_ocean_mask
            target = get_target_grid_05()
            if self._source_kind == "prepared":
                if "ocean_mask" not in self.static:
                    raise FrameValidationError(
                        "prepared dataset static.zarr is missing ocean_mask",
                        reason="missing_ocean_mask",
                    )
                _validate_axis(
                    self.static.latitude.values,
                    np.asarray(target.latitude.values),
                    axis="ocean mask latitude",
                )
                _validate_axis(
                    self.static.longitude.values,
                    np.asarray(target.longitude.values),
                    axis="ocean mask longitude",
                )
                mask = self.static["ocean_mask"].load().values.astype(
                    bool, copy=False
                )
            else:
                if "land_sea_mask" not in self.static:
                    raise FrameValidationError(
                        "native dataset static.zarr is missing land_sea_mask",
                        reason="missing_land_sea_mask",
                    )
                land = conservative_remap(
                    self.static[["land_sea_mask"]].load(),
                    target,
                )["land_sea_mask"].values
                if not np.isfinite(land).all():
                    raise NonFiniteValuesError(
                        "remapped land_sea_mask contains non-finite values",
                        channel="land_sea_mask",
                        count=int((~np.isfinite(land)).sum()),
                    )
                mask = land <= 0.5
            if mask.shape != CANONICAL_FRAME_SHAPE[-2:]:
                raise UnsupportedGridError(
                    "ocean mask does not match the canonical model grid",
                    actual_shape=list(mask.shape),
                )
            cached = np.asarray(mask, dtype=bool)
            cached.setflags(write=False)
            self._cached_ocean_mask = cached
            return cached

    def prepare_frame(self, timestamp: datetime) -> PreparedEra5Frame:
        """Load exactly one timestamp and return validated physical model input."""

        resolved = self._resolve_timestamp(timestamp)
        tensor = self._load_tensor(resolved)
        ocean_mask = self._load_ocean_mask().copy()
        target = get_target_grid_05()
        latitude = np.asarray(target.latitude.values).copy()
        longitude = np.asarray(target.longitude.values).copy()
        validation = validate_prepared_frame(
            tensor=tensor,
            ocean_mask=ocean_mask,
            latitude=latitude,
            longitude=longitude,
            channel_order=CHANNEL_NAMES,
        )
        for array in (tensor, ocean_mask, latitude, longitude):
            array.setflags(write=False)
        return PreparedEra5Frame(
            timestamp=_datetime_from_np(resolved),
            tensor=tensor,
            ocean_mask=ocean_mask,
            latitude=latitude,
            longitude=longitude,
            channels=CHANNEL_SPEC,
            dataset_id=self.dataset_id,
            dataset_source=self.dataset_source,
            dataset_split=self.split,
            schema_version=self.schema_version,
            validation=validation,
        )
