"""First-order conservative remapping for regular global latitude/longitude grids.

The implementation is intentionally dependency-light: it uses one sparse
spatial weight matrix and supports the descending ERA5 latitude coordinate.
It is appropriate for bounded batches such as a WeatherBench2 sample.  It is
not a substitute for a tiled/distributed global regridding service.
"""
from __future__ import annotations

import hashlib
import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix, csr_matrix


_WEIGHT_CACHE: dict[tuple[bytes, bytes, bytes, bytes], csr_matrix] = {}


def _cell_bounds(centres: np.ndarray, *, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    """Return monotonic cell bounds for monotonically increasing centres."""

    values = np.asarray(centres, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("cell centres must be a strictly increasing one-dimensional array")
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = (values[:-1] + values[1:]) * 0.5
    bounds[0] = values[0] - (bounds[1] - values[0])
    bounds[-1] = values[-1] + (values[-1] - bounds[-2])
    if lower is not None:
        bounds[0] = max(bounds[0], lower)
    if upper is not None:
        bounds[-1] = min(bounds[-1], upper)
    return bounds


def _axis_weights(
    source: np.ndarray,
    target: np.ndarray,
    *,
    spherical_latitude: bool,
    periodic: bool,
) -> csr_matrix:
    """Build target-by-source overlap weights for a single spatial axis."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_order = np.argsort(source)
    target_order = np.argsort(target)
    source_sorted = source[source_order]
    target_sorted = target[target_order]
    source_bounds = _cell_bounds(
        source_sorted,
        lower=-90.0 if spherical_latitude else None,
        upper=90.0 if spherical_latitude else None,
    )
    target_bounds = _cell_bounds(
        target_sorted,
        lower=-90.0 if spherical_latitude else None,
        upper=90.0 if spherical_latitude else None,
    )

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for sorted_target_index in range(target_sorted.size):
        lo = target_bounds[sorted_target_index]
        hi = target_bounds[sorted_target_index + 1]
        if spherical_latitude:
            denominator = np.sin(np.deg2rad(hi)) - np.sin(np.deg2rad(lo))
        else:
            denominator = hi - lo
        if denominator <= 0:
            raise ValueError("target cells must have positive area")

        for sorted_source_index in range(source_sorted.size):
            source_lo = source_bounds[sorted_source_index]
            source_hi = source_bounds[sorted_source_index + 1]
            shifts = (-360.0, 0.0, 360.0) if periodic else (0.0,)
            for shift in shifts:
                overlap_lo = max(lo, source_lo + shift)
                overlap_hi = min(hi, source_hi + shift)
                if overlap_hi <= overlap_lo:
                    continue
                if spherical_latitude:
                    numerator = np.sin(np.deg2rad(overlap_hi)) - np.sin(np.deg2rad(overlap_lo))
                else:
                    numerator = overlap_hi - overlap_lo
                rows.append(int(target_order[sorted_target_index]))
                cols.append(int(source_order[sorted_source_index]))
                values.append(float(numerator / denominator))

    weights = coo_matrix((values, (rows, cols)), shape=(target.size, source.size)).tocsr()
    row_sums = np.asarray(weights.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("source grid does not fully cover target grid")
    return weights


def _build_conservative_weights(
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
) -> csr_matrix:
    """Return a sparse target-cell-by-source-cell conservative weight matrix."""

    latitude = _axis_weights(lat_in, lat_out, spherical_latitude=True, periodic=False)
    longitude = _axis_weights(lon_in, lon_out, spherical_latitude=False, periodic=True)
    # For C-order flattened [latitude, longitude] fields, this Kronecker
    # product maps each latitude row with the matching longitude weights.
    from scipy.sparse import kron

    return kron(latitude, longitude, format="csr")


def _coordinate_digest(values: np.ndarray) -> bytes:
    """Return a stable cache key component without retaining coordinate arrays."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8)).digest()


def _cached_conservative_weights(
    lat_in: np.ndarray,
    lon_in: np.ndarray,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
) -> csr_matrix:
    """Reuse immutable weights across consecutive WeatherBench2 frames."""

    key = tuple(_coordinate_digest(values) for values in (lat_in, lon_in, lat_out, lon_out))
    weights = _WEIGHT_CACHE.get(key)
    if weights is None:
        weights = _build_conservative_weights(lat_in, lon_in, lat_out, lon_out)
        _WEIGHT_CACHE[key] = weights
    return weights


def _remap_values(values: np.ndarray, weights: csr_matrix, output_shape: tuple[int, int]) -> np.ndarray:
    """Conservatively remap ``[..., lat, lon]`` and renormalise finite values."""

    array = np.asarray(values, dtype=np.float32)
    leading_shape = array.shape[:-2]
    flat = array.reshape(-1, array.shape[-2] * array.shape[-1])
    finite = np.isfinite(flat)
    weighted_values = weights.dot(np.where(finite, flat, 0.0).T).T
    coverage = weights.dot(finite.astype(np.float32).T).T
    # SST is missing over land.  Dividing by coverage prevents coastal cells
    # from being biased toward zero; a cell with no valid source stays NaN.
    remapped = np.divide(
        weighted_values,
        coverage,
        out=np.full_like(weighted_values, np.nan, dtype=np.float32),
        where=coverage > 0,
    )
    return remapped.reshape(*leading_shape, *output_shape).astype(np.float32, copy=False)


def conservative_remap(ds_in: xr.Dataset, ds_out_grid: xr.Dataset) -> xr.Dataset:
    """Conservatively remap every spatial variable in a bounded dataset.

    Both ascending and descending latitude inputs are supported.  Longitude
    is treated as periodic.  Missing source values are excluded and the
    remaining overlap weights are re-normalised, which preserves SST over
    coastal target cells while leaving entirely dry cells as NaN.
    """

    lat_in = np.asarray(ds_in["latitude"].values)
    lon_in = np.asarray(ds_in["longitude"].values)
    lat_out = np.asarray(ds_out_grid["latitude"].values)
    lon_out = np.asarray(ds_out_grid["longitude"].values)
    weights = _cached_conservative_weights(lat_in, lon_in, lat_out, lon_out)

    remapped_variables: dict[str, xr.DataArray] = {}
    for name, field in ds_in.data_vars.items():
        if "latitude" not in field.dims or "longitude" not in field.dims:
            remapped_variables[name] = field.copy(deep=False)
            continue
        non_spatial_dims = tuple(dim for dim in field.dims if dim not in {"latitude", "longitude"})
        transposed = field.transpose(*non_spatial_dims, "latitude", "longitude")
        values = _remap_values(transposed.values, weights, (lat_out.size, lon_out.size))
        coords = {dim: transposed.coords[dim] for dim in non_spatial_dims}
        coords["latitude"] = lat_out
        coords["longitude"] = lon_out
        remapped_variables[name] = xr.DataArray(
            values,
            dims=(*non_spatial_dims, "latitude", "longitude"),
            coords=coords,
            attrs=dict(field.attrs),
            name=name,
        ).transpose(*field.dims)
    return xr.Dataset(remapped_variables, attrs=dict(ds_in.attrs))
