"""Leakage-safe scientific metrics for local codec reconstructions."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

EVALUATOR_VERSION = "local-scientific-v1"
GRAVITY_M_S2 = 9.80665
SURFACE_CHANNEL_COUNT = 8
PRESSURE_CHANNEL_COUNT = 20


def train_statistics_checksum(
    train_std: np.ndarray,
    train_ranges: np.ndarray,
) -> str:
    """Return a stable checksum for train-only evaluator statistics."""

    std = np.asarray(train_std, dtype=np.float64).reshape(-1)
    ranges = np.asarray(train_ranges, dtype=np.float64).reshape(-1)
    if std.shape != ranges.shape:
        raise ValueError("train_std and train_ranges must have the same length")
    if not np.all(np.isfinite(std)) or not np.all(np.isfinite(ranges)):
        raise ValueError("train statistics must be finite")
    payload = {
        "train_only": True,
        "std": [float(value) for value in std],
        "ranges": [float(value) for value in ranges],
    }
    return _statistics_checksum(payload)


def fit_train_channel_ranges(
    train_physical: np.ndarray,
    *,
    validity_mask: np.ndarray,
) -> np.ndarray:
    """Fit per-channel physical ranges using only valid training values."""

    bounds = fit_train_channel_bounds(train_physical, validity_mask=validity_mask)
    return (bounds[:, 1] - bounds[:, 0]).astype(np.float64)


def fit_train_channel_bounds(
    train_physical: np.ndarray,
    *,
    validity_mask: np.ndarray,
) -> np.ndarray:
    """Fit per-channel ``[minimum, maximum]`` bounds on the training split."""

    values, mask = _validate_training_arrays(train_physical, validity_mask)
    bounds = np.empty((values.shape[1], 2), dtype=np.float64)
    for channel_index in range(values.shape[1]):
        valid = mask[:, channel_index] & np.isfinite(values[:, channel_index])
        if not np.any(valid):
            raise ValueError(f"training channel {channel_index} has no valid finite values")
        channel_values = values[:, channel_index][valid]
        bounds[channel_index] = (
            float(np.min(channel_values)),
            float(np.max(channel_values)),
        )
    return bounds


def evaluate_reconstruction(
    original_physical: np.ndarray,
    reconstruction_physical: np.ndarray,
    *,
    latitudes: np.ndarray,
    channel_order: tuple[str, ...],
    train_std: np.ndarray,
    train_ranges: np.ndarray,
    validity_mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a physical reconstruction with latitude-weighted metrics.

    ``train_std`` and ``train_ranges`` must be fitted on the training split.
    The evaluator does not fit either statistic from validation or test values.
    """

    original, reconstruction, mask, latitudes = _validate_evaluation_arrays(
        original_physical,
        reconstruction_physical,
        latitudes=latitudes,
        channel_order=channel_order,
        train_std=train_std,
        train_ranges=train_ranges,
        validity_mask=validity_mask,
    )
    channel_count = original.shape[1]
    std = np.asarray(train_std, dtype=np.float64).reshape(-1)
    ranges = _normalize_ranges(train_ranges, channel_count)
    latitude_weights = _latitude_weights(latitudes)

    per_channel: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(channel_order):
        valid = (
            (mask[:, channel_index] > 0)
            & np.isfinite(original[:, channel_index])
            & np.isfinite(reconstruction[:, channel_index])
        )
        rmse_value, valid_count = _weighted_rmse(
            original[:, channel_index],
            reconstruction[:, channel_index],
            valid,
            latitude_weights,
            channel_name=channel,
        )
        psnr_db, psnr_status = _psnr(
            rmse_value,
            float(ranges[channel_index]),
        )
        per_channel.append(
            {
                "channel": channel,
                "index": channel_index,
                "rmse_physical": rmse_value,
                "nrmse": float(rmse_value / std[channel_index]),
                "psnr_db": psnr_db,
                "psnr_status": psnr_status,
                "train_std": float(std[channel_index]),
                "train_range": float(ranges[channel_index]),
                "valid_value_count": valid_count,
                "invalid_value_count": int(valid.size - valid_count),
            }
        )

    groups = _group_scores(per_channel, channel_order)
    diagnostics = _physical_diagnostics(
        original,
        reconstruction,
        mask,
        channel_order,
        latitude_weights,
    )
    train_statistics_payload = {
        "train_only": True,
        "std": [float(value) for value in std],
        "ranges": [float(value) for value in ranges],
    }
    train_statistics_payload["checksum"] = train_statistics_checksum(std, ranges)

    finite_psnr = [
        float(row["psnr_db"])
        for row in per_channel
        if row["psnr_db"] is not None and math.isfinite(float(row["psnr_db"]))
    ]
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "metric_space": "physical",
        "latitude_weighting": True,
        "train_only": True,
        "channel_order": list(channel_order),
        "per_channel": per_channel,
        "groups": groups,
        "surface_score": groups["surface"]["nrmse"],
        "pressure_score": groups["pressure"]["nrmse"],
        "overall_score": groups["overall"]["nrmse"],
        "mean_finite_psnr_db": (
            float(np.mean(finite_psnr, dtype=np.float64)) if finite_psnr else None
        ),
        "physical_diagnostics": diagnostics,
        "train_statistics": train_statistics_payload,
        "invalid_value_count": int(mask.size - np.count_nonzero(mask)),
    }


def _validate_training_arrays(
    values: np.ndarray,
    validity_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values_array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(validity_mask, dtype=bool)
    if values_array.ndim != 4:
        raise ValueError(f"train_physical must have shape [N,C,H,W], got {values_array.shape}")
    if mask_array.shape != values_array.shape:
        raise ValueError(
            f"validity_mask shape {mask_array.shape} does not match train_physical {values_array.shape}"
        )
    return values_array, mask_array


def _validate_evaluation_arrays(
    original: np.ndarray,
    reconstruction: np.ndarray,
    *,
    latitudes: np.ndarray,
    channel_order: tuple[str, ...],
    train_std: np.ndarray,
    train_ranges: np.ndarray,
    validity_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original_array = np.asarray(original, dtype=np.float64)
    reconstruction_array = np.asarray(reconstruction, dtype=np.float64)
    mask_array = np.asarray(validity_mask, dtype=bool)
    latitudes_array = np.asarray(latitudes, dtype=np.float64).reshape(-1)
    if original_array.ndim != 4:
        raise ValueError(
            f"original_physical must have shape [N,C,H,W], got {original_array.shape}"
        )
    if reconstruction_array.shape != original_array.shape:
        raise ValueError(
            f"reconstruction_physical shape {reconstruction_array.shape} "
            f"does not match original_physical {original_array.shape}"
        )
    if mask_array.shape != original_array.shape:
        raise ValueError(
            f"validity_mask shape {mask_array.shape} does not match "
            f"original_physical {original_array.shape}"
        )
    if len(channel_order) != original_array.shape[1]:
        raise ValueError("channel_order length does not match tensor channels")
    if len(set(channel_order)) != len(channel_order):
        raise ValueError("channel_order must contain unique channel names")
    if len(channel_order) != SURFACE_CHANNEL_COUNT + PRESSURE_CHANNEL_COUNT:
        raise ValueError("scientific evaluator requires exactly 28 channels")
    if latitudes_array.shape != (original_array.shape[2],):
        raise ValueError(
            f"latitudes must have shape ({original_array.shape[2]},), got {latitudes_array.shape}"
        )
    if not np.all(np.isfinite(latitudes_array)):
        raise ValueError("latitudes must be finite")
    std = np.asarray(train_std, dtype=np.float64).reshape(-1)
    if std.shape != (original_array.shape[1],):
        raise ValueError("train_std length does not match tensor channels")
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("train_std must contain finite positive values")
    _normalize_ranges(train_ranges, original_array.shape[1])
    return original_array, reconstruction_array, mask_array, latitudes_array


def _normalize_ranges(train_ranges: np.ndarray, channel_count: int) -> np.ndarray:
    ranges = np.asarray(train_ranges, dtype=np.float64)
    if ranges.shape == (channel_count, 2):
        ranges = ranges[:, 1] - ranges[:, 0]
    elif ranges.shape != (channel_count,):
        raise ValueError("train_ranges must have shape [C] or [C,2]")
    if not np.all(np.isfinite(ranges)) or np.any(ranges < 0):
        raise ValueError("train_ranges must contain finite non-negative values")
    return ranges


def _latitude_weights(latitudes: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(latitudes))
    return np.clip(weights, 0.0, None)


def _weighted_rmse(
    original: np.ndarray,
    reconstruction: np.ndarray,
    valid: np.ndarray,
    latitude_weights: np.ndarray,
    *,
    channel_name: str,
) -> tuple[float, int]:
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        raise ValueError(f"channel {channel_name} has no valid evaluation values")
    weights = latitude_weights.reshape(1, -1, 1)
    weighted_mask = weights * valid
    denominator = float(np.sum(weighted_mask, dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError(f"channel {channel_name} has no positive latitude weight")
    # ``NaN * 0`` is still NaN.  SST is deliberately NaN over land, so mask
    # invalid values before the weighted reduction rather than multiplying a
    # NaN error by a zero validity weight.
    squared_error = np.where(valid, (reconstruction - original) ** 2, 0.0)
    value = float(np.sqrt(np.sum(squared_error * weighted_mask, dtype=np.float64) / denominator))
    return value, int(np.count_nonzero(valid))


def _psnr(rmse_value: float, train_range: float) -> tuple[float | None, str]:
    if train_range <= 0.0:
        return None, "zero_train_range"
    if rmse_value == 0.0:
        return None, "perfect_reconstruction"
    return float(20.0 * math.log10(train_range / rmse_value)), "ok"


def _group_scores(
    per_channel: list[dict[str, Any]],
    channel_order: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    groups = {
        "surface": list(range(SURFACE_CHANNEL_COUNT)),
        "pressure": list(
            range(SURFACE_CHANNEL_COUNT, SURFACE_CHANNEL_COUNT + PRESSURE_CHANNEL_COUNT)
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, indices in groups.items():
        rows = [per_channel[index] for index in indices]
        result[name] = {
            "channels": [channel_order[index] for index in indices],
            "channel_count": len(rows),
            "rmse_physical": float(np.mean([row["rmse_physical"] for row in rows])),
            "nrmse": float(np.mean([row["nrmse"] for row in rows])),
            "psnr_db": _mean_psnr(rows),
        }
    result["overall"] = {
        "channels": list(channel_order),
        "channel_count": len(per_channel),
        "rmse_physical": float(
            0.5 * result["surface"]["rmse_physical"]
            + 0.5 * result["pressure"]["rmse_physical"]
        ),
        "nrmse": float(0.5 * result["surface"]["nrmse"] + 0.5 * result["pressure"]["nrmse"]),
        "psnr_db": _mean_psnr(per_channel),
    }
    return result


def _mean_psnr(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["psnr_db"])
        for row in rows
        if row["psnr_db"] is not None and math.isfinite(float(row["psnr_db"]))
    ]
    return float(np.mean(values, dtype=np.float64)) if values else None


def _physical_diagnostics(
    original: np.ndarray,
    reconstruction: np.ndarray,
    validity_mask: np.ndarray,
    channel_order: tuple[str, ...],
    latitude_weights: np.ndarray,
) -> dict[str, Any]:
    indices = {name: index for index, name in enumerate(channel_order)}
    diagnostics: dict[str, Any] = {}

    mslp_name = next((name for name in ("mslp", "msl") if name in indices), None)
    if mslp_name is not None:
        rmse_pa = _diagnostic_channel_rmse(
            original,
            reconstruction,
            validity_mask,
            indices[mslp_name],
            latitude_weights,
            mslp_name,
        )
        diagnostics["mslp"] = {
            "channel": mslp_name,
            "rmse_pa": rmse_pa,
            "rmse_hpa": float(rmse_pa / 100.0),
            "units": {"rmse_pa": "Pa", "rmse_hpa": "hPa"},
        }
        diagnostics["mslp_rmse_pa"] = rmse_pa
        diagnostics["mslp_rmse_hpa"] = float(rmse_pa / 100.0)

    if "tp6h" in indices:
        rmse_m = _diagnostic_channel_rmse(
            original,
            reconstruction,
            validity_mask,
            indices["tp6h"],
            latitude_weights,
            "tp6h",
        )
        diagnostics["tp6h"] = {
            "rmse_m": rmse_m,
            "rmse_mm_per_6h": float(rmse_m * 1000.0),
            "units": {"rmse_m": "m", "rmse_mm_per_6h": "mm/6h"},
        }
        diagnostics["tp6h_rmse_m"] = rmse_m
        diagnostics["tp6h_rmse_mm_per_6h"] = float(rmse_m * 1000.0)

    geopotential: dict[str, Any] = {}
    for channel, channel_index in indices.items():
        if not channel.startswith("Z"):
            continue
        rmse_z = _diagnostic_channel_rmse(
            original,
            reconstruction,
            validity_mask,
            channel_index,
            latitude_weights,
            channel,
        )
        row = {
            "rmse_m2_s2": rmse_z,
            "rmse_height_m": float(rmse_z / GRAVITY_M_S2),
            "rmse_z_over_g": float(rmse_z / GRAVITY_M_S2),
            "gravity_m_s2": GRAVITY_M_S2,
            "units": {"rmse_m2_s2": "m2 s-2", "rmse_height_m": "m"},
        }
        geopotential[channel] = row
        diagnostics[f"{channel}_rmse_m2_s2"] = rmse_z
        diagnostics[f"{channel}_rmse_height_m"] = row["rmse_height_m"]
    if geopotential:
        diagnostics["geopotential"] = geopotential

    for wind_component in ("u10", "v10"):
        if wind_component not in indices:
            continue
        rmse_component = _diagnostic_channel_rmse(
            original,
            reconstruction,
            validity_mask,
            indices[wind_component],
            latitude_weights,
            wind_component,
        )
        diagnostics[wind_component] = {
            "rmse_m_s": rmse_component,
            "units": {"rmse_m_s": "m s-1"},
        }
        diagnostics[f"{wind_component}_rmse_m_s"] = rmse_component

    if "u10" in indices and "v10" in indices:
        u_index = indices["u10"]
        v_index = indices["v10"]
        valid = (
            (validity_mask[:, u_index] > 0)
            & (validity_mask[:, v_index] > 0)
            & np.isfinite(original[:, u_index])
            & np.isfinite(original[:, v_index])
            & np.isfinite(reconstruction[:, u_index])
            & np.isfinite(reconstruction[:, v_index])
        )
        original_speed = np.hypot(original[:, u_index], original[:, v_index])
        reconstruction_speed = np.hypot(
            reconstruction[:, u_index], reconstruction[:, v_index]
        )
        rmse_speed, valid_count = _weighted_rmse(
            original_speed,
            reconstruction_speed,
            valid,
            latitude_weights,
            channel_name="wind_speed",
        )
        diagnostics["wind_speed"] = {
            "rmse_m_s": rmse_speed,
            "units": {"rmse_m_s": "m s-1"},
            "valid_value_count": valid_count,
        }
        diagnostics["wind_speed_rmse_m_s"] = rmse_speed
    return diagnostics


def _diagnostic_channel_rmse(
    original: np.ndarray,
    reconstruction: np.ndarray,
    validity_mask: np.ndarray,
    channel_index: int,
    latitude_weights: np.ndarray,
    channel_name: str,
) -> float:
    valid = (
        (validity_mask[:, channel_index] > 0)
        & np.isfinite(original[:, channel_index])
        & np.isfinite(reconstruction[:, channel_index])
    )
    value, _ = _weighted_rmse(
        original[:, channel_index],
        reconstruction[:, channel_index],
        valid,
        latitude_weights,
        channel_name=channel_name,
    )
    return value


def _statistics_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
