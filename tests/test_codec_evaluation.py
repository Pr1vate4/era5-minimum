from __future__ import annotations

import json
from math import log10, sqrt

import numpy as np
import pytest

from era5_minimum.codec.evaluation import (
    EVALUATOR_VERSION,
    evaluate_reconstruction,
    fit_train_channel_ranges,
)


CHANNELS = (
    "t2m",
    "mslp",
    "u10",
    "v10",
    "tp6h",
    "sst",
    "tcwv",
    "tcc",
    "T1000",
    "T925",
    "T850",
    "T700",
    "U1000",
    "U925",
    "U850",
    "U700",
    "V1000",
    "V925",
    "V850",
    "V700",
    "Z1000",
    "Z925",
    "Z850",
    "Z700",
    "Q1000",
    "Q925",
    "Q850",
    "Q700",
)


def _stats(channels: int = 28) -> tuple[np.ndarray, np.ndarray]:
    return np.ones(channels, dtype=np.float64), np.full(channels, 10.0, dtype=np.float64)


def test_evaluator_uses_latitude_weighted_physical_rmse_and_equal_channel_groups() -> None:
    original = np.zeros((1, 28, 2, 1), dtype=np.float32)
    reconstruction = np.zeros_like(original)
    reconstruction[:, 0, :, 0] = (1.0, 3.0)
    reconstruction[:, 1, 0, 0] = 1.0
    reconstruction[:, 1, 1, 0] = 100.0
    reconstruction[:, 2:8] = 1.0
    reconstruction[:, 8:] = 2.0
    validity_mask = np.ones_like(original, dtype=np.float32)
    validity_mask[:, 1, 1, 0] = 0.0

    result = evaluate_reconstruction(
        original,
        reconstruction,
        latitudes=np.array([0.0, 60.0]),
        channel_order=CHANNELS,
        train_std=np.ones(28),
        train_ranges=np.full(28, 10.0),
        validity_mask=validity_mask,
    )

    expected_channel_zero = sqrt((1.0 + 0.5 * 9.0) / 1.5)
    assert result["evaluator_version"] == EVALUATOR_VERSION
    assert result["per_channel"][0]["channel"] == "t2m"
    assert result["per_channel"][0]["rmse_physical"] == pytest.approx(expected_channel_zero)
    assert result["per_channel"][0]["nrmse"] == pytest.approx(expected_channel_zero)
    assert result["per_channel"][1]["rmse_physical"] == pytest.approx(1.0)
    assert result["groups"]["surface"]["channel_count"] == 8
    assert result["groups"]["pressure"]["channel_count"] == 20
    assert result["groups"]["surface"]["nrmse"] == pytest.approx(
        (expected_channel_zero + 7.0) / 8.0
    )
    assert result["groups"]["pressure"]["nrmse"] == pytest.approx(2.0)
    assert result["groups"]["overall"]["nrmse"] == pytest.approx(
        0.5 * result["groups"]["surface"]["nrmse"] + 0.5 * 2.0
    )


def test_train_ranges_ignore_invalid_values_and_psnr_uses_train_range() -> None:
    train = np.zeros((3, 28, 1, 1), dtype=np.float32)
    train[:, 0, 0, 0] = (1.0, 9.0, 100.0)
    train_mask = np.ones_like(train, dtype=np.float32)
    train_mask[2, 0, 0, 0] = 0.0

    ranges = fit_train_channel_ranges(train, validity_mask=train_mask)

    assert ranges.shape == (28,)
    assert ranges[0] == pytest.approx(8.0)

    original = np.zeros((1, 28, 1, 1), dtype=np.float32)
    reconstruction = original.copy()
    reconstruction[0, 0, 0, 0] = 2.0
    result = evaluate_reconstruction(
        original,
        reconstruction,
        latitudes=np.array([0.0]),
        channel_order=CHANNELS,
        train_std=np.ones(28),
        train_ranges=ranges,
        validity_mask=np.ones_like(original, dtype=np.float32),
    )

    assert result["per_channel"][0]["psnr_db"] == pytest.approx(20.0 * log10(4.0))
    assert result["train_statistics"]["train_only"] is True
    assert result["train_statistics"]["ranges"][0] == pytest.approx(8.0)


def test_psnr_is_json_safe_for_perfect_and_zero_range_channels() -> None:
    original = np.zeros((1, 28, 1, 1), dtype=np.float32)
    reconstruction = original.copy()
    train_std = np.ones(28)
    train_ranges = np.ones(28)
    train_ranges[1] = 0.0

    result = evaluate_reconstruction(
        original,
        reconstruction,
        latitudes=np.array([0.0]),
        channel_order=CHANNELS,
        train_std=train_std,
        train_ranges=train_ranges,
        validity_mask=np.ones_like(original, dtype=np.float32),
    )

    assert result["per_channel"][0]["psnr_db"] is None
    assert result["per_channel"][0]["psnr_status"] == "perfect_reconstruction"
    assert result["per_channel"][1]["psnr_db"] is None
    assert result["per_channel"][1]["psnr_status"] == "zero_train_range"
    json.dumps(result, allow_nan=False)


def test_physical_diagnostics_preserve_units_and_derive_wind_speed() -> None:
    original = np.zeros((1, 28, 1, 1), dtype=np.float32)
    reconstruction = original.copy()
    reconstruction[0, 1, 0, 0] = 100.0
    reconstruction[0, 2, 0, 0] = 3.0
    reconstruction[0, 3, 0, 0] = 4.0
    reconstruction[0, 4, 0, 0] = 0.002
    reconstruction[0, 20, 0, 0] = 9.80665

    result = evaluate_reconstruction(
        original,
        reconstruction,
        latitudes=np.array([0.0]),
        channel_order=CHANNELS,
        train_std=np.ones(28),
        train_ranges=np.full(28, 10.0),
        validity_mask=np.ones_like(original, dtype=np.float32),
    )
    diagnostics = result["physical_diagnostics"]

    assert diagnostics["mslp"]["rmse_pa"] == pytest.approx(100.0)
    assert diagnostics["mslp"]["rmse_hpa"] == pytest.approx(1.0)
    assert diagnostics["tp6h"]["rmse_m"] == pytest.approx(0.002)
    assert diagnostics["tp6h"]["rmse_mm_per_6h"] == pytest.approx(2.0)
    assert diagnostics["geopotential"]["Z1000"]["rmse_m2_s2"] == pytest.approx(9.80665)
    assert diagnostics["geopotential"]["Z1000"]["rmse_height_m"] == pytest.approx(1.0)
    assert diagnostics["u10"]["rmse_m_s"] == pytest.approx(3.0)
    assert diagnostics["v10"]["rmse_m_s"] == pytest.approx(4.0)
    assert diagnostics["wind_speed"]["rmse_m_s"] == pytest.approx(5.0)
