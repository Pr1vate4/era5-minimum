"""Regression coverage for SST land masking in physical metrics."""
from __future__ import annotations

import numpy as np

from era5_minimum.codec.evaluation import _weighted_rmse


def test_weighted_rmse_ignores_nan_values_outside_sst_mask() -> None:
    original = np.array([[[1.0], [np.nan]]])
    reconstruction = np.array([[[2.0], [np.nan]]])
    valid = np.array([[[True], [False]]])
    result, count = _weighted_rmse(
        original,
        reconstruction,
        valid,
        np.array([1.0, 1.0]),
        channel_name="sst",
    )
    assert count == 1
    assert result == 1.0
