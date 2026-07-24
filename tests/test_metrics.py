import numpy as np
import pytest
import torch

from era5_minimum.metrics import latitude_weighted_rmse, masked_rmse, per_channel_rmse, rmse


def test_zero_error_metrics() -> None:
    tensor = torch.ones(2, 3, 8, 16)
    latitudes = np.linspace(90, -90, 8, dtype=np.float32)
    assert rmse(tensor, tensor) == 0.0
    assert latitude_weighted_rmse(tensor, tensor, latitudes) == 0.0


def test_masked_rmse_ignores_invalid_points() -> None:
    prediction = torch.tensor([[[[3.0, 1.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 1.0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

    assert masked_rmse(prediction, target, mask) == 0.0


def test_per_channel_rmse_ignores_invalid_points() -> None:
    prediction = torch.tensor([[[[3.0, 1.0]], [[5.0, 2.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 1.0]], [[1.0, 2.0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[0.0, 1.0]], [[1.0, 1.0]]]], dtype=torch.float32)

    values = per_channel_rmse(prediction, target, ["a", "b"], mask=mask)

    assert values["a"] == 0.0
    assert values["b"] == pytest.approx(2.8284271247461903)
