import numpy as np
import torch

from era5_minimum.metrics import latitude_weighted_rmse, rmse


def test_zero_error_metrics() -> None:
    tensor = torch.ones(2, 3, 8, 16)
    latitudes = np.linspace(90, -90, 8, dtype=np.float32)
    assert rmse(tensor, tensor) == 0.0
    assert latitude_weighted_rmse(tensor, tensor, latitudes) == 0.0
