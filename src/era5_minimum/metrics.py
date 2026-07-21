from __future__ import annotations

import numpy as np
import torch


def latitude_weights(latitudes: np.ndarray) -> torch.Tensor:
    """Return normalized cosine latitude weights with shape [1, 1, H, 1]."""
    values = np.cos(np.deg2rad(latitudes.astype(np.float64)))
    values = np.clip(values, 0.0, None)
    values = values / values.mean()
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((prediction - target) ** 2)).item())


def mae(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(prediction - target)).item())


def latitude_weighted_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: np.ndarray,
) -> float:
    weights = latitude_weights(latitudes).to(prediction.device)
    squared_error = (prediction - target) ** 2
    return float(torch.sqrt(torch.mean(squared_error * weights)).item())


def per_channel_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_names: list[str],
) -> dict[str, float]:
    if prediction.shape[1] != len(channel_names):
        raise ValueError("channel_names length does not match tensor channels")
    values: dict[str, float] = {}
    for index, name in enumerate(channel_names):
        values[name] = rmse(prediction[:, index : index + 1], target[:, index : index + 1])
    return values
