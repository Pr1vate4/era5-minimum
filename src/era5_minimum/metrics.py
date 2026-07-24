from __future__ import annotations

import numpy as np
import torch


def latitude_weights(latitudes: np.ndarray) -> torch.Tensor:
    """Return normalized cosine latitude weights with shape [1, 1, H, 1]."""
    values = np.cos(np.deg2rad(latitudes.astype(np.float64)))
    values = np.clip(values, 0.0, None)
    values = values / values.mean()
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(values)
    valid_mask = mask.to(values.device, dtype=values.dtype)
    valid_count = torch.sum(valid_mask)
    if float(valid_count.item()) <= 0.0:
        raise ValueError("mask does not select any valid values")
    return torch.sum(values * valid_mask) / valid_count


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return rmse(prediction, target, mask=mask)


def rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    return float(torch.sqrt(_masked_mean((prediction - target) ** 2, mask=mask)).item())


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    return mae(prediction, target, mask=mask)


def mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    return float(_masked_mean(torch.abs(prediction - target), mask=mask).item())


def latitude_weighted_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: np.ndarray,
    mask: torch.Tensor | None = None,
) -> float:
    weights = latitude_weights(latitudes).to(prediction.device)
    squared_error = (prediction - target) ** 2
    if mask is None:
        return float(torch.sqrt(torch.mean(squared_error * weights)).item())
    valid_mask = mask.to(prediction.device, dtype=weights.dtype)
    weighted_mask = weights * valid_mask
    valid_weight = torch.sum(weighted_mask)
    if float(valid_weight.item()) <= 0.0:
        raise ValueError("mask does not select any valid values")
    return float(torch.sqrt(torch.sum(squared_error * weighted_mask) / valid_weight).item())


def per_channel_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    channel_names: list[str],
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if prediction.shape[1] != len(channel_names):
        raise ValueError("channel_names length does not match tensor channels")
    values: dict[str, float] = {}
    for index, name in enumerate(channel_names):
        channel_mask = None if mask is None else mask[:, index : index + 1]
        values[name] = rmse(prediction[:, index : index + 1], target[:, index : index + 1], mask=channel_mask)
    return values
