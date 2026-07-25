"""Numerical reconstruction metrics and Prometheus metric submodules.

The package name intentionally matches the historical ``metrics.py`` module.
Keeping the numerical public API here prevents Python from resolving the newer
package and hiding the functions used by experiments and the codec runtime.
"""

from __future__ import annotations

import numpy as np
import torch


def latitude_weights(latitudes: np.ndarray) -> torch.Tensor:
    """Return normalized cosine latitude weights with shape ``[1, 1, H, 1]``."""

    values = np.clip(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0, None)
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


def rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Return the (optionally masked) root mean square error."""

    return float(torch.sqrt(_masked_mean((prediction - target) ** 2, mask=mask)).item())


def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Return RMSE on the supplied valid-value mask."""

    return rmse(prediction, target, mask=mask)


def mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Return the (optionally masked) mean absolute error."""

    return float(_masked_mean(torch.abs(prediction - target), mask=mask).item())


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Return MAE on the supplied valid-value mask."""

    return mae(prediction, target, mask=mask)


def latitude_weighted_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latitudes: np.ndarray,
    mask: torch.Tensor | None = None,
) -> float:
    """Return cosine-latitude weighted RMSE, optionally over valid points."""

    weights = latitude_weights(latitudes).to(prediction.device)
    squared_error = (prediction - target) ** 2
    if mask is None:
        return float(torch.sqrt(torch.mean(squared_error * weights)).item())
    weighted_mask = weights * mask.to(prediction.device, dtype=weights.dtype)
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
    """Return an RMSE mapping without changing channel ordering."""

    if prediction.shape[1] != len(channel_names):
        raise ValueError("channel_names length does not match tensor channels")
    return {
        name: rmse(
            prediction[:, index : index + 1],
            target[:, index : index + 1],
            mask=None if mask is None else mask[:, index : index + 1],
        )
        for index, name in enumerate(channel_names)
    }


__all__ = [
    "latitude_weights",
    "latitude_weighted_rmse",
    "mae",
    "masked_mae",
    "masked_rmse",
    "per_channel_rmse",
    "rmse",
]
