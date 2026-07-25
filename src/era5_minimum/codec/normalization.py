from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NormalizationSpec:
    """Train-only per-channel normalization provenance."""

    channel_order: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    source_manifest_sha256: str
    train_only: bool = True

    def __post_init__(self) -> None:
        channel_order = tuple(self.channel_order)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if len(channel_order) == 0:
            raise ValueError("channel_order must not be empty")
        if mean.shape != std.shape:
            raise ValueError(f"mean.shape {mean.shape} != std.shape {std.shape}")
        if mean.ndim != 1:
            raise ValueError("mean and std must be one-dimensional")
        if mean.shape[0] != len(channel_order):
            raise ValueError("channel_order length must match mean/std length")
        if np.any(std <= 0):
            raise ValueError("std must be strictly positive")
        object.__setattr__(self, "channel_order", channel_order)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "train_only", bool(self.train_only))
        if not self.train_only:
            raise ValueError("normalization stats must be train-only")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_order": list(self.channel_order),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "source_manifest_sha256": self.source_manifest_sha256,
            "train_only": self.train_only,
        }


def normalize_physical_tensor(
    values: np.ndarray,
    *,
    spec: NormalizationSpec,
    ocean_mask: np.ndarray | None = None,
    sst_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize physical fields and replace invalid model inputs with zero."""

    tensor = np.asarray(values, dtype=np.float32)
    if tensor.ndim != 4 or tensor.shape[1] != len(spec.channel_order):
        raise ValueError(
            f"values must have shape [batch, {len(spec.channel_order)}, height, width], got {tensor.shape}"
        )
    mean = spec.mean.reshape(1, -1, 1, 1)
    std = spec.std.reshape(1, -1, 1, 1)
    normalized = (tensor - mean) / std
    validity_mask = np.isfinite(normalized)

    if ocean_mask is not None:
        if sst_index is None:
            raise ValueError("sst_index is required when ocean_mask is provided")
        mask = np.asarray(ocean_mask) > 0
        if mask.shape != tensor.shape[-2:]:
            raise ValueError(f"ocean_mask shape {mask.shape} != spatial shape {tensor.shape[-2:]}")
        validity_mask[:, int(sst_index)] &= mask[None, :, :]

    normalized = np.where(validity_mask, normalized, 0.0).astype(np.float32)
    return normalized, validity_mask.astype(np.float32), int(np.size(validity_mask) - np.count_nonzero(validity_mask))


def denormalize_reconstruction(
    values: np.ndarray,
    *,
    spec: NormalizationSpec,
    ocean_mask: np.ndarray | None = None,
    sst_index: int | None = None,
) -> np.ndarray:
    """Convert normalized reconstructions to physical units."""

    tensor = np.asarray(values, dtype=np.float32)
    if tensor.ndim != 4 or tensor.shape[1] != len(spec.channel_order):
        raise ValueError(
            f"values must have shape [batch, {len(spec.channel_order)}, height, width], got {tensor.shape}"
        )
    physical = tensor * spec.std.reshape(1, -1, 1, 1) + spec.mean.reshape(1, -1, 1, 1)
    if ocean_mask is not None:
        if sst_index is None:
            raise ValueError("sst_index is required when ocean_mask is provided")
        mask = np.asarray(ocean_mask) > 0
        if mask.shape != tensor.shape[-2:]:
            raise ValueError(f"ocean_mask shape {mask.shape} != spatial shape {tensor.shape[-2:]}")
        physical[:, int(sst_index)] = np.where(mask[None, :, :], physical[:, int(sst_index)], 0.0)
    return physical.astype(np.float32)
