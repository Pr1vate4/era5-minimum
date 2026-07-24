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
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if len(self.channel_order) == 0:
            raise ValueError("channel_order must not be empty")
        if mean.shape != std.shape:
            raise ValueError(f"mean.shape {mean.shape} != std.shape {std.shape}")
        if mean.ndim != 1:
            raise ValueError("mean and std must be one-dimensional")
        if mean.shape[0] != len(self.channel_order):
            raise ValueError("channel_order length must match mean/std length")
        if np.any(std <= 0):
            raise ValueError("std must be strictly positive")
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
