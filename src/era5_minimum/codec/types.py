from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CodecConfig:
    """Versioned codec configuration."""

    version: str
    channel_order: tuple[str, ...]
    grid: str
    quantization_step: float
    seed: int
    git_commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "channel_order": list(self.channel_order),
            "grid": self.grid,
            "quantization_step": float(self.quantization_step),
            "seed": int(self.seed),
            "git_commit": self.git_commit,
        }


@dataclass(frozen=True)
class CodecResult:
    """Summary for one codec run."""

    bitstream_path: Path
    metadata_path: Path
    tensor_ratio: float
    serialized_ratio: float
    roundtrip_ok: bool
    metadata: dict[str, Any]
    decoded_latent: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bitstream_path": str(self.bitstream_path),
            "metadata_path": str(self.metadata_path),
            "tensor_ratio": float(self.tensor_ratio),
            "serialized_ratio": float(self.serialized_ratio),
            "roundtrip_ok": self.roundtrip_ok,
            "metadata": self.metadata,
            "decoded_latent_shape": list(self.decoded_latent.shape) if self.decoded_latent is not None else None,
        }
