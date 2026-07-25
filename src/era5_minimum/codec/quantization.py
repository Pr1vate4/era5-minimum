from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarQuantizer:
    """Deterministic scalar quantizer for fixed-step symbolization."""

    step: float
    zero_point: int = 0

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")

    def quantize(self, values: np.ndarray) -> np.ndarray:
        scaled = np.asarray(values, dtype=np.float32) / self.step
        return np.rint(scaled).astype(np.int32) + self.zero_point

    def dequantize(self, symbols: np.ndarray) -> np.ndarray:
        shifted = np.asarray(symbols, dtype=np.int32).astype(np.float32) - self.zero_point
        return shifted * self.step
