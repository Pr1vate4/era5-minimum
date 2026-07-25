"""Immutable input contract shared by ERA5 preparation and codec serving."""

from __future__ import annotations

import numpy as np

from .channel_spec import CHANNEL_NAMES


MODEL_GRID_SHAPE = (360, 720)
CANONICAL_FRAME_SHAPE = (1, len(CHANNEL_NAMES), *MODEL_GRID_SHAPE)
NPZ_DATA_KEY = "data"
NPZ_CHANNEL_ORDER_KEY = "channel_order"


class ModelInputContractError(ValueError):
    """Raised when an array cannot be consumed by the accepted model."""


def validate_model_input(
    values: np.ndarray,
    *,
    channel_order: tuple[str, ...] | None,
    allow_unbatched: bool = False,
) -> np.ndarray:
    """Return a view of a canonical physical ERA5 model input."""

    tensor = np.asarray(values)
    if allow_unbatched and tensor.shape == CANONICAL_FRAME_SHAPE[1:]:
        tensor = tensor[None, ...]
    if tensor.shape != CANONICAL_FRAME_SHAPE:
        raise ModelInputContractError(
            "data must have shape [1, 28, 360, 720]"
            + (" (or [28, 360, 720])" if allow_unbatched else "")
            + f", got {tuple(tensor.shape)}"
        )
    if tensor.dtype != np.float32:
        raise ModelInputContractError(
            f"data must use float32 physical units, got {tensor.dtype}"
        )
    if channel_order is not None and tuple(channel_order) != CHANNEL_NAMES:
        raise ModelInputContractError(
            "channel_order does not match the canonical 28-channel ERA5 contract"
        )
    return tensor
