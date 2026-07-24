from __future__ import annotations

from collections.abc import Callable

import torch


TensorPredictor = Callable[[torch.Tensor], torch.Tensor]


def run_tiled_inference(
    tensor: torch.Tensor,
    *,
    tile_height: int,
    tile_width: int,
    halo: int,
    predictor: TensorPredictor,
) -> torch.Tensor:
    """Run tiled inference with latitude clamping and longitude wrap-around."""

    values = torch.as_tensor(tensor)
    if values.ndim != 4:
        raise ValueError(f"tensor must have shape [batch, channel, height, width], got {tuple(values.shape)}")
    if tile_height < 1 or tile_width < 1:
        raise ValueError("tile dimensions must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")

    batch, channels, height, width = values.shape
    output = torch.empty_like(values)

    for top in range(0, height, tile_height):
        bottom = min(top + tile_height, height)
        core_height = bottom - top
        h_indices = torch.arange(top - halo, bottom + halo, device=values.device).clamp(0, height - 1)
        for left in range(0, width, tile_width):
            right = min(left + tile_width, width)
            core_width = right - left
            w_indices = torch.remainder(
                torch.arange(left - halo, right + halo, device=values.device),
                width,
            )
            tile = values.index_select(2, h_indices).index_select(3, w_indices)
            predicted = predictor(tile)
            if predicted.shape != tile.shape:
                raise ValueError(
                    f"predictor must preserve tile shape, got {tuple(predicted.shape)} for input {tuple(tile.shape)}"
                )
            output[:, :, top:bottom, left:right] = predicted[
                :,
                :,
                halo : halo + core_height,
                halo : halo + core_width,
            ]

    assert output.shape == (batch, channels, height, width)
    return output


def compute_tile_seam_error(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tile_height: int,
    tile_width: int,
    boundary_width: int = 1,
) -> float:
    """Compute RMSE on internal tile boundaries."""

    ref = torch.as_tensor(reference)
    pred = torch.as_tensor(candidate)
    if ref.shape != pred.shape:
        raise ValueError(f"reference.shape {tuple(ref.shape)} != candidate.shape {tuple(pred.shape)}")
    if ref.ndim != 4:
        raise ValueError(f"reference must have shape [batch, channel, height, width], got {tuple(ref.shape)}")
    if tile_height < 1 or tile_width < 1 or boundary_width < 1:
        raise ValueError("tile dimensions and boundary_width must be positive")

    _, _, height, width = ref.shape
    mask = torch.zeros((height, width), dtype=torch.bool, device=ref.device)

    for row in range(tile_height, height, tile_height):
        top = max(0, row - boundary_width)
        bottom = min(height, row + boundary_width)
        mask[top:bottom, :] = True
    for column in range(tile_width, width, tile_width):
        left = max(0, column - boundary_width)
        right = min(width, column + boundary_width)
        mask[:, left:right] = True

    if not torch.any(mask):
        return 0.0

    error = (pred - ref) ** 2
    masked_error = error[:, :, mask]
    return float(torch.sqrt(torch.mean(masked_error)).item())
