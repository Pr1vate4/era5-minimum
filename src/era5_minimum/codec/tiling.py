from __future__ import annotations

from collections.abc import Callable

import torch


TensorPredictor = Callable[[torch.Tensor], torch.Tensor]
TensorDecoder = Callable[[torch.Tensor], torch.Tensor]


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


def decode_latent_tiled(
    latent: torch.Tensor,
    *,
    output_size: tuple[int, int],
    tile_height: int,
    tile_width: int,
    halo: int,
    scale_factor: int,
    decoder: TensorDecoder,
) -> torch.Tensor:
    """Decode a spatial latent in tiles using output-grid tile dimensions."""

    values = torch.as_tensor(latent)
    if values.ndim != 4:
        raise ValueError(f"latent must have shape [batch, channel, height, width], got {tuple(values.shape)}")
    if scale_factor < 1:
        raise ValueError("scale_factor must be positive")
    if tile_height < 1 or tile_width < 1:
        raise ValueError("tile dimensions must be positive")
    if halo < 0:
        raise ValueError("halo must be non-negative")
    if tile_height % scale_factor or tile_width % scale_factor or halo % scale_factor:
        raise ValueError("tile dimensions and halo must be divisible by scale_factor")

    target_height, target_width = (int(output_size[0]), int(output_size[1]))
    if target_height < 1 or target_width < 1:
        raise ValueError("output_size must be positive")

    batch, _, latent_height, latent_width = values.shape
    if target_height > latent_height * scale_factor or target_width > latent_width * scale_factor:
        raise ValueError(
            f"output_size {(target_height, target_width)} exceeds latent decode bounds "
            f"{(latent_height * scale_factor, latent_width * scale_factor)}"
        )

    latent_tile_height = tile_height // scale_factor
    latent_tile_width = tile_width // scale_factor
    latent_halo = halo // scale_factor
    output: torch.Tensor | None = None

    for top in range(0, latent_height, latent_tile_height):
        bottom = min(top + latent_tile_height, latent_height)
        core_height = bottom - top
        h_indices = torch.arange(
            top - latent_halo,
            bottom + latent_halo,
            device=values.device,
        ).clamp(0, latent_height - 1)
        for left in range(0, latent_width, latent_tile_width):
            right = min(left + latent_tile_width, latent_width)
            core_width = right - left
            w_indices = torch.remainder(
                torch.arange(left - latent_halo, right + latent_halo, device=values.device),
                latent_width,
            )
            latent_tile = values.index_select(2, h_indices).index_select(3, w_indices)
            decoded_tile = decoder(latent_tile)
            expected_height = int(latent_tile.shape[-2]) * scale_factor
            expected_width = int(latent_tile.shape[-1]) * scale_factor
            if decoded_tile.ndim != 4 or decoded_tile.shape[0] != batch:
                raise ValueError("decoder must return [batch, channel, height, width]")
            if decoded_tile.shape[-2:] != (expected_height, expected_width):
                raise ValueError(
                    "decoder spatial scale mismatch: "
                    f"expected {(expected_height, expected_width)}, got {tuple(decoded_tile.shape[-2:])}"
                )
            if output is None:
                output = torch.empty(
                    (batch, int(decoded_tile.shape[1]), target_height, target_width),
                    dtype=decoded_tile.dtype,
                    device=decoded_tile.device,
                )

            output_top = top * scale_factor
            output_left = left * scale_factor
            write_height = min(core_height * scale_factor, target_height - output_top)
            write_width = min(core_width * scale_factor, target_width - output_left)
            if write_height <= 0 or write_width <= 0:
                continue
            crop_top = latent_halo * scale_factor
            crop_left = latent_halo * scale_factor
            output[
                :,
                :,
                output_top : output_top + write_height,
                output_left : output_left + write_width,
            ] = decoded_tile[
                :,
                :,
                crop_top : crop_top + write_height,
                crop_left : crop_left + write_width,
            ]

    if output is None:
        raise RuntimeError("tiled decoder produced no output")
    return output


def compute_tile_seam_error(
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tile_height: int,
    tile_width: int,
    boundary_width: int = 1,
    validity_mask: torch.Tensor | None = None,
    include_internal_boundaries: bool = True,
    include_longitude_wrap: bool = True,
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

    if include_internal_boundaries:
        for row in range(tile_height, height, tile_height):
            top = max(0, row - boundary_width)
            bottom = min(height, row + boundary_width)
            mask[top:bottom, :] = True
        for column in range(tile_width, width, tile_width):
            left = max(0, column - boundary_width)
            right = min(width, column + boundary_width)
            mask[:, left:right] = True
    if include_longitude_wrap:
        mask[:, :boundary_width] = True
        mask[:, max(0, width - boundary_width) :] = True
    elif include_internal_boundaries:
        mask[:, :boundary_width] = False
        mask[:, max(0, width - boundary_width) :] = False

    seam_mask = mask.reshape(1, 1, height, width).expand_as(ref)
    if validity_mask is not None:
        valid = torch.as_tensor(validity_mask, device=ref.device) > 0
        try:
            valid = torch.broadcast_to(valid, ref.shape)
        except RuntimeError as error:
            raise ValueError(
                f"validity_mask shape {tuple(valid.shape)} is not broadcastable to {tuple(ref.shape)}"
            ) from error
        seam_mask = seam_mask & valid

    if not torch.any(seam_mask):
        return 0.0

    error = (pred - ref) ** 2
    masked_error = error[seam_mask]
    return float(torch.sqrt(torch.mean(masked_error)).item())
