from __future__ import annotations

import pytest
import torch

from era5_minimum.codec import decode_latent_tiled
from era5_minimum.codec.tiling import compute_tile_seam_error, run_tiled_inference
from era5_minimum.models import ConvAutoencoder


def _wrapped_local_operator(tile: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    padded_h = torch.nn.functional.pad(tile, (0, 0, 1, 1), mode="replicate")
    padded = torch.cat([padded_h[..., -1:], padded_h, padded_h[..., :1]], dim=-1)
    return torch.nn.functional.conv2d(padded, kernel)


def test_tiled_inference_preserves_shape_for_360x720_grid() -> None:
    sample = torch.arange(2 * 3 * 360 * 720, dtype=torch.float32).reshape(2, 3, 360, 720)

    restored = run_tiled_inference(
        sample,
        tile_height=128,
        tile_width=128,
        halo=0,
        predictor=lambda tile: tile,
    )

    assert restored.shape == sample.shape
    assert torch.equal(restored, sample)


def test_tiled_inference_preserves_shape_for_721x1440_grid() -> None:
    sample = torch.arange(1 * 1 * 721 * 1440, dtype=torch.float32).reshape(1, 1, 721, 1440)

    restored = run_tiled_inference(
        sample,
        tile_height=128,
        tile_width=128,
        halo=0,
        predictor=lambda tile: tile,
    )

    assert restored.shape == sample.shape
    assert torch.equal(restored, sample)


def test_tiled_inference_matches_full_frame_with_longitude_wrap() -> None:
    sample = torch.arange(1 * 1 * 9 * 11, dtype=torch.float32).reshape(1, 1, 9, 11)
    kernel = torch.tensor([[[[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]]])

    def local_operator(tile: torch.Tensor) -> torch.Tensor:
        return _wrapped_local_operator(tile, kernel)

    full = local_operator(sample)
    tiled = run_tiled_inference(
        sample,
        tile_height=4,
        tile_width=5,
        halo=1,
        predictor=local_operator,
    )

    assert torch.allclose(tiled, full)


def test_tiled_inference_reports_zero_seam_error_for_identity() -> None:
    sample = torch.randn(1, 2, 17, 19)
    tiled = run_tiled_inference(
        sample,
        tile_height=8,
        tile_width=8,
        halo=0,
        predictor=lambda tile: tile,
    )

    seam = compute_tile_seam_error(reference=sample, candidate=tiled, tile_height=8, tile_width=8)

    assert seam == 0.0


def test_tiled_inference_matches_full_frame_for_local_conv_with_halo() -> None:
    torch.manual_seed(7)
    sample = torch.randn(1, 3, 15, 17)
    kernel = torch.randn(3, 3, 3, 3)

    def predictor(tile: torch.Tensor) -> torch.Tensor:
        return _wrapped_local_operator(tile, kernel)

    full = predictor(sample)
    tiled = run_tiled_inference(
        sample,
        tile_height=6,
        tile_width=7,
        halo=1,
        predictor=predictor,
    )

    assert torch.allclose(tiled, full, atol=1e-6)


def test_tiled_latent_decode_matches_full_decode_for_odd_output_shape() -> None:
    latent = torch.arange(1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)

    def decoder(tile: torch.Tensor) -> torch.Tensor:
        return tile.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)

    full = decoder(latent)[..., :9, :17]
    tiled = decode_latent_tiled(
        latent,
        output_size=(9, 17),
        tile_height=8,
        tile_width=8,
        halo=8,
        scale_factor=8,
        decoder=decoder,
    )

    assert tiled.shape == (1, 2, 9, 17)
    assert torch.equal(tiled, full)


@pytest.mark.parametrize(
    ("output_size", "latent_size"),
    [
        ((360, 720), (45, 90)),
        ((721, 1440), (91, 180)),
    ],
)
def test_tiled_latent_decode_covers_supported_global_grids(
    output_size: tuple[int, int],
    latent_size: tuple[int, int],
) -> None:
    latent = torch.ones(1, 1, *latent_size)

    decoded = decode_latent_tiled(
        latent,
        output_size=output_size,
        tile_height=128,
        tile_width=128,
        halo=16,
        scale_factor=8,
        decoder=lambda tile: tile.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1),
    )

    assert decoded.shape == (1, 1, *output_size)
    assert torch.equal(decoded, torch.ones_like(decoded))


def test_tiled_latent_decode_matches_convtranspose_away_from_global_boundaries() -> None:
    torch.manual_seed(5)
    model = ConvAutoencoder(in_channels=3, latent_channels=2).eval()
    latent = torch.randn(1, 2, 6, 8)

    with torch.no_grad():
        full = model.decode(latent)
        tiled = decode_latent_tiled(
            latent,
            output_size=(48, 64),
            tile_height=16,
            tile_width=16,
            halo=8,
            scale_factor=8,
            decoder=model.decode,
        )

    assert torch.equal(tiled[..., 8:-8, 8:-8], full[..., 8:-8, 8:-8])


def test_tile_seam_error_includes_longitude_wrap_boundary() -> None:
    reference = torch.zeros(1, 1, 8, 16)
    candidate = reference.clone()
    candidate[..., 0] = 1.0
    candidate[..., -1] = 1.0

    seam = compute_tile_seam_error(
        reference=reference,
        candidate=candidate,
        tile_height=4,
        tile_width=8,
        boundary_width=1,
    )

    assert seam > 0.0


def test_tile_seam_error_separates_internal_and_longitude_wrap_boundaries() -> None:
    reference = torch.zeros(1, 1, 8, 16)
    candidate = reference.clone()
    candidate[..., 0] = 1.0
    candidate[..., -1] = 1.0

    internal = compute_tile_seam_error(
        reference=reference,
        candidate=candidate,
        tile_height=4,
        tile_width=8,
        boundary_width=1,
        include_longitude_wrap=False,
    )
    longitude_wrap = compute_tile_seam_error(
        reference=reference,
        candidate=candidate,
        tile_height=4,
        tile_width=8,
        boundary_width=1,
        include_internal_boundaries=False,
    )

    assert internal == 0.0
    assert longitude_wrap > 0.0


def test_tile_seam_error_excludes_invalid_values() -> None:
    reference = torch.zeros(1, 1, 8, 16)
    candidate = reference.clone()
    candidate[..., 0] = 1.0
    validity_mask = torch.ones_like(reference)
    validity_mask[..., 0] = 0.0

    seam = compute_tile_seam_error(
        reference=reference,
        candidate=candidate,
        tile_height=4,
        tile_width=8,
        boundary_width=1,
        validity_mask=validity_mask,
    )

    assert seam == 0.0
