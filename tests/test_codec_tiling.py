from __future__ import annotations

import torch

from era5_minimum.codec.tiling import compute_tile_seam_error, run_tiled_inference


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
