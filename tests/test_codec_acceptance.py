from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from era5_minimum.codec.acceptance import (
    AcceptanceError,
    WeatherCodec,
    compression_metrics,
    evaluate_physical_reconstruction,
)
from era5_minimum.data.channel_spec import CHANNEL_NAMES
from era5_minimum.models import ConvAutoencoder


def _checkpoint(path: Path) -> Path:
    """Write a tiny deterministic test fixture, never a scientific model."""

    torch.manual_seed(3)
    model = ConvAutoencoder(in_channels=28, latent_channels=2)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"in_channels": 28, "latent_channels": 2},
            "codec_config": {"version": "test-1", "quantization_step": 0.25},
            "normalization": {
                "channel_order": list(CHANNEL_NAMES),
                "mean": [0.0] * 28,
                "std": [1.0] * 28,
                "source_manifest_sha256": "train-fixture-only",
                "train_only": True,
            },
            "channel_order": list(CHANNEL_NAMES),
        },
        path,
    )
    return path


def test_weather_codec_roundtrip_is_independently_decodable(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "fixture.ckpt")
    codec = WeatherCodec.load(checkpoint)
    physical = np.linspace(-1.0, 1.0, 28 * 8 * 8, dtype=np.float32).reshape(1, 28, 8, 8)
    ocean = np.ones((8, 8), dtype=bool)
    model_input, _, _ = codec.preprocess(physical, ocean_mask=ocean)

    result = codec.reconstruct(model_input)
    independent = WeatherCodec.load(checkpoint)
    decoded, symbols, _, header = independent.decompress(result.bitstream)

    assert isinstance(result.bitstream, bytes)
    assert result.bitstream[:4] == b"E5AC"
    assert header["original_shape"] == [1, 28, 8, 8]
    np.testing.assert_array_equal(symbols, result.symbols)
    np.testing.assert_array_equal(decoded, result.reconstruction_normalized)
    assert result.latent_shape[1] == 2


def test_checkpoint_rejects_noncanonical_channel_order(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "wrong-order.ckpt")
    payload = torch.load(checkpoint, weights_only=False)
    payload["channel_order"] = list(reversed(CHANNEL_NAMES))
    torch.save(payload, checkpoint)
    with pytest.raises(AcceptanceError, match="channel_order"):
        WeatherCodec.load(checkpoint)


def test_preprocess_postprocess_restores_sst_land_nan(tmp_path: Path) -> None:
    codec = WeatherCodec.load(_checkpoint(tmp_path / "fixture.ckpt"))
    physical = np.ones((1, 28, 4, 4), dtype=np.float32)
    physical[:, CHANNEL_NAMES.index("sst"), 0, 0] = np.nan
    ocean = np.ones((4, 4), dtype=bool)
    ocean[0, 0] = False
    normalized, _, filled = codec.preprocess(physical, ocean_mask=ocean)
    restored = codec.postprocess(normalized, ocean_mask=ocean)
    assert filled >= 1
    assert np.isnan(restored[:, CHANNEL_NAMES.index("sst"), 0, 0]).all()
    np.testing.assert_allclose(restored[:, CHANNEL_NAMES.index("t2m")], physical[:, CHANNEL_NAMES.index("t2m")])


def test_compression_metrics_charge_header_and_side_information() -> None:
    header = {"format": "x"}
    bitstream = b"header-and-payload"
    result = compression_metrics(original_shape=(1, 28, 2, 2), bitstream=bitstream, header=header)
    assert result["total_bitstream_bytes"] == len(bitstream)
    assert result["header_bytes"] > 8
    assert result["side_information_bytes"] == result["header_bytes"]
    assert result["compression_ratio"] == pytest.approx((32 * 28 * 4) / (8 * len(bitstream)))


def test_physical_metrics_are_latitude_weighted_and_sst_is_ocean_only() -> None:
    original = np.zeros((1, 28, 2, 1), dtype=np.float32)
    reconstructed = original.copy()
    reconstructed[:, 0, 0, 0] = 2.0
    reconstructed[:, CHANNEL_NAMES.index("sst"), :, 0] = [5.0, 100.0]
    ocean = np.array([[True], [False]])
    metrics = evaluate_physical_reconstruction(
        original, reconstructed, latitudes=np.array([0.0, 60.0]), ocean_mask=ocean, train_std=None
    )
    assert metrics["per_channel"]["t2m"]["latitude_weighted_rmse"] == pytest.approx(np.sqrt(4.0 / 1.5))
    assert metrics["per_channel"]["sst"]["latitude_weighted_rmse"] == pytest.approx(5.0)
    assert metrics["per_channel"]["sst"]["valid_point_count"] == 1
    assert metrics["per_channel"]["t2m"]["nrmse"] is None
    assert metrics["per_channel"]["t2m"]["nrmse_status"] == "not_available_without_train_std"
