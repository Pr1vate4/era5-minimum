import torch

from era5_minimum.models import ConvAutoencoder


def test_roundtrip_shape_and_32x_ratio() -> None:
    model = ConvAutoencoder(in_channels=8, latent_channels=16)
    sample = torch.randn(2, 8, 32, 64)
    reconstruction = model(sample)
    assert reconstruction.shape == sample.shape
    assert model.tensor_compression_ratio(sample[:1]) == 32.0


def test_64x_ratio() -> None:
    model = ConvAutoencoder(in_channels=8, latent_channels=8)
    sample = torch.randn(1, 8, 32, 64)
    assert model.tensor_compression_ratio(sample) == 64.0


def test_roundtrip_shape_for_non_multiple_of_eight_grid() -> None:
    model = ConvAutoencoder(in_channels=8, latent_channels=8)
    sample = torch.randn(1, 8, 33, 65)
    reconstruction = model(sample)
    assert reconstruction.shape == sample.shape
