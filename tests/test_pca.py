import numpy as np

from era5_minimum.models import PCAAutoencoder


def test_pca_roundtrip_and_ratio() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(size=(20, 8, 8, 8)).astype(np.float32)
    model = PCAAutoencoder(latent_dim=16)
    model.fit(train)
    latent = model.encode(train[:2])
    restored = model.decode(latent)
    assert latent.shape == (2, 16)
    assert restored.shape == train[:2].shape
    assert model.tensor_compression_ratio() == 32.0
