import numpy as np

from era5_minimum.data.synthetic import make_synthetic_era5


def test_synthetic_is_deterministic() -> None:
    first, first_lat = make_synthetic_era5(4, 16, 32, seed=7)
    second, second_lat = make_synthetic_era5(4, 16, 32, seed=7)
    assert first.shape == (4, 8, 16, 32)
    assert np.array_equal(first, second)
    assert np.array_equal(first_lat, second_lat)
