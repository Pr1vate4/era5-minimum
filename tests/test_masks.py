import pytest
import numpy as np
import xarray as xr
from era5_minimum.data.masks import apply_masked_sst_remap


@pytest.fixture
def mock_regridder():
    class MockRegridder:
        def __call__(self, data):
            # Имитация ремаппинга: просто возвращаем данные (для теста логики маски)
            return data

    return MockRegridder()


# 23. Ocean mask
def test_23_ocean_mask():
    mask = xr.DataArray(np.array([[1.0, 0.0], [0.5, 0.2]]), dims=["lat", "lon"])
    assert mask.min() >= 0.0 and mask.max() <= 1.0


# 24. SST NaN над сушей
def test_24_sst_nan_over_land(mock_regridder):
    ds_in = xr.Dataset({"sea_surface_temperature": (["lat", "lon"], np.full((2, 2), 300.0))})
    ocean_frac = xr.DataArray(np.array([[1.0, 0.0], [1.0, 1.0]]), dims=["lat", "lon"])
    ds_out_grid = xr.Dataset({"lat": (["lat"], [0, 1]), "lon": (["lon"], [0, 1])})

    res = apply_masked_sst_remap(ds_in, ds_out_grid, ocean_frac, mock_regridder)
    assert np.isnan(res.values[0, 1])  # Суша (0.0) -> NaN


# 25. SST valid над океаном
def test_25_sst_valid_over_ocean(mock_regridder):
    ds_in = xr.Dataset({"sea_surface_temperature": (["lat", "lon"], np.full((2, 2), 300.0))})
    ocean_frac = xr.DataArray(np.ones((2, 2)), dims=["lat", "lon"])
    ds_out_grid = xr.Dataset({"lat": (["lat"], [0, 1]), "lon": (["lon"], [0, 1])})

    res = apply_masked_sst_remap(ds_in, ds_out_grid, ocean_frac, mock_regridder)
    assert not np.isnan(res.values).any()
    assert np.allclose(res.values, 300.0)


# 26. Ремаппинг SST с маской
def test_26_masked_sst_remapping(mock_regridder):
    ds_in = xr.Dataset({"sea_surface_temperature": (["lat", "lon"], np.full((2, 2), 300.0))})
    ocean_frac = xr.DataArray(np.array([[0.8, 0.2], [0.5, 0.1]]), dims=["lat", "lon"])
    ds_out_grid = xr.Dataset({"lat": (["lat"], [0, 1]), "lon": (["lon"], [0, 1])})

    res = apply_masked_sst_remap(ds_in, ds_out_grid, ocean_frac, mock_regridder)
    # Все значения > 0.5 (порог) должны быть валидными
    assert not np.isnan(res.values[0, 0])


# 27. Полностью сухая target cell
def test_27_fully_dry_target_cell(mock_regridder):
    ds_in = xr.Dataset({"sea_surface_temperature": (["lat", "lon"], np.full((2, 2), 300.0))})
    ocean_frac = xr.DataArray(np.zeros((2, 2)), dims=["lat", "lon"])  # Полностью суша
    ds_out_grid = xr.Dataset({"lat": (["lat"], [0, 1]), "lon": (["lon"], [0, 1])})

    res = apply_masked_sst_remap(ds_in, ds_out_grid, ocean_frac, mock_regridder)
    assert np.isnan(res.values).all()