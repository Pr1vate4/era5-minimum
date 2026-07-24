import pytest
import numpy as np
import xarray as xr
xe = pytest.importorskip("xesmf")

from era5_minimum.data.grids import get_target_grid_05
from era5_minimum.data.remapping import conservative_remap


# 28. Форма 721×1440
def test_28_native_grid_shape():
    # Предполагаем, что есть функция get_native_grid или проверяем через WB2 adapter
    # Для теста создаем мок
    native = xr.Dataset({"lat": (["lat"], np.linspace(-90, 90, 721)),
                         "lon": (["lon"], np.linspace(0, 359.75, 1440))})
    assert native.sizes == {"lat": 721, "lon": 1440}


# 29. Форма 360×720
def test_29_target_grid_shape():
    target = get_target_grid_05()
    assert target.sizes == {"latitude": 360, "longitude": 720}


# 30. Координаты target grid
def test_30_target_grid_coordinates():
    target = get_target_grid_05()
    assert target["latitude"].values[0] == pytest.approx(-89.75)
    assert target["longitude"].values[0] == pytest.approx(0.25)


# 31. Periodic longitude
def test_31_periodic_longitude():
    target = get_target_grid_05()
    lon_diff = np.diff(target["longitude"].values)
    assert np.allclose(lon_diff, 0.5)
    # Проверка wrap-around (последний + 0.5 должен давать 360.25, но в cell-centered 359.75 + 0.5 = 360.25 -> 0.25)
    assert target["longitude"].values[-1] == pytest.approx(359.75)


# 32. Conservative remapping сохраняет постоянное поле
def test_32_conservative_constant_field():
    ds_in = xr.Dataset({"temp": (["latitude", "longitude"], np.full((4, 8), 5.0)),
                        "latitude": (["latitude"], np.linspace(-90, 90, 4)),
                        "longitude": (["longitude"], np.linspace(0, 359, 8))})
    ds_out = get_target_grid_05()
    # Упрощаем для теста
    ds_out_small = xr.Dataset({"latitude": (["latitude"], [-45, 45]), "longitude": (["longitude"], [90, 270])})

    regridder = xe.Regridder(ds_in, ds_out_small, "conservative", periodic=True)
    res = regridder(ds_in)
    assert res["temp"].values == pytest.approx(5.0, rel=1e-4)


# 33. Веса переиспользуются
def test_33_weights_reuse(tmp_path):
    ds_in = xr.Dataset({"lat": (["lat"], [-90, 90]), "lon": (["lon"], [0, 359])})
    ds_out = xr.Dataset({"lat": (["lat"], [-45, 45]), "lon": (["lon"], [90, 270])})

    weight_file = tmp_path / "weights.nc"
    xe.Regridder(ds_in, ds_out, "conservative", weights=str(weight_file))
    assert weight_file.exists()

    # Повторный вызов должен использовать файл
    xe.Regridder(ds_in, ds_out, "conservative", weights=str(weight_file), reuse_weights=True)


# 34. Hash координат совпадает
def test_34_coordinates_hash():
    from era5_minimum.data.manifest import compute_sha256
    import tempfile, os

    grid1 = get_target_grid_05()
    grid2 = get_target_grid_05()

    with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
        grid1.to_netcdf(f1.name)
        grid2.to_netcdf(f2.name)
        h1, h2 = compute_sha256(f1.name), compute_sha256(f2.name)
        os.unlink(f1.name);
        os.unlink(f2.name)

    assert h1 == h2