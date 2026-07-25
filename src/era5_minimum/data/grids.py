import xarray as xr
import numpy as np

def get_target_grid_05() -> xr.Dataset:
    """Создает целевую сетку 0.5° (360x720, cell-centred)."""
    lat = xr.DataArray(np.linspace(-89.75, 89.75, 360), dims=["latitude"], name="latitude")
    lon = xr.DataArray(np.linspace(0.25, 359.75, 720), dims=["longitude"], name="longitude")
    return xr.Dataset({"lat": lat, "lon": lon})
