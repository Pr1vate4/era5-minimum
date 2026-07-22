from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


VARIABLE_ALIASES = {
    "t2m": ("t2m", "2m_temperature"),
    "mslp": ("msl", "mslp", "mean_sea_level_pressure"),
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "tp6h": ("tp6h",),
    "sst": ("sst", "sea_surface_temperature"),
    "tcwv": ("tcwv", "total_column_water_vapour"),
    "tcc": ("tcc", "total_cloud_cover"),
}


def load_netcdf_tensor(path: str | Path, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load a prepared NetCDF file into [time, channel, latitude, longitude].

    The input must already contain `tp6h`; raw ERA5 precipitation accumulation
    semantics should be handled in a dedicated preprocessing step.
    """
    dataset = xr.open_dataset(path)
    lat_name = "latitude" if "latitude" in dataset.coords else "lat"
    lon_name = "longitude" if "longitude" in dataset.coords else "lon"
    time_name = "time" if "time" in dataset.coords else "valid_time"

    arrays = []
    for channel in channels:
        aliases = VARIABLE_ALIASES.get(channel, (channel,))
        variable = next((name for name in aliases if name in dataset.data_vars), None)
        if variable is None:
            raise KeyError(f"Missing channel {channel}; tried aliases {aliases}")
        array = dataset[variable].transpose(time_name, lat_name, lon_name).values
        arrays.append(array.astype(np.float32))

    tensor = np.stack(arrays, axis=1)
    latitudes = dataset[lat_name].values.astype(np.float32)
    return tensor, latitudes
