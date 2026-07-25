import xarray as xr
import numpy as np


def apply_masked_sst_remap(ds_in: xr.Dataset, ds_out_grid: xr.Dataset, ocean_fraction: xr.DataArray,
                           regridder) -> xr.DataArray:
    """
    Маскированный ремаппинг SST: суша остается NaN.
    ocean_fraction должен быть на исходной сетке ds_in.
    """
    sst = ds_in["sea_surface_temperature"]

    # Ремаппинг числителя и знаменателя
    num = regridder(sst * ocean_fraction)
    den = regridder(ocean_fraction)

    # Избегаем деления на ноль, порог океана (например, > 0.5)
    remapped_sst = xr.where(den > 0.5, num / den, np.nan)
    remapped_sst.name = "sst"
    return remapped_sst