import os
import xarray as xr
import numpy as np


def write_zarr_with_resume(ds: xr.Dataset, target_path: str, chunk_shape: tuple = (1, 28, 180, 180)):
    """Запись в Zarr с поддержкой resume и без дубликатов."""
    dims = ["time", "channel", "lat", "lon"]
    ds = ds.chunk(dict(zip(dims, chunk_shape)))

    if os.path.exists(target_path):
        existing = xr.open_zarr(target_path)
        last_time = existing.time.values[-1]
        # Фильтруем уже записанные timestamps
        ds = ds.sel(time=slice(np.datetime64(last_time) + np.timedelta64(1, 'h'), None))
        if len(ds.time) == 0:
            print("Все данные уже записаны. Resume не требуется.")
            return

        # Append mode
        ds.to_zarr(target_path, append_dim="time", compute=True)
    else:
        # Первая запись
        ds.to_zarr(target_path, mode="w", compute=True)