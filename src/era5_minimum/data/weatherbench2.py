import xarray as xr

from .channel_spec import CHANNEL_SPEC, validate_wb2_compatibility

WB2_URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"


def open_wb2_era5(time_slice: slice | None = None, validate: bool = True) -> xr.Dataset:
    """Open the remote WeatherBench2 ERA5 store when cloud access is requested."""
    try:
        import gcsfs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WeatherBench2 access requires gcsfs. Install the project data "
            "dependencies before using the remote dataset."
        ) from exc

    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(WB2_URL)
    ds = xr.open_zarr(mapper, consolidated=True)

    if time_slice:
        ds = ds.sel(time=time_slice)

    # Fail-fast валидация перед началом обработки
    if validate:
        validate_wb2_compatibility(ds, CHANNEL_SPEC)

    return ds
