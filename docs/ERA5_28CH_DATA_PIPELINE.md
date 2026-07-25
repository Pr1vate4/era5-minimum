# ERA5 28-channel data pipeline

The canonical source is the anonymous WeatherBench2 ERA5 Zarr declared in
`era5_minimum.data.weatherbench2.WB2_URL`. It contains the eight surface fields
`t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc` and T/U/V/Z/Q at 1000, 925, 850,
700 hPa. The exact model order is `CHANNEL_NAMES` in
`src/era5_minimum/data/channel_spec.py`; source variables remain native in
storage and the lazy adapter creates `[time, channel, latitude, longitude]`.

Install the project data dependencies from `pyproject.toml`, then inspect only
remote metadata:

```bash
python scripts/data/prepare_era5_28ch.py inspect
```

The native 0.25° output directory contains `train.zarr`, `validation.zarr`,
`test.zarr`, `static.zarr`, `manifest.json`, `manifest.json.sha256`, and
`_SUCCESS` only after all three named splits complete. Dynamic fields are
float32 and are chunked `time=4, latitude=180, longitude=180, level=4`.
SST stays NaN over land in physical stores. When present, `ocean_mask` is
`land_sea_mask <= 0.5`; invalid SST may become zero only in a normalized ML
batch together with this separate mask.

For a safe remote smoke range (not a full split):

```bash
python scripts/data/prepare_era5_28ch.py download --split validation \
  --start 2020-01-01 --end 2020-01-01 --output-dir data/era5_28ch_smoke
python scripts/data/prepare_era5_28ch.py validate --dataset-dir data/era5_28ch_smoke
python scripts/data/prepare_era5_28ch.py layer --dataset-dir data/era5_28ch_smoke \
  --split validation --variable T1000 --pressure-level-hpa 1000 --timestamp 2020-01-01T00:00:00
```

Full split commands are intentionally explicit and can be very large:

```bash
python scripts/data/prepare_era5_28ch.py download --split validation
python scripts/data/prepare_era5_28ch.py download --split test
python scripts/data/prepare_era5_28ch.py download --split train
python scripts/data/prepare_era5_28ch.py validate
python scripts/data/prepare_era5_28ch.py statistics
python scripts/data/prepare_era5_28ch.py remap --source data/era5_28ch_0p25_6h --output data/era5_28ch_0p5_6h
```

`statistics` performs explicit lazy Dask mean/std/count reductions on `train`
only; it never reads validation/test, but it can still take substantial time.
`remap` deliberately fails closed today: no nearest/subsample/coarsen/bilinear
replacement is allowed for the required first-order conservative 0.5° remap.
Implement a validated remapper with persistent weights before using that
command for production. A native full 0.25° split
is roughly 28 × 721 × 1440 × 4 bytes = 116 MB per timestamp before Zarr
compression; six years at four frames/day are therefore about 1 TB raw,
excluding metadata/static fields. Plan multiple TB of working space and do not
launch train without approval.

`get_layer` reads one selected timestamp/variable/level only and returns its
coordinates, values, finite mask, extrema and valid count for a later backend;
it must not be used to return whole splits or to serialize NaN directly to JSON.
