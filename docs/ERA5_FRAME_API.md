# ERA5 Frame Preparation API

Frame Preparation Service превращает точный timestamp из настроенного
WeatherBench/Zarr split в физический вход уже принятого N32 codec:

```text
WeatherBench/Zarr
-> exact timestamp selection
-> canonical channel assembly
-> established conservative 0.25 degree to 0.5 degree remapping
-> float32 [1, 28, 360, 720]
-> existing train-only normalization and codec
```

Сервис находится в
`src/era5_minimum/api/era5_frame_service.py`. Он не скачивает данные, не
подбирает ближайший timestamp, не пересчитывает статистики нормализации и не
создаёт промежуточный NPZ при сжатии.

## Model Contract

Единственный источник порядка и metadata каналов:
`src/era5_minimum/data/channel_spec.py`.

| Index | Channel | WeatherBench source | Level, hPa | Unit |
| ---: | --- | --- | ---: | --- |
| 0 | `t2m` | `2m_temperature` | - | `K` |
| 1 | `mslp` | `mean_sea_level_pressure` | - | `Pa` |
| 2 | `u10` | `10m_u_component_of_wind` | - | `m s-1` |
| 3 | `v10` | `10m_v_component_of_wind` | - | `m s-1` |
| 4 | `tp6h` | `total_precipitation_6hr` | - | `m` |
| 5 | `sst` | `sea_surface_temperature` | - | `K` |
| 6 | `tcwv` | `total_column_water_vapour` | - | `kg m-2` |
| 7 | `tcc` | `total_cloud_cover` | - | `0-1` |
| 8 | `T1000` | `temperature` | 1000 | `K` |
| 9 | `T925` | `temperature` | 925 | `K` |
| 10 | `T850` | `temperature` | 850 | `K` |
| 11 | `T700` | `temperature` | 700 | `K` |
| 12 | `U1000` | `u_component_of_wind` | 1000 | `m s-1` |
| 13 | `U925` | `u_component_of_wind` | 925 | `m s-1` |
| 14 | `U850` | `u_component_of_wind` | 850 | `m s-1` |
| 15 | `U700` | `u_component_of_wind` | 700 | `m s-1` |
| 16 | `V1000` | `v_component_of_wind` | 1000 | `m s-1` |
| 17 | `V925` | `v_component_of_wind` | 925 | `m s-1` |
| 18 | `V850` | `v_component_of_wind` | 850 | `m s-1` |
| 19 | `V700` | `v_component_of_wind` | 700 | `m s-1` |
| 20 | `Z1000` | `geopotential` | 1000 | `m2 s-2` |
| 21 | `Z925` | `geopotential` | 925 | `m2 s-2` |
| 22 | `Z850` | `geopotential` | 850 | `m2 s-2` |
| 23 | `Z700` | `geopotential` | 700 | `m2 s-2` |
| 24 | `Q1000` | `specific_humidity` | 1000 | `kg kg-1` |
| 25 | `Q925` | `specific_humidity` | 925 | `kg kg-1` |
| 26 | `Q850` | `specific_humidity` | 850 | `kg kg-1` |
| 27 | `Q700` | `specific_humidity` | 700 | `kg kg-1` |

The canonical model grid is cell-centred:

- latitude: `-89.75 ... 89.75`, south to north, 360 cells;
- longitude: `0.25 ... 359.75`, west to east, 720 periodic cells.

The configured native WeatherBench2 source is validated as the established
721 x 1440 grid before the existing first-order conservative remapper is used.
No interpolation or nearest-neighbour resampling is introduced.

## SST And Missing Values

`static.zarr/land_sea_mask` is conservatively remapped and the existing
`land_sea_mask <= 0.5` ocean policy is applied. WeatherBench2 SST can also be
missing in some ocean/ice cells. The established training policy intersects
the static ocean mask with `np.isfinite(SST)`, so the service preserves these
SST `NaN` values and reports their count separately. A `NaN` in another
channel or any positive/negative infinity rejects the frame.

The preparation service leaves SST land `NaN` values in physical data. The
existing codec receives the separate ocean mask, applies the checkpoint's
train-only normalization, and fills invalid normalized model cells according
to the established codec policy. The service never calls normalization fit.

## Endpoints

| Method | Endpoint | Result |
| --- | --- | --- |
| `GET` | `/api/v1/era5/timestamps` | Exact timestamps from the configured split |
| `GET` | `/api/v1/era5/frames/{timestamp}` | Validated frame metadata |
| `GET` | `/api/v1/era5/frames/{timestamp}/npz` | Compressed NPZ with `data` and `channel_order` |
| `POST` | `/api/v1/era5/frames/{timestamp}/compress` | Existing codec job response |

`timestamps` supports `start`, `end`, `offset`, and `limit`. Frame endpoints
require an exact timezone-qualified timestamp. If it is absent, the API
returns `timestamp_not_available` with `nearest_before` and `nearest_after`.

The NPZ endpoint uses the same keys as manual codec upload. The compression
endpoint passes the in-memory ndarray and ocean mask directly to
`CodecService`; it does not call the backend over HTTP and does not write a
temporary file.

## Runtime Configuration

The service reuses the weather API environment:

| Variable | Meaning |
| --- | --- |
| `ERA5_WEATHER_PROVIDER` | Must be `zarr` |
| `ERA5_DATASET_ROOT` | Local prepared dataset root |
| `ERA5_DATASET_SPLIT` | Split served by the API |

Synthetic arrays are used only in tests. If the configured manifest, split,
static mask, grid, channel metadata, or timestamp is unavailable, the
production API returns a domain error instead of demo data.
