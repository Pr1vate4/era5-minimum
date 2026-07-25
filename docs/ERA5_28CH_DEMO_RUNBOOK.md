# ERA5 28-channel real-data demo

`data/era5_28ch_demo` is a local, ignored integration dataset, not a training
corpus. It contains `validation.zarr`, `static.zarr`, `manifest.json` and its
SHA256. The verified range is 2020-01-01 00:00/06:00/12:00/18:00 UTC: four
float32 721×1440 global frames. Native stores preserve physical WeatherBench2
variables and SST NaNs over land. The logical order is `t2m, mslp, u10, v10,
tp6h, sst, tcwv, tcc, T1000..T700, U1000..U700, V1000..V700, Z1000..Z700,
Q1000..Q700`; `assemble_model_tensor` creates it lazily.

Validate and serve it:

```bash
python scripts/data/prepare_era5_28ch.py validate --dataset-dir data/era5_28ch_demo
ERA5_WEATHER_PROVIDER=zarr ERA5_DATASET_ROOT=data/era5_28ch_demo ERA5_DATASET_SPLIT=validation \
  python -m uvicorn era5_minimum.api.app:app --port 8000
curl http://localhost:8000/api/v1/datasets/current
curl http://localhost:8000/api/v1/timestamps
curl 'http://localhost:8000/api/v1/layers?variable=t2m&timestamp=2020-01-01T00:00:00Z&mode=original&target_width=360&target_height=180'
```

Layers default to 360×180 *visualization index sampling*. It is neither the
scientific 0.5° product nor conservative remapping. `reconstructed` and
`error` return 501 until a real codec is connected.

For the browser globe, start the API stack and Vite:

```bash
make monitoring-up
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173/#/overview`. Vite proxies `/api` to the local API;
the globe requests the real catalog and an `original` layer at 720×360 display
sampling (approximately 0.5°). The green `ERA5 Zarr · реальные данные` badge
and point inspector values are produced from the API response, not from
`public/data/globe`. This display sampling is for visualization only and must
not be reported as a conservative scientific remap. Reconstruction/error
modes and the separate Earth cloud overlay still require a connected codec and
a live TCC cloud-texture adapter, respectively.

Do not commit this dataset. To transfer it after checking free disk space,
create an archive manually (not from CI):

```bash
tar -I 'zstd -T0 -10' -cf era5_28ch_demo.tar.zst -C data era5_28ch_demo
sha256sum era5_28ch_demo.tar.zst > era5_28ch_demo.tar.zst.sha256
tar -I zstd -xf era5_28ch_demo.tar.zst -C data
```

The next research step is an approved train split, fixed nested N selections,
train-only statistics and a separately validated conservative 0.5° remap.
