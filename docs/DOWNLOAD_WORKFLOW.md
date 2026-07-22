# ERA5 download workflow

## 1. Purpose

`scripts/download_era5.py` — единственный воспроизводимый способ получить raw ERA5 pair для этого repository. Он не меняет raw variables: `tp` остаётся `tp` до rename в loader.

## 2. Who needs CDS access

CDS account и принятая licence нужны только участнику, который запускает реальный download. Остальные участники используют shared verified copy, loader и tests без CDS access.

## 3. Installing download support

```bash
python -m pip install -e ".[download]"
```

Обычная development installation и CI не устанавливают `cdsapi`.

## 4. Configuring `~/.cdsapirc`

Настройте credential стандартным способом CDS в личном `~/.cdsapirc`. Downloader передаёт это `cdsapi` и не читает, не печатает и не копирует файл. Не добавляйте token в repository, `.env`, Issue или metadata.

## 5. Dry-run

```bash
python scripts/download_era5.py \
  --date 2024-01-01 \
  --times 00:00 06:00 12:00 18:00 \
  --output-root data/raw \
  --dry-run
```

Dry-run validates date/times, normalizes duplicate times in chronological order, prints request and plan, but does not import `cdsapi`, contact CDS, create directories, or write files.

## 6. Downloading one day

```bash
python scripts/download_era5.py \
  --date 2024-01-01 \
  --times 00:00 06:00 12:00 18:00 \
  --output-root data/raw
```

## 7. Downloading full hourly timestamps

For a future sequential hourly dataset, pass all required hourly values explicitly. For example:

```bash
python scripts/download_era5.py --date 2024-01-01 \
  --times 00:00 01:00 02:00 03:00 04:00 05:00 06:00 \
  --output-root data/raw
```

The downloader permits any valid hourly/minute timestamp; it does not infer a precipitation aggregation.

## 8. Output structure

```text
data/raw/era5_single_2024_01_01/
├── data_stream-oper_stepType-instant.nc
├── data_stream-oper_stepType-accum.nc
├── request.json
├── metadata.json
└── SHA256SUMS.txt
```

With `--keep-archive`, `source_download.zip` is retained when CDS returned a ZIP archive.

## 9. `request.json`

The file records the exact request for `reanalysis-era5-single-levels`: product type, variables, date components, times, `data_format`, and `download_format`. It has no credentials.

## 10. `metadata.json`

Metadata records UTC creation time, requested fields, relative output directory, detected archive type, extracted files, sizes, NetCDF SHA-256 values, downloader revision when available, and precipitation warnings. It never records credentials, user identifiers, home paths, or environment variables.

## 11. SHA-256 verification

Run inside the day directory:

```bash
sha256sum -c SHA256SUMS.txt
```

The manifest covers both NetCDF files and `request.json`. `metadata.json` contains data hashes but is not included in the manifest, avoiding a self-reference.

## 12. Uploading into shared storage

First verify the manifest, then upload the immutable pair and provenance artifacts to the approved Team Workspace. Do not commit them to Git. Follow [storage-upload/README_DATA.md](../storage-upload/README_DATA.md).

## 13. Team workflow without CDS credentials

Use a verified shared copy, verify its checksums, then run the inspector and loader. `pytest`, `make verify`, and `make download-dry-run` need neither `cdsapi` nor `~/.cdsapirc`.

## 14. Overwrite policy

Existing per-day data cause a clear failure by default. `--overwrite` prepares a complete replacement in a staging directory and publishes it only after file discovery, metadata, and checksums succeed. It never writes directly over a final raw file.

## 15. Common errors

- `cdsapi is required`: install `python -m pip install -e ".[download]"`.
- Existing target: inspect it, then use `--overwrite` only with explicit approval.
- Unsafe ZIP member or missing pair: do not use the artifact; report the CDS response metadata.
- CDS access/licence failure: resolve it in CDS, not in repository configuration.

## 16. Security rules

Never commit raw data, archives, `request.json`, `metadata.json`, checksums containing raw filenames, credentials, tokens, or `~/.cdsapirc`. The script never accepts a token CLI argument.

## 17. Relationship to `DATA_CONTRACT.md`

[DATA_CONTRACT.md](DATA_CONTRACT.md) defines the confirmed raw schema and the loader’s canonical `tp1h` rename. The downloader merely obtains and names the raw instant/accumulated files; strict schema validation remains in `era5_loader.py`.

## 18. Precipitation warning

Raw `tp` is accumulated precipitation. The four sparse timestamps 00:00, 06:00, 12:00, and 18:00 do not create `tp6h`. This downloader does not rename `tp`, create `tp1h`, or derive `tp6h`.
