# ERA5 range download workflow

## Purpose

`scripts/download_era5_range.py` downloads an inclusive date range sequentially by calling the existing safe daily downloader. It does not duplicate the CDS request, credential, ZIP extraction, per-day checksum or raw-variable logic.

## Why datasets stay split by day

Every date remains an independent `era5_single_YYYY_MM_DD/` directory. This supports resume, isolated checksum verification, safe day-level overwrite and partial transfer to approved shared storage. Days are never merged into one raw NetCDF, ZIP or NPZ.

## Installing requirements

Dry-run and tests require the normal development environment. A real download additionally requires:

```bash
python -m pip install -e ".[download]"
```

CDS credentials stay only in `~/.cdsapirc`.

## Dry-run

```bash
python scripts/download_era5_range.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-03 \
  --output-root /tmp/era5-range-dry-run \
  --dry-run
```

Dry-run validates the inclusive dates, lists all 24 hourly timestamps, reads existing day state and prints planned actions. It does not import `cdsapi`, access CDS, create directories or write range metadata.

## Downloading a range

```bash
python scripts/download_era5_range.py \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --output-root data/raw
```

Dates are processed in stable chronological order with 24 timestamps, `00:00` through `23:00`, for every day. There are no parallel workers.

## Full 24-hour request

The hourly list is generated once by the range script and passed to the daily downloader. It does not infer precipitation aggregation: raw `tp` remains raw `tp`, while loader semantics remain defined by [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Output structure

```text
data/raw/
├── era5_single_2024_01_01/
├── era5_single_2024_01_02/
└── ranges/era5_range_2024_01_01_2024_01_07/
    ├── range_request.json
    └── range_manifest.json
```

The range directory contains no NetCDF files.

## Range request

`range_request.json` stores dates, total days, hourly timestamps, safe output-root representation, flags, creation time and range downloader name. It never stores credentials, user names, home paths or environment values.

## Range manifest

`range_manifest.json` records each date, relative target directory, status, timestamps, file names, checksum state, action and error. Its summary reports requested, downloaded, verified, skipped and failed day counts plus `completed`.

## Status meanings

- `planned`: not yet processed or dry-run action.
- `verified`: newly downloaded or overwritten day whose checksums passed.
- `skipped`: existing complete day verified and left unchanged.
- `failed`: unsafe, incomplete or unsuccessful day.

`completed=true` only when every requested day is `verified` or `skipped` and there are no failures.

## Resume behavior

Repeat the same command without `--overwrite`. Complete verified days are skipped; the first missing day is downloaded. A run resumes naturally because each day is independent and its daily manifest is checked before CDS is contacted.

## Existing complete days

A day is complete only when it has instant/accumulated NetCDF, `request.json`, `metadata.json`, `SHA256SUMS.txt`, valid JSON metadata and passing required SHA-256 entries. It is skipped without `--overwrite`.

## Existing incomplete days

An existing directory with a missing file, invalid JSON, malformed manifest, unsafe manifest path or checksum mismatch is not complete. Without `--overwrite` it becomes a clear failure; with `--overwrite` only the daily downloader performs its staging replacement. The range tool never deletes it itself.

## Continue-on-error

Default behavior stops after the first failed day and writes a partial manifest. `--continue-on-error` records the failure and processes later dates, but exits non-zero if any failed day remains.

## Overwrite policy

`--overwrite` is forwarded to the daily downloader for each date. It also causes a complete existing day to be refreshed using the daily downloader's safe staging policy. Do not use it merely to retry a healthy day.

## Checksum verification

The range tool verifies `SHA256SUMS.txt` in Python. It accepts only standard `<sha256>  <relative filename>` lines, rejects absolute/traversal paths, requires key daily entries and compares streaming SHA-256 values without modifying the manifest.

## Storage requirements

Raw range outputs remain ignored by Git. Verify every successful day before copying it, with its daily provenance artifacts, to approved Team Workspace.

## Seven-day pilot

Seven consecutive days are a technical pilot for request volume, daily manifests and storage workflow. They are **not** the final research dataset and cannot alone establish seasonal or weather-regime diversity.

## Shared storage upload

Upload only after the owner verifies complete daily manifests and checksums. Do not mark DATA-002 complete until the final intended range, diversity criteria and shared-storage verification are accepted.

## Security

No token CLI argument exists. Dry-run needs no credentials. Do not commit range metadata generated alongside raw data, raw artifacts or `~/.cdsapirc`; follow [SECURITY.md](../SECURITY.md).

## Common failures

- Invalid range: use real ISO dates with start date not later than end date.
- Existing incomplete day: inspect the manifest; use `--overwrite` only with explicit approval.
- Failed day: inspect `range_manifest.json`; resume later or use `--continue-on-error` when independent later dates should proceed.
- Missing `cdsapi`: install the optional download extra for a real request.

## Relationship to DATA-002

The range tool implements DATA-002B only. DATA-002C, a real seven-day CDS pilot, and the final diverse sequential dataset remain separate, uncompleted checkpoints in [TASKS.md](TASKS.md).

## Limitations

This tool does not run parallel downloads, retries without bounds, temporal split, normalization, model training or `tp6h` derivation. Real CDS range download is not exercised by CI or offline tests.
