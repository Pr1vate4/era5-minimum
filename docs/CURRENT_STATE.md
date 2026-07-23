# Текущее состояние проекта

**Дата обновления:** 2026-07-23

## Цель проекта

Исследовать минимальный объём и состав обучающей выборки для нейросетевого сжатия ERA5 в 32–64 раза без подмены tensor ratio реальным файловым сжатием.

## Confirmed

- Актуальный source of truth по данным: [DATA_CONTRACT.md](DATA_CONTRACT.md).
- Получена пара raw NetCDF за 2024-01-01: instant-поля и accumulated-поле.
- Временные точки: 00:00, 06:00, 12:00 и 18:00 UTC.
- Исходная сетка: 721 × 1440, 0.25°; latitude убывает от 90 до -90, longitude возрастает от 0 до 359.75.
- `expver` тестовой выборки равен `0001`.
- `sst` содержит пропуски над сушей; valid fraction после subsampling около `0.660934`.
- В исходных данных давление называется `msl`, а accumulated precipitation называется `tp`; loader использует каноническое имя `tp1h`, не `tp6h`.
- Четырёх timestamps недостаточно для честного temporal split, обучения модели или построения `tp6h`.
- Для 2024-01-02 получена отдельная суточная raw pair с 24 последовательными hourly timestamps от 00:00 до 23:00 UTC. Это smoke-test pipeline, а не достаточный research dataset.

## Implemented

- `pyproject.toml` — canonical dependency source; CI использует Python 3.12 и project extras.
- Generated `*.egg-info` удалены из Git и игнорируются.
- NetCDF schema inspector.
- Строгий loader пары instant/accumulated NetCDF с проверками variables, units, coordinates, `expver`, NaN и SST mask.
- Канонический порядок: `u10, v10, t2m, msl, sst, tcc, tcwv, tp1h`.
- Детерминированный subsampling 0.25° → 0.5°.
- Output tensor `[4, 8, 361, 720]` и отдельная SST mask.
- `loader_report.json`, тесты loader и inspector, запрет молчаливого создания `tp6h`.
- Единый безопасный ERA5 downloader с offline dry-run, request/metadata/checksum artifacts и deprecated wrappers старых scripts.
- Sequential range downloader, который переиспользует daily downloader, хранит дни независимо, имеет offline tests/dry-run и прошёл реальный семидневный pipeline pilot.
- Team onboarding, assigned first-stage backlog, security policy, CODEOWNERS, Issue/PR templates и Python 3.12 CI.

### Real 24-hour ERA5 smoke test — completed

- Настоящий CDS-запрос за 2024-01-02 успешно создал `data/raw/era5_single_2024_01_02/` с `source_download.zip`, instant/accumulated NetCDF pair, `request.json`, `metadata.json` и `SHA256SUMS.txt`.
- Downloader безопасно распаковал ZIP, обнаружил raw pair, сохранил provenance и checksums; raw `tp` не изменялся и `tp6h` не создавался.
- SHA-256 verification passed: `data_stream-oper_stepType-instant.nc`, `data_stream-oper_stepType-accum.nc` и `request.json` проверены командой `sha256sum -c SHA256SUMS.txt`.
- Real loader успешно создал `outputs/real_era5/sample_2024_01_02.npz` и `outputs/real_era5/loader_report_2024_01_02.json`. NPZ содержит data, SST mask, timestamps, latitude, longitude, channel names и units.
- Source shape: `[24, 8, 721, 1440]`; output shape: `[24, 8, 361, 720]`; data dtype: `float32`.
- Канонический порядок: `u10, v10, t2m, msl, sst, tcc, tcwv, tp1h`; физические units сохранены.
- SST mask имеет shape `[361, 720]`, dtype `bool`, согласована для всех timestamps; SST NaN count before fill: `8445024`; valid fraction: `0.660934133579563`.
- Timestamps имеют shape `[24]`; latitude и longitude — `[361]` и `[720]`; coordinate ranges в loader report: latitude `-90.0`—`90.0`, longitude `0.0`—`359.5`; downsampling — `subsampling_every_second_grid_point`.
- Warning loader остаётся явным: `tp1h` — one-hour accumulation, не `tp6h`; `tp6h` не создан.
- Одних суток недостаточно для temporal split, train-only normalization, честного обучения, PCA/ConvAE research, sample-efficiency research или вывода о минимальном размере выборки.

### Real ERA5 seven-day pilot — completed (DATA-002C)

- Локально проверен период `2024-01-02`—`2024-01-08` UTC: семь последовательных дней и 168 последовательных hourly timestamps от `2024-01-02T00:00:00Z` до `2024-01-08T23:00:00Z`; пропусков и дубликатов нет.
- Все семь дневных raw-наборов содержат instant/accumulated NetCDF pair, `request.json`, `metadata.json` и `SHA256SUMS.txt`; все проверяемые SHA-256 entries прошли.
- `range_manifest.json` содержит `requested_days=7`, `verified_days=7`, `skipped_days=1`, `failed_days=0` и `completed=true`. День `2024-01-02` корректно пропущен как уже complete (`skip-existing`); `2024-01-03`—`2024-01-08` загружены и verified.
- Loader успешно обработал каждый день: найдены семь NPZ и семь valid JSON loader reports. Каждый NPZ имеет shape `[24, 8, 361, 720]`, dtype `float32`, а всего получено 168 time slices.
- Во всех NPZ совпадают canonical channel order `u10, v10, t2m, msl, sst, tcc, tcwv, tp1h`, units, coordinates и SST mask. `tp1h` не переименовывался в `tp6h`; `tp6h` не создавался.

The seven-day dataset is a real multi-day pipeline pilot, not the final research dataset.

## Completed in current milestone

- Team Bootstrap final verification: documentation, ownership, workflow, security policy и onboarding audit.

## Not started

- Temporal split без утечки.
- Train-only normalization.
- PCA baseline на достаточных подготовленных real data.
- ConvAE 32× baseline на real data.
- Sample-efficiency research.
- FastAPI, React frontend и Prometheus/Grafana.
- Реальное файловое сжатие.

## Known conflicts

- Старые документы и шаблоны всё ещё содержат `tp6h` и `mslp`, хотя raw pipeline использует `tp1h` и `msl`.
- `configs/research_template.yaml` содержит `source=netcdf`, но `experiments.py` пока поддерживает только synthetic source.
- Подробное расхождение planned config и current runner: [CONFIG_CONTRACT_GAP.md](CONFIG_CONTRACT_GAP.md).

## Current milestone

DATA-002 — расширение последовательного ERA5 dataset. 24-часовой smoke-test, range downloader и семидневный real pipeline pilot completed, но DATA-002 остаётся in progress до согласования и получения достаточно длинного, разнообразного final research dataset с verified shared-storage copy.

Следующий этап DATA-002: согласовать итоговый временной диапазон с ML Lead, определить сезонные блоки или иной способ обеспечить разнообразие, оценить доступное хранилище, загрузить итоговый исследовательский набор и передать его в DATA-003.

## Blocking conditions

- Нельзя начинать temporal split, train-only normalization, честное обучение или sample-efficiency research на одних сутках, даже при 24 последовательных timestamps.
- Нельзя создавать или называть данные `tp6h` без подтверждённой семантики и достаточной последовательности часов.
- Нельзя менять data contract, units, channel order, split или metric definitions без review и документации.
