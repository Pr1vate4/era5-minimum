# Текущее состояние проекта

**Дата обновления:** 2026-07-22

## Цель проекта

Исследовать минимальный объём и состав обучающей выборки для нейросетевого сжатия ERA5 в 32–64 раза без подмены tensor ratio реальным файловым сжатием.

## Confirmed

- Получена пара raw NetCDF за 2024-01-01: instant-поля и accumulated-поле.
- Временные точки: 00:00, 06:00, 12:00 и 18:00 UTC.
- Исходная сетка: 721 × 1440, 0.25°; latitude убывает от 90 до -90, longitude возрастает от 0 до 359.75.
- `expver` тестовой выборки равен `0001`.
- `sst` содержит пропуски над сушей; valid fraction после subsampling около `0.660934`.
- В исходных данных давление называется `msl`, а accumulated precipitation называется `tp`. Текущий `tp` — `tp1h`, не `tp6h`.
- Четырёх timestamps недостаточно для честного temporal split, обучения модели или построения `tp6h`.

## Implemented

- NetCDF schema inspector.
- Строгий loader пары instant/accumulated NetCDF с проверками variables, units, coordinates, `expver`, NaN и SST mask.
- Канонический порядок: `u10, v10, t2m, msl, sst, tcc, tcwv, tp1h`.
- Детерминированный subsampling 0.25° → 0.5°.
- Output tensor `[4, 8, 361, 720]` и отдельная SST mask.
- `loader_report.json`, тесты loader и inspector, запрет молчаливого создания `tp6h`.

## In progress

- Team Bootstrap: ownership, GitHub workflow, Issue/PR templates и handoff.

## Not started

- Temporal split без утечки.
- Train-only normalization.
- Real-data training baseline.
- Sample-efficiency research.
- Реальное файловое сжатие и frontend/demo.

## Known conflicts

- Старые документы и шаблоны всё ещё содержат `tp6h` и `mslp`, хотя raw pipeline использует `tp1h` и `msl`.
- `configs/research_template.yaml` содержит `source=netcdf`, но `experiments.py` пока поддерживает только synthetic source.
- Два download-скрипта используют конфликтующий output path.
- `pyproject.toml` и `requirements.txt` содержат разные наборы зависимостей.
- `src/era5_minimum.egg-info` отслеживается Git, хотя это generated metadata.

## Current milestone

Team Bootstrap.

## Next technical milestone

Temporal split и train-only normalization — только после документирования командного процесса и получения достаточного временного диапазона.

## Blocking conditions

- Нельзя начинать честное обучение или sample-efficiency research на четырёх timestamps.
- Нельзя создавать или называть данные `tp6h` без подтверждённой семантики и достаточной последовательности часов.
- Нельзя менять data contract, units, channel order, split или metric definitions без review и документации.
