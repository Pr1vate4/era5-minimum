# HANDOFF — DAY 2: real ERA5 loader

**Дата:** 2026-07-22
**Текущий milestone:** Team Bootstrap

Актуальная schema данных зафиксирована в [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Выполнено

- Реализован NetCDF schema inspector.
- Реализован strict loader для пары instant/accumulated ERA5 NetCDF.
- Проверяются variables, units, `expver`, timestamps, coordinates, NaN и SST mask.
- Канонические каналы: `u10, v10, t2m, msl, sst, tcc, tcwv, tp1h`.
- Реализован deterministic subsampling 0.25° → 0.5°.
- Реальный output tensor имеет форму `[4, 8, 361, 720]`.
- SST mask возвращается и сохраняется отдельно; SST valid fraction около `0.660934`.
- Есть явное предупреждение: `tp1h` не является `tp6h`; автоматическое создание `tp6h` запрещено.
- Добавлены тесты loader и inspector.

## Current known conflicts

- В старых документах, configs и synthetic MVP встречаются `tp6h` и `mslp`.
- `research_template.yaml` задаёт `source=netcdf`, но текущий runner поддерживает только synthetic source.
- Download-скрипты используют конфликтующий output path.
- `pyproject.toml` и `requirements.txt` расходятся.
- `src/era5_minimum.egg-info` отслеживается Git.

## Чего нельзя делать дальше

- Нельзя начинать честное обучение и sample-efficiency research на четырёх timestamps.
- Нельзя создавать `tp6h` из текущих разреженных временных точек.
- Нельзя менять units, channel order, SST mask или data contract молча.
- Нельзя коммитить raw data, generated outputs, checkpoints или secrets.

## Следующий организационный шаг

Завершить Team Bootstrap: принять workflow, ownership, PR/Issue templates и правила review.

## Следующий технический шаг

После получения достаточного временного диапазона реализовать temporal split без утечки и train-only normalization с тестами и timestamp metadata.
