# ERA5-Minimum Data Contract

## 1. Purpose

Этот документ — единственный актуальный versioned source of truth для структуры данных ERA5-Minimum. Документы планирования и старые handoff-файлы могут описывать предыдущую стадию; при расхождении приоритет имеет этот contract.

## 2. Contract version

`DATA_CONTRACT_VERSION = 1.0`

## 3. Source dataset

Тестовая выгрузка — ERA5 hourly data on single levels за 2024-01-01. Она предназначена для проверки schema и loader, но не для честного обучения, temporal split или sample-efficiency research.

## 4. Raw storage layout

Локальный raw input находится в `data/raw/era5_single_2024_01_01/`. Контролируемая storage-копия находится в `storage-upload/era5-single-levels/2024-01-01/`. Raw data не коммитятся и не перезаписываются.

## 5. Raw file pair

- `data_stream-oper_stepType-instant.nc` — мгновенные поля.
- `data_stream-oper_stepType-accum.nc` — накопленное поле.

Рабочим raw input является именно эта пара, а не единый `prepared_era5.nc`.

## 6. Dimensions and coordinates

- Dimensions: `valid_time`, `latitude`, `longitude`.
- Raw field shape: `[4, 721, 1440]`.
- `valid_time`: 2024-01-01 00:00, 06:00, 12:00, 18:00 UTC.
- `latitude`: 90.0 → -90.0, строго убывает, шаг 0.25°.
- `longitude`: 0.0 → 359.75, строго возрастает, шаг 0.25°.
- `expver`: только `0001`.

## 7. Raw variable schema

| Файл | Raw variable | GRIB step type |
| --- | --- | --- |
| instant | `u10`, `v10`, `t2m`, `msl`, `sst`, `tcc`, `tcwv` | `instant` |
| accumulated | `tp` | `accum` |

## 8. Canonical channel schema

Канонический model channel order неизменен:

1. `u10`
2. `v10`
3. `t2m`
4. `msl`
5. `sst`
6. `tcc`
7. `tcwv`
8. `tp1h`

Raw `tp` переименовывается loader-ом в `tp1h`. Каноническое имя давления — `msl`, не `mslp`.

## 9. Units

| Channel | Unit |
| --- | --- |
| `u10`, `v10` | `m s**-1` |
| `t2m`, `sst` | `K` |
| `msl` | `Pa` |
| `tcc` | `(0 - 1)` |
| `tcwv` | `kg m**-2` |
| raw `tp` / canonical `tp1h` | `m` |

Training tensor сохраняет эти физические единицы до отдельного явно описанного шага нормализации. Преобразования K → °C, Pa → hPa, m → mm и `tcc` 0–1 → percent допустимы только для display, reports или явно описанных derived artifacts; они не должны молча менять training tensor.

## 10. Loader transformations

Loader проверяет raw schema, variables, units, `GRIB_stepType`, `expver`, timestamps, coordinates, NaN и SST mask. Затем он переименовывает `valid_time` в `time` и raw `tp` в canonical `tp1h`. Он не меняет физические единицы.

## 11. SST missing-data policy

Raw `sst` содержит NaN над сушей. Raw Dataset сохраняет эти NaN без изменения. Loader возвращает отдельную SST-valid mask; mask должна быть согласована между timestamps. В tensor representation SST NaN могут быть временно заменены нулём только вместе с отдельной mask.

## 12. Precipitation semantics

`tp` в этой выгрузке — accumulated field, а canonical loader name — `tp1h`. `tp6h` запрещено использовать, пока оно не построено из шести последовательных часовых значений с документированной агрегацией. Четыре timestamp с шагом шесть часов не образуют `tp6h`.

## 13. Temporal semantics

Текущие timestamps: 00:00, 06:00, 12:00, 18:00 UTC за один день. Они не достаточны для temporal split, model training или sample-efficiency research.

## 14. Half-degree subsampling

Первый smoke sample использует deterministic subsampling: берётся каждая вторая точка latitude и longitude. Это не interpolation и не averaging. Итоговая сетка — 361 × 720 с разрешением 0.5°.

## 15. Tensor contract

При half-degree subsampling tensor имеет shape `[4, 8, 361, 720]`, порядок `[time, channel, latitude, longitude]` и dtype `float32`. SST mask возвращается отдельно. Для текущей выгрузки SST valid fraction около `0.660934`.

## 16. Metadata contract

Каждый processed sample и loader report обязан содержать:

- `data_contract_version`;
- `source_dataset`;
- `source_files`;
- `source_checksums`;
- `timestamps`;
- `source_shape`;
- `output_shape`;
- `channel_order`;
- `units`;
- `latitude_range`;
- `longitude_range`;
- `grid_resolution`;
- `downsampling_method`;
- `sst_valid_fraction`;
- `sst_mask_consistent`;
- `missing_values_before_fill`;
- `precipitation_semantics`;
- `loader_version` или `git_commit`;
- `created_at`.

Это обязательства контракта; их добавление в код выполняется отдельным PR.

## 17. Processed artifact naming

- Smoke sample: `era5_smoke_<date-or-range>_<grid>_dc<contract-version>.npz`
- Dataset: `era5_<split>_<date-range>_<grid>_v<dataset-version>.zarr`
- Mask: `sst_ocean_mask_<grid>_v<mask-version>.npy`
- Report: `loader_report_<date-or-range>_dc<contract-version>.json`

Пример: `era5_smoke_2024-01-01_05deg_dc1.0.npz`.

## 18. Train/validation/test restrictions

Train, validation и test timestamps не пересекаются. Validation/test фиксируются между train sizes. Между split должен быть configurable temporal gap. Текущая четырёхточечная выгрузка не удовлетворяет этому назначению.

## 19. Normalization restrictions

Статистики нормализации fit только на выбранном training subset. Они не используют validation или test и не являются частью loader-а.

## 20. Storage and checksum policy

Raw files immutable. Их storage-copy проверяется через `storage-upload/manifests/SHA256SUMS.txt`. Processed artifacts, outputs и checkpoints не коммитятся в Git; в metadata сохраняются source checksums.

## 21. Security rules

Не хранить `.cdsapirc`, `.env`, API tokens, keys или credentials в Git, reports, Issues, PRs и storage manifests. Raw data распространяются только через разрешённый Team Workspace.

## 22. Compatibility rules

Любой consumer data обязан читать `data_contract_version`, channel order, units, mask и precipitation semantics. Legacy names `mslp`, `tp6h` и single-file `prepared_era5.nc` не совместимы с этим raw contract без отдельного явно документированного adapter-а.

## 23. Contract change procedure

Изменение contract version требует отдельного Issue и PR, review от @Pr1vate4 и @VsevolodCod, обновления `docs/DECISIONS.md`, tests и metadata schema. Изменение не может быть молчаливым.

## 24. Known limitations

- Есть только четыре timestamps за один день.
- Temporal split, train-only normalization и real-data training ещё не реализованы.
- `tp1h` не является `tp6h`.
- Реальное file/bitstream compression не измерялось.
