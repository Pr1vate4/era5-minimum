# DAY 2 — Реальные данные ERA5 и NetCDF pipeline

> Historical planning document. Актуальная схема данных находится в [DATA_CONTRACT.md](DATA_CONTRACT.md). Указанный ниже single-file path `data/raw/era5_single_2024_01_01.nc` устарел и запрещён для нового workflow; используйте raw pair и [DOWNLOAD_WORKFLOW.md](DOWNLOAD_WORKFLOW.md).

## Цель дня

Перевести проект с synthetic-only MVP на первый честный real-data pipeline:

`NetCDF ERA5 → schema report → [time, channel, lat, lon] → temporal split → train-only normalization → smoke baseline`

Сегодня не требуется высокая точность модели. День считается успешным, если реальные данные корректно читаются, валидируются и проходят через pipeline.

## Definition of Done

- [ ] Стартовый synthetic MVP по-прежнему запускается.
- [ ] Получен маленький ERA5 NetCDF sample или официальный sample организаторов.
- [ ] Сырые данные не добавлены в Git.
- [ ] Создан `schema_report.json` с variables, dimensions, coordinates, units, time range, cadence, NaN и статистиками.
- [ ] Реализован loader с выходом `[time, channel, latitude, longitude]`.
- [ ] Поддерживается latitude по возрастанию и убыванию.
- [ ] Не выполнено молчаливое преобразование `total_precipitation` в `tp6h`.
- [ ] Реализован хронологический train/validation/test split без пересечения timestamps.
- [ ] Нормализация вычисляется только по выбранному train subset.
- [ ] Есть маленький unit-test NetCDF loader.
- [ ] `pytest -q` проходит.
- [ ] Synthetic MVP не сломан.

## Шаг 0 — проверить стартовый проект

```bash
cd era5-minimum-starter
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python -m era5_minimum.experiments --config configs/mvp.yaml
```

## Шаг 1 — получить real-data sample

Предпочтение:

1. sample/dataset организаторов;
2. если его пока нет — маленький официальный sample CDS.

Для CDS:

```bash
pip install "cdsapi>=0.7.7" netCDF4
python scripts/download_era5_example.py
```

Ожидаемый путь:

```text
data/raw/era5_single_2024_01_01.nc
```

Перед запуском требуется CDS account, принятие лицензии датасета и локальный `~/.cdsapirc`. Токен нельзя добавлять в репозиторий.

## Шаг 2 — Codex: schema inspector

Передать Codex промт `DAY2 PROMPT A` из текущего чата. Результат должен создать:

```text
src/era5_minimum/data/inspect_netcdf.py
tests/test_inspect_netcdf.py
outputs/schema/schema_report.json
```

Пример запуска:

```bash
python -m era5_minimum.data.inspect_netcdf \
  --path data/raw/era5_single_2024_01_01.nc \
  --output outputs/schema/schema_report.json
```

## Шаг 3 — человеческая проверка schema report

Проверить вручную:

- названия dimensions;
- названия variables;
- units;
- порядок latitude;
- longitude range;
- timestamp cadence;
- shape;
- NaN/masks, особенно SST;
- фактическое имя pressure;
- семантику precipitation.

Нельзя переходить к модели, пока неизвестны units и семантика каналов.

## Шаг 4 — Codex: NetCDF loader

Передать Codex промт `DAY2 PROMPT B` из текущего чата. Loader должен:

- читать NetCDF через xarray;
- разрешать явное mapping raw-variable → canonical-channel;
- выдавать `[time, channel, latitude, longitude]`;
- сохранять timestamps, lat, lon, units;
- проверять наличие всех requested channels;
- явно падать, если requested `tp6h`, а доступен только raw `total_precipitation`;
- не читать всё в RAM без необходимости;
- иметь unit-test с tiny generated NetCDF.

## Шаг 5 — Codex: temporal split

После loader передать промт `DAY2 PROMPT C`.

Обязательные инварианты:

- train, validation и test не пересекаются;
- validation/test одинаковы для всех train sizes;
- есть configurable temporal gap;
- normalization fit только на selected train subset;
- timestamps splits сохраняются в metadata.

## Шаг 6 — smoke run на real data

На 24 временных срезах выполнить только лёгкую проверку:

- загрузка;
- split;
- normalization;
- encode/decode PCA либо один forward ConvAE;
- метрики;
- сохранение resolved config.

Не тратить день на обучение тяжёлой модели.

## Разделение ответственности

### Пользователь / команда

- создать GitHub repository и дать Codex доступ;
- запустить команды;
- зарегистрироваться в CDS и принять лицензию;
- получить sample;
- присылать ядру проекта schema report, логи, diff и ошибки;
- проверять физический смысл units/variables;
- принимать коммиты после ревью.

### Ядро проекта (ChatGPT)

- формировать точные Codex prompts;
- разбирать schema report;
- определять canonical mapping;
- проверять split, normalization, metrics и compression claims;
- анализировать Codex diff и тесты;
- исправлять методологические ошибки;
- обновлять roadmap, decisions и следующие задачи;
- готовить исследовательский план и защиту.

### Codex

- реализовывать код в репозитории;
- добавлять тесты;
- запускать проверки;
- показывать diff, риски и команды воспроизведения.

## Что прислать ядру после каждого шага

1. Команду, которую запускали.
2. Полный вывод терминала при ошибке.
3. `git diff --stat`.
4. Изменённые файлы.
5. `schema_report.json` после инспекции данных.
6. Результат `pytest -q`.

## Stop conditions

Немедленно остановить разработку модели, если:

- units неизвестны;
- precipitation semantics не подтверждены;
- train/validation timestamps пересекаются;
- normalization использует validation/test;
- channel order меняется между runs;
- compression ratio считается только по latent shape, но называется file compression.
