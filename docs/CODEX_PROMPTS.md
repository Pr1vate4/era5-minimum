# Готовые промты для Codex

Каждый промт построен по схеме Goal / Context / Constraints / Done when.

## 0. Аудит стартового каркаса

```text
Goal:
Проведи строгий технический аудит стартового репозитория ERA5-Minimum и исправь только критические ошибки, мешающие воспроизводимому запуску MVP.

Context:
Прочитай AGENTS.md, README.md, configs/mvp.yaml, src/era5_minimum и tests. Проект исследует минимальный размер выборки для автокодировщика ERA5 при 32–64x сжатии.

Constraints:
Не усложняй архитектуру, не добавляй UI, не меняй определения метрик и не выдавай tensor compression ratio за реальный bitstream ratio. Сохрани CPU-совместимость.

Done when:
pytest -q проходит; python -m era5_minimum.experiments --config configs/mvp.yaml завершается; summary.csv создаётся; ты перечислил найденные риски и внесённые изменения.
```

## 1. Реальный NetCDF pipeline

```text
Goal:
Добавь production-quality загрузку подготовленного ERA5 NetCDF в существующий экспериментальный pipeline.

Context:
Используй AGENTS.md, src/era5_minimum/data/netcdf.py и configs/mvp.yaml. Ожидаемый тензор: [time, channel, latitude, longitude]. Каналы: t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc.

Constraints:
Не реализуй молча неверное преобразование raw total_precipitation в tp6h. Если tp6h отсутствует, выдай понятную ошибку и документируй требуемую preprocessing-команду. Поддержи latitude в обоих направлениях. Не загружай весь файл в память без необходимости, если xarray/dask доступны. Статистики нормализации — только train split.

Done when:
Есть config source=netcdf; unit-тест с маленьким временным NetCDF; проверка размерностей и каналов; старый synthetic MVP не сломан; все тесты проходят.
```

## 2. Честный хронологический split

```text
Goal:
Реализуй временной train/validation/test split без утечки и сохрани timestamps каждого split в metadata эксперимента.

Context:
Работай с data pipeline и experiments runner. Нам нужно сравнивать разные train sizes на неизменных validation/test периодах.

Constraints:
Нельзя случайно перемешивать временные срезы между split. Добавь configurable temporal gap между train и validation. Fit normalization only on selected training subset. Не меняй test set между train sizes.

Done when:
Тест доказывает отсутствие пересечения timestamps; resolved config и metadata содержат границы split; все эксперименты используют один validation/test набор.
```

## 3. Визуальный отчёт

```text
Goal:
Добавь генерацию статического научного отчёта по одному experiment run.

Context:
Используй summary.csv, сохранённые reconstruction samples, lat/lon и channel names.

Constraints:
Показывай original, reconstruction и absolute error в физических единицах. Не используй misleading разные шкалы без явного обозначения. Для каждого изображения подписывай channel, timestamp, units и train size.

Done when:
Одна команда создаёт report/ с PNG-графиками и markdown-summary; отчёт содержит quality-vs-train-size, per-channel metrics и limitation note про tensor/file ratio.
```

## 4. Sample-selection strategies

```text
Goal:
Добавь четыре стратегии выбора обучающих временных срезов: contiguous, random, seasonal-balanced и diversity-based.

Context:
Главная гипотеза проекта: разнообразие погодных режимов важнее простого увеличения N. Metadata должен содержать выбранные timestamps.

Constraints:
Одинаковый test set и одинаковый training budget. Diversity strategy сначала реализуй простым и объяснимым способом: features из spatial mean/std/min/max по каналам + clustering или farthest-point sampling. Не используй test данные при выборе train samples.

Done when:
Для каждой стратегии есть тесты, воспроизводимость по seed и единый benchmark, создающий сравнительную таблицу.
```

## 5. Реальное файловое сжатие

```text
Goal:
Добавь минимальный честный codec для latent: quantization + serialization + zstd baseline и измерь реальный размер файла.

Context:
Сейчас проект измеряет только tensor compression ratio. Новый код должен отдельно показывать raw input bytes, latent float bytes, quantized latent bytes и compressed bitstream bytes.

Constraints:
Не называть результат entropy-coded neural codec, если используется только zstd. Сохраняй reconstruction quality после quantization. Codec должен декодироваться обратно без доступа к исходному tensor.

Done when:
Есть encode/decode CLI, round-trip test, file compression ratio и таблица rate-distortion.
```

## 6. Финальный adversarial review

```text
Goal:
Выступи как строгий член жюри и найди причины не засчитать наше решение.

Context:
Прочитай весь репозиторий, результаты экспериментов, презентацию и AGENTS.md.

Constraints:
Проверяй data leakage, cherry-picking, неверные units, неправильное понимание tp6h, нечестный compression ratio, отсутствие baseline, невоспроизводимость и слабую связь выводов с данными. Ничего не исправляй до завершения списка проблем.

Done when:
Сначала создан risk report с severity/evidence; затем исправлены critical/high проблемы; тесты и ключевые эксперименты перезапущены; итоговый отчёт содержит доказательства исправлений.
```
