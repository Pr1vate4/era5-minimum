# HANDOFF — PRE-HACK DAY 1

> Historical handoff. Актуальное состояние проекта находится в [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md), а актуальный data contract — в [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md).

**Date:** 2026-07-21  
**Project:** ERA5-Minimum  
**Status:** стартовый каркас готов и проверен

## Зафиксированная цель

Найти минимальную репрезентативную обучающую выборку для 32–64× автокодировщика ERA5. Сравнивать не только N, но и стратегии отбора погодных режимов.

## Готово

- структура репозитория;
- постоянные инструкции `AGENTS.md`;
- мастер-промт управляющего чата;
- план до защиты;
- теория, риски и decision log;
- готовые Codex-промты;
- synthetic ERA5-like generator на 8 каналах;
- CPU-safe PCA smoke baseline с tensor ratio 32×;
- ConvAutoencoder scaffold для GPU;
- normalized aggregate metrics;
- per-channel RMSE в физических шкалах synthetic data;
- latitude-weighted RMSE;
- reproducible configs и outputs;
- pytest и GitHub Actions.

## Проверка

```text
5 tests passed
MVP experiment completed
train sizes: 20, 40, 64
reported tensor compression ratio: 32.0x
outputs/mvp/summary.csv generated
```

Synthetic-результаты подтверждают только работоспособность pipeline, а не качество на ERA5.

## Критические открытые вопросы

1. Формат и объём данных организаторов.
2. Семантика `tp6h`.
3. Обязательные pressure levels.
4. Официальная метрика.
5. Определение 32–64×.
6. Доступная GPU.
7. Нужен ли настоящий bitstream.

## Следующие три приоритета

1. Отправить вопросы организаторам и получить dataset schema.
2. Подключить реальный NetCDF pipeline и строгий temporal split.
3. Обучить первый ConvAE 32× на малой реальной выборке и сохранить карты ошибок.

## Первое действие следующей сессии

Запустить Codex-промт `0. Аудит стартового каркаса`, затем создать репозиторий и закоммитить состояние до подключения данных.
