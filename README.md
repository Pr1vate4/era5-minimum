# ERA5-Minimum

Исследовательский каркас для Nuclear IT Hack: поиск минимальной репрезентативной обучающей выборки для автокодировщика ERA5 при сжатии 32–64×.

## Победная гипотеза

Не только число временных срезов, но и разнообразие погодных режимов определяет достаточный объём выборки. Проект должен находить минимальный набор, после которого качество восстановления перестаёт существенно улучшаться.

## Что уже работает

- синтетический ERA5-подобный датасет из 8 каналов;
- быстрый PCA-autoencoder smoke baseline с честными 32× по числу значений;
- каркас ConvAutoencoder для GPU-экспериментов 32× или 64×;
- обучение на нескольких размерах выборки;
- обычные и широтно-взвешенные метрики;
- CSV/JSON-отчёт;
- тесты формы, метрик и воспроизводимости.

## Быстрый запуск

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
make verify
```

Результаты появятся в `outputs/mvp/summary.csv`.

Проверить ERA5 download request без CDS credentials, сети и записи файлов:

```bash
make download-dry-run
```

## Честное ограничение MVP

Текущий коэффициент сжатия — **отношение числа исходных float-значений к числу latent-значений**. Это ещё не размер итогового бинарного файла. Реальное битовое сжатие требует квантизации и энтропийного кодирования и вынесено в следующий этап.

## Главные документы

- `AGENTS.md` — постоянные инструкции Codex;
- `CONTRIBUTING.md` — правила командного вклада и Definition of Done;
- `docs/CURRENT_STATE.md` — подтверждённое текущее состояние и блокеры;
- `docs/TEAM_ROLES.md` — роли и ownership;
- `docs/WORKFLOW.md` — GitHub workflow;
- `docs/HANDOFF_DAY2.md` — актуальный handoff после real ERA5 loader;
- [docs/DATA_CONTRACT.md](docs/DATA_CONTRACT.md) — единственный актуальный контракт ERA5 schema, channels и units;
- `docs/CONFIG_CONTRACT_GAP.md` — известное расхождение planned real-data config и текущего runner;
- `docs/DEVELOPMENT.md` — установка Python 3.12, local commands и CI-equivalent workflow;
- `docs/DOWNLOAD_WORKFLOW.md` — безопасная воспроизводимая выгрузка ERA5 через CDS;
- `docs/DAY1_PLAN.md` — задачи первого дня;
- `docs/ROADMAP.md` — этапы до защиты;
- `docs/THEORY.md` — теория и методология;
- `docs/CODEX_PROMPTS.md` — готовые промты;
- `docs/PITCH.md` — каркас защиты.
