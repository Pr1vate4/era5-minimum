# First-stage backlog

Статусы отражают текущий repository state; они не содержат оценок сроков. Актуальные data semantics: [DATA_CONTRACT.md](DATA_CONTRACT.md).

## DATA-002 — Extended sequential ERA5 dataset

- **Цель:** получить достаточный последовательный hourly ERA5 range для честных future split и research.
- **Owner:** @Pr1vate4.
- **Priority / status:** P0 / in progress.
- **Dependencies:** none.
- **Inputs:** confirmed CDS request schema, safe downloader, approved Team Workspace.
- **Expected outputs:** полный hourly range; `request.json`; `metadata.json`; `SHA256SUMS.txt`; shared-storage upload; подтверждённая schema.
- **Acceptance criteria:** raw pair не в Git; manifest проходит verification; schema соответствует contract либо расхождение документировано и заблокировало consumer work.
- **Verification:** downloader dry-run, checksum verification, inspector и strict loader на representative pair.
- **Risks:** CDS licence/access, storage capacity, changed schema, insufficient temporal coverage.

### Completed checkpoint: 24-hour real-data smoke test

- [x] Выполнен настоящий CDS-запрос за 2024-01-02.
- [x] Получены 24 последовательных hourly timestamps.
- [x] Получена пара instant/accumulated NetCDF.
- [x] Downloader создал `request.json`, `metadata.json` и `SHA256SUMS.txt`.
- [x] Проверена целостность файлов: `sha256sum -c SHA256SUMS.txt` завершилась с `OK` для обоих NetCDF и `request.json`.
- [x] Real loader создал tensor `[24, 8, 361, 720]`.
- [x] Проверены canonical channels и физические единицы.
- [x] Проверена SST mask.
- [x] Подтверждено, что `tp1h` не является `tp6h`.

### DATA-002B — range downloader implementation — completed

- [x] Реализован последовательный range downloader, переиспользующий safe daily downloader.
- [x] Добавлены offline tests для range planning, checksums, resume и error policy.
- [x] Добавлены offline CLI и CI dry-run checks без CDS credentials или сети.

### DATA-002C — seven-day real CDS pilot

- [ ] Выполнить отдельную real CDS pilot за семь последовательных дней после явного запуска владельцем задачи.
- [ ] Проверить daily manifests и передать только подтверждённый результат в shared storage.

### Remaining DATA-002 work

- [ ] Определить итоговый временной диапазон для исследования.
- [ ] Скачать достаточно длинный последовательный hourly dataset.
- [ ] Обеспечить сезонное и погодное разнообразие данных.
- [ ] Загрузить подтверждённый raw dataset в общее хранилище.
- [ ] Проверить schema и checksums всего итогового набора.
- [ ] Передать dataset в DATA-003 для temporal split и train-only normalization.

## DATA-003 — Temporal split and train-only normalization

- **Цель:** создать leakage-safe split и training-only statistics для prepared real-data pipeline.
- **Owner:** @Pr1vate4.
- **Priority / status:** P0 / blocked.
- **Dependencies:** DATA-002.
- **Inputs:** verified sequential timestamps, contract version and raw checksums.
- **Expected outputs:** documented split metadata, normalization artifacts fit only on train, tests and reproducible configuration contract.
- **Acceptance criteria:** no overlap or adjacent leakage; validation/test fixed across train sizes; units and channel order preserved.
- **Verification:** unit/integration tests with timestamp assertions and documented commands.
- **Risks:** insufficient temporal range, hidden accumulation semantics, changing contract without migration.

## ML-001 — PCA baseline on prepared real data

- **Цель:** establish a reproducible PCA reference on real prepared data.
- **Owner:** @VsevolodCod.
- **Priority / status:** P1 / blocked.
- **Dependencies:** DATA-003.
- **Inputs:** approved split, normalized training data, fixed validation/test, data metadata.
- **Expected outputs:** config, metrics by canonical channel in physical units, runtime, tensor ratio and artifact metadata.
- **Acceptance criteria:** reproducible command; no data leakage; tensor ratio is not presented as file ratio.
- **Verification:** tests plus run on approved real-data subset.
- **Risks:** compute/memory constraints, incorrect inverse normalization, overfitting to a small range.

## ML-002 — ConvAE 32× baseline

- **Цель:** train the first controlled ConvAE 32× baseline on the same real-data protocol.
- **Owner:** @VsevolodCod.
- **Priority / status:** P1 / blocked.
- **Dependencies:** DATA-003.
- **Inputs:** ML-001 protocol, real prepared data and fixed split.
- **Expected outputs:** model config, reproducible run metadata, per-channel physical metrics and comparison with PCA.
- **Acceptance criteria:** fixed split/seed policy, latitude-weighted metrics and documented limitations.
- **Verification:** relevant tests and approved training run.
- **Risks:** available GPU memory, runtime, unstable optimization, misleading comparison with changed protocol.

## FE-001 — React dashboard on mock JSON

- **Цель:** prepare a product-facing dashboard without depending on unfinished real API.
- **Owner:** @devlifeee.
- **Priority / status:** P1 / ready; may run in parallel.
- **Dependencies:** none.
- **Inputs:** agreed mock JSON and visualization needs from README/roadmap.
- **Expected outputs:** mock-driven dashboard source, documented mock schema and screenshots or local run instructions.
- **Acceptance criteria:** no raw data or credentials; clearly labelled mock values; no claim of real experiment results.
- **Verification:** frontend-local checks defined in its Issue.
- **Risks:** mock contract drift before API-001, scope expansion into production API.

## API-001 — Visualization artifact/API contract

- **Цель:** fix the versioned contract between experiment artifacts and visualization consumers.
- **Owner:** @Pr1vate4; **review:** @devlifeee.
- **Priority / status:** P1 / planned.
- **Dependencies:** none.
- **Inputs:** current output metadata, FE-001 mock needs and data contract restrictions.
- **Expected outputs:** documented JSON/artifact schema, versioning and example synthetic/mock artifact without raw data.
- **Acceptance criteria:** frontend can render mock artifact; contract separates tensor/file ratio and preserves channel/units semantics.
- **Verification:** schema review by both owners and a consumer fixture.
- **Risks:** coupling to unfinished experiment runner, exposing raw data or unstable metric definitions.

## INFRA-002 — Docker development environment

- **Цель:** define a reproducible development container after local workflow stabilizes.
- **Owner:** @Pr1vate4.
- **Priority / status:** P2 / planned.
- **Dependencies:** none.
- **Inputs:** Python 3.12 environment contract and CI dependency set.
- **Expected outputs:** Docker development environment, documented build/run commands and no embedded credentials/data.
- **Acceptance criteria:** container runs tests and synthetic smoke without CDS, GPU or raw data.
- **Verification:** clean build and `make verify` inside container.
- **Risks:** image size, PyTorch platform compatibility, accidental inclusion of local artifacts.

## MON-001 — Prometheus/Grafana skeleton

- **Цель:** define observability skeleton only after the development runtime exists.
- **Owner:** @Pr1vate4.
- **Priority / status:** P3 / blocked.
- **Dependencies:** INFRA-002.
- **Inputs:** approved container/runtime boundaries and future service needs.
- **Expected outputs:** minimal monitored-service plan and safe local-only configuration.
- **Acceptance criteria:** no credentials, no production endpoints and no claim that model monitoring exists before services do.
- **Verification:** configuration validation specified by the future implementation Issue.
- **Risks:** premature infrastructure, unnecessary dependencies and secret-bearing dashboards.
