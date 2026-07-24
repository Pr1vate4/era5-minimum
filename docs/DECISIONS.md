# Decision log

Актуальный contract данных: [DATA_CONTRACT.md](DATA_CONTRACT.md).

## D-001 — Основной объект исследования

**Решение:** искать минимальную репрезентативную выборку, а не максимизировать размер модели.

**Причина:** напрямую соответствует кейсу и позволяет получить сильный вывод при ограниченном compute.

## D-002 — Первый baseline

**Решение:** deterministic ConvAutoencoder до VAE/Transformer.

**Причина:** нужен быстро проверяемый baseline и изоляция ошибок data pipeline.

## D-003 — Два определения сжатия

**Решение:** отдельно хранить tensor ratio и file/bitstream ratio.

**Причина:** latent меньшего размера не равен готовому сжатому файлу.

## D-004 — Основная метрика

**Временное решение:** latitude-weighted RMSE плюс per-variable RMSE/MAE.

**Статус:** требует подтверждения организаторов и сверки с предоставленным baseline.

## D-005 — Первый MVP на synthetic data

**Решение:** проверить программный цикл до доступа к реальному датасету.

**Причина:** synthetic data не доказывает качество модели, но быстро обнаруживает ошибки форм, метрик, конфигов и автоматизации.

## DEC-007 — Единый источник истины по данным

**Status:** accepted
**Date:** 2026-07-22

**Context:** schema, names и пути данных были распределены между planning documents и loader implementation.

**Decision:** единственным актуальным источником истины по структуре данных является `docs/DATA_CONTRACT.md`.

**Consequences:** актуальные документы ссылаются на contract; historical documents сохраняются с явным предупреждением.

## DEC-008 — Canonical precipitation channel

**Status:** accepted
**Date:** 2026-07-22

**Context:** raw accumulated variable называется `tp`, а четыре разреженных timestamps не подтверждают семантику `tp6h`.

**Decision:** raw precipitation сохраняет имя `tp`, а после loader rename canonical model channel называется `tp1h`.

**Consequences:** `tp6h` нельзя создавать или использовать без шести последовательных часовых значений и отдельной документированной агрегации.

## DEC-009 — Canonical sea-level pressure name

**Status:** accepted
**Date:** 2026-07-22

**Context:** confirmed raw instant file использует variable name `msl`, тогда как часть historical material использует `mslp`.

**Decision:** каноническое имя давления на уровне моря — `msl`.

**Consequences:** новые consumers используют `msl`; legacy references требуют явной классификации или adapter-а.

## DEC-010 — Raw input layout

**Status:** accepted
**Date:** 2026-07-22

**Context:** фактическая выгрузка разделена по GRIB step type на два NetCDF.

**Decision:** рабочий raw input — пара instant/accumulated NetCDF, а не единый `prepared_era5.nc`.

**Consequences:** loader и future pipeline принимают pair paths; single-file legacy plan не считается текущим contract.

## DEC-011 — Historical documentation compatibility

**Status:** accepted
**Date:** 2026-07-22

**Context:** historical planning documents важны для происхождения решений, но могут содержать устаревшие names и planned architecture.

**Decision:** historical documents сохраняются, но должны ссылаться на актуальный data contract или иметь historical banner.

**Consequences:** история не переписывается; при конфликте приоритет имеет `docs/DATA_CONTRACT.md`.

## DEC-012 — Canonical dependency source

**Status:** accepted
**Date:** 2026-07-22

**Context:** `pyproject.toml` и `requirements.txt` содержали разные dependency sets, а CI не получал NetCDF backend.

**Decision:** canonical dependency source — `pyproject.toml`; `requirements.txt` остаётся compatibility-file и зеркалит core плюс test dependency.

**Consequences:** dependency changes сначала вносятся в `pyproject.toml`; CI и Makefile устанавливают project extras через editable install.

## DEC-013 — Official Python version

**Status:** accepted
**Date:** 2026-07-22

**Context:** командная версия Python не была зафиксирована.

**Decision:** официальная командная и CI-версия — Python 3.12.

**Consequences:** `pyproject.toml` требует Python 3.12 или новее; CI проверяет Python 3.12 без необоснованного верхнего предела.

## DEC-014 — Optional CDS download support

**Status:** accepted
**Date:** 2026-07-22

**Context:** только download scripts импортируют `cdsapi`; loader, tests и CI не требуют CDS credentials.

**Decision:** `cdsapi` оформлен optional extra `download`.

**Consequences:** обычная установка и CI не устанавливают CDS client; скачивающий участник явно выбирает `.[download]` и использует только `~/.cdsapirc`.

## DEC-015 — Generated package metadata

**Status:** accepted
**Date:** 2026-07-22

**Context:** generated `src/era5_minimum.egg-info` был отслеживаемым и мог расходиться с package source.

**Decision:** generated package metadata, включая `*.egg-info`, не хранится в Git.

**Consequences:** metadata создаётся editable install локально и игнорируется Git; source package остаётся в `src/era5_minimum`.

## DEC-016 — Unified ERA5 downloader

**Status:** accepted
**Date:** 2026-07-22

**Context:** два scripts содержали разные CDS request schemas, time coverage и конфликтующий single-file output path.

**Decision:** единый downloader находится в `scripts/download_era5.py`; старые scripts остаются только deprecated wrappers без request schema.

**Consequences:** documentation и новые integrations используют один CLI; wrappers предупреждают участника и делегируют ему аргументы.

## DEC-017 — Safe download staging and overwrite

**Status:** accepted
**Date:** 2026-07-22

**Context:** raw ERA5 files нельзя частично записывать или молча заменять.

**Decision:** download идёт во temporary `.part` и staging directory; существующий day dataset отклоняется без `--overwrite` и заменяется только после подготовки новой полной пары.

**Consequences:** failed request не выглядит как successful dataset; overwrite требует явного user intent и оставляет старые данные до готовности replacement.

## DEC-018 — CDS credential boundary

**Status:** accepted
**Date:** 2026-07-22

**Context:** tests, CI и data loader не должны зависеть от личных CDS credentials.

**Decision:** credentials хранятся только в `~/.cdsapirc`; downloader lazy-imports `cdsapi` только для real request.

**Consequences:** dry-run и tests offline; credentials никогда не входят в repository, request.json, metadata.json, manifests или logs.

## DEC-019 — Download provenance artifacts

**Status:** accepted
**Date:** 2026-07-22

**Context:** raw data должны быть проверяемыми и воспроизводимыми без хранения credentials.

**Decision:** каждая successful raw download сохраняет `request.json`, `metadata.json` и `SHA256SUMS.txt` рядом с confirmed NetCDF pair.

**Consequences:** request, files and hashes можно проверить перед shared-storage upload; manifest не содержит self-referential metadata hash.

## DEC-020 — Independent daily raw datasets

**Status:** accepted
**Date:** 2026-07-23

**Context:** multi-day raw ERA5 downloads must be resumable, individually verifiable and safe to replace one day at a time.

**Decision:** многодневный ERA5 dataset хранится отдельными дневными наборами, а не одним огромным raw-файлом.

**Consequences:** каждая дата сохраняет собственные request, metadata и checksums; failure одного дня не повреждает остальные.

## DEC-021 — Range downloader delegates to daily downloader

**Status:** accepted
**Date:** 2026-07-23

**Context:** CDS request schema, ZIP safety, credentials and daily provenance уже реализованы и проверены в одном месте.

**Decision:** range downloader переиспользует однодневный downloader и не дублирует CDS request schema.

**Consequences:** изменения per-day download logic остаются централизованными; range layer отвечает только за dates, resume и range metadata.

## DEC-022 — Verified-day resume rule

**Status:** accepted
**Date:** 2026-07-23

**Context:** наличие директории не доказывает целостность raw data.

**Decision:** повторный запуск пропускает только полные дни с успешно проверенными checksums.

**Consequences:** incomplete или corrupted daily directory без `--overwrite` фиксируется как failure; range layer самостоятельно ничего не удаляет.

## DEC-023 — Honest range failure reporting

**Status:** accepted
**Date:** 2026-07-23

**Context:** partial range downloads требуют проверяемого статуса для каждого дня и безопасного resume.

**Decision:** ошибки диапазона фиксируются в `range_manifest.json`, а наличие хотя бы одного failed day запрещает status `completed`.

**Consequences:** default run stops at first failure; `--continue-on-error` processes later dates but still returns a non-zero result when failures remain.

## DEC-024 — Seven-day range is a pipeline pilot, not a research dataset

**Status:** accepted
**Date:** 2026-07-23

**Context:** локально подтверждённый диапазон `2024-01-02`—`2024-01-08` показывает, что multi-day download, checksum verification, resume/skip-existing и loader workflow технически устойчивы на 168 hourly timestamps.

**Decision:** считать этот семидневный диапазон техническим multi-day pipeline pilot, а не окончательным исследовательским dataset.

**Consequences:** пилот не подтверждает сезонную или погодную репрезентативность и не открывает temporal split, train-only normalization или model research. Итоговый объём и selection strategy определяются отдельно совместно с ML Lead до завершения DATA-002.

## DEC-025 — First honest codec step uses scalar quantization plus canonical Huffman

**Status:** accepted
**Date:** 2026-07-24

**Context:** проекту нужен реальный codec contour с сериализованным bitstream и exact symbol roundtrip до появления финальной 32x/64x модели. Latent reduction без entropy coding не считается codec-результатом, а сложный entropy model на первом шаге повышает implementation risk.

**Decision:** первый ML-001 codec harness использует детерминированное per-channel scalar quantization и canonical Huffman coding с self-describing header.

**Consequences:** проект сразу получает честный serialized payload, проверку exact roundtrip по квантованным символам и отдельный serialized ratio. Первый end-to-end integration path прогоняет через codec квантованный PCA latent и считает reconstruction metrics уже после quantize/dequantize. Tensor ratio и serialized ratio остаются разными метриками и не подменяют друг друга. Этот contour нужен для локальной проверки bitstream contract и rate accounting, а не как финальная конкурсная архитектура.

## DEC-026 — Latent probe starts from frozen PCA latent

**Status:** accepted
**Date:** 2026-07-24

**Context:** ML-001 требует отдельный +6h latent probe with persistence baseline, but the full learned codec is not ready yet.

**Decision:** first latent probe uses frozen PCA latents, trains a compact predictor on consecutive latent pairs, and records improvement against persistence.

**Consequences:** probe workflow becomes runnable now, stays reproducible, and keeps encoder/decoder frozen during probe training.

## DEC-027 — Codec smoke uses synthetic 28-channel ConvAE baseline

**Status:** accepted
**Date:** 2026-07-24

**Context:** ML-001 still needs a runnable smoke codec path before the full real-data 32x/64x training stack lands.

**Decision:** smoke codec training uses a synthetic 28-channel tensor, a small ConvAE, masked SST on land, real canonical Huffman bitstreams, and separate validation/test bitstreams.

**Consequences:** the repo now has a reproducible local codec contour with checkpoint, resource log, metrics, and exact symbol roundtrip, while remaining explicit that it is a smoke baseline rather than the final research model.
