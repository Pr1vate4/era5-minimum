# Локальная разработка

## 1. Supported environment

Единственный canonical source зависимостей — `pyproject.toml`. Официально проверяемая командная и CI-версия — Python 3.12. Другие версии могут работать, но CI их не гарантирует.

## 2. Python version

```bash
python3.12 --version
```

## 3. Clone repository

```bash
git clone https://github.com/Pr1vate4/era5-minimum.git
cd era5-minimum
```

## 4. Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 5. Install base environment

```bash
make install
```

Эквивалентная canonical команда: `python -m pip install -e .`.

## 6. Install development environment

```bash
make install-dev
```

Эта команда устанавливает core dependencies и `pytest` из optional extra `dev`.

## 7. Install download support

Только участник, который скачивает raw ERA5 из CDS, устанавливает optional extra:

```bash
python -m pip install -e ".[download]"
```

## 8. Run tests

```bash
make test
```

## 9. Run synthetic smoke test

```bash
make smoke
```

Это synthetic MVP, не обучение на реальных данных.

## 10. Run ERA5 inspector

Raw files не входят в Git. Для одной части raw pair:

```bash
python -m era5_minimum.data.inspect_netcdf \
  --path data/raw/era5_single_2024_01_01/data_stream-oper_stepType-instant.nc \
  --output outputs/schema/instant_schema_report.json
```

## 11. Run real ERA5 loader

```bash
python -m era5_minimum.data.era5_loader \
  --instant data/raw/era5_single_2024_01_01/data_stream-oper_stepType-instant.nc \
  --accumulated data/raw/era5_single_2024_01_01/data_stream-oper_stepType-accum.nc \
  --half-degree \
  --output outputs/real_era5/era5_smoke_2024-01-01_05deg_dc1.0.npz \
  --report outputs/real_era5/loader_report_2024-01-01_dc1.0.json
```

Текущие четыре timestamps достаточны только для schema/loader smoke check, не для temporal split или real training. Полный contract: [DATA_CONTRACT.md](DATA_CONTRACT.md).

## 12. Make targets

- `make install` — editable core installation.
- `make install-dev` — editable installation с `pytest`.
- `make test` — test suite.
- `make mvp` — synthetic MVP.
- `make smoke` — alias synthetic MVP.
- `make verify` — tests и synthetic smoke test.
- `make download-dry-run` — offline validation ERA5 request без CDS, сети и файлов.
- `make clean` — удаляет generated synthetic outputs и cache directories, но не raw data.

## 13. Data locations

Raw pair ожидается в `data/raw/era5_single_2024_01_01/`. Storage copy и checksums описаны в `storage-upload/README_DATA.md`. Raw data, outputs, NPZ/Zarr artifacts и checkpoints не коммитятся.

## 14. CDS credentials

`cdsapi` не обязателен для tests, loader или CI. CDS credentials находятся только в `~/.cdsapirc`; не помещайте token в `.env`, `.env.example`, repository или Issue.

## 15. ERA5 download workflow

Единственный downloader — `scripts/download_era5.py`. Перед реальным запросом всегда выполните offline-проверку:

```bash
make download-dry-run
```

Для реальной выгрузки нужен только optional extra `download` и настроенный `~/.cdsapirc`:

```bash
python scripts/download_era5.py \
  --date 2024-01-01 \
  --times 00:00 06:00 12:00 18:00 \
  --output-root data/raw
```

Подробные правила staging, ZIP safety, checksums и overwrite: [DOWNLOAD_WORKFLOW.md](DOWNLOAD_WORKFLOW.md).

## 16. Common problems

- Если `netCDF4` не импортируется, переустановите base environment через `make install`.
- Если `research_template.yaml` не запускается, это ожидаемо: current runner поддерживает только synthetic source. См. [CONFIG_CONTRACT_GAP.md](CONFIG_CONTRACT_GAP.md).
- Не используйте четыре текущих timestamp для обучения, split или создания `tp6h`.

## 17. Range download dry-run

Range downloader обрабатывает дни последовательно и переиспользует safe daily downloader. Offline-проверка не создаёт dataset:

```bash
make download-range-dry-run
```

Реальный range запуск, resume, checksums и seven-day pilot описаны в [RANGE_DOWNLOAD_WORKFLOW.md](RANGE_DOWNLOAD_WORKFLOW.md).

## 18. Clean generated files

```bash
make clean
```

Команда не удаляет raw data.

## 19. Before creating a Pull Request

```bash
make verify
git diff --check
git diff --stat
git status --short
```

Проверьте, что в diff нет secrets, raw data, outputs, checkpoints и unrelated changes.
