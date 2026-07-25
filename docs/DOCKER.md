# Запуск проекта в Docker

## Требования

Нужны Docker Engine и Docker Compose v2. Для команд `gpu-*` дополнительно
нужны совместимые NVIDIA driver и NVIDIA Container Toolkit.

CPU-образ использует Python 3.12, как CI, и зависимости из `pyproject.toml`.
Данные, контрольные точки моделей, битовые потоки и outputs не попадают в образ:
они подключаются как локальные bind mount.

## Первый запуск

```bash
cp .env.example .env
make docker-config
make docker-build
make app-up
```

Проверка API:

```bash
make app-health
make app-logs
```

- frontend: <http://localhost:5173>
- API: <http://localhost:8000>
- OpenAPI: <http://localhost:8000/docs>
- метрики Prometheus: <http://localhost:8000/metrics>

Остановка: `make app-down`.

## Модель N32

ZIP с моделью не добавляется в Git. Поместите полученный архив в корень
репозитория и один раз выполните:

```bash
make model-unpack
```

Команда распакует модель в игнорируемый `artifacts/model-n32`. API контейнер
подключает эту директорию только для чтения. После запуска проверьте:

```bash
curl http://localhost:8000/api/v1/codec/status
```

## Подключаемые каталоги

| Переменная окружения | Путь в контейнере | Назначение |
| --- | --- | --- |
| `DATA_DIR` | `/workspace/data` | исходные и подготовленные данные |
| `OUTPUTS_DIR` | `/workspace/outputs` | результаты запусков |
| `CHECKPOINTS_DIR` | `/workspace/checkpoints` | локальные контрольные точки |
| `BITSTREAMS_DIR` | `/workspace/bitstreams` | сгенерированные битовые потоки |
| `ARTIFACTS_DIR` | `/workspace/artifacts` | артефакты, включая распакованную модель |
| `SUBMISSION_DIR` | `/workspace/submission` | материалы сдачи |

Не добавляйте содержимое этих каталогов в Git.

## Проверка контейнера

```bash
make docker-config
make verify-container
```

Конкурсные лимиты — одна GPU, до 20 млн параметров, до 50 000 шагов, до 24 ГиБ
VRAM и до 48 GPU-часов — Docker сам не ограничивает. Они должны быть измерены и
сохранены в артефактах эксперимента.
