# Вклад в ERA5-Minimum

## Начало работы

1. Выберите существующий Issue и договоритесь об одном основном владельце.
2. Создайте ветку от `dev`: `feat/<name>`, `fix/<name>`, `docs/<name>`, `research/<name>` или `infra/<name>`.
3. Передавайте Codex только контекст Issue, ограничения и Definition of Done. Результат Codex всегда проверяет человек.

## Работа с изменениями

Перед Pull Request просмотрите изменения:

```bash
git diff --check
git diff --stat
git status --short
```

Для текущего synthetic MVP используйте только существующие команды:

```bash
make test
make mvp
```

Их эквиваленты:

```bash
pytest -q
python -m era5_minimum.experiments --config configs/mvp.yaml
```

## Pull Request

Один PR решает одну Issue. Откройте PR в `dev`, заполните шаблон и приложите фактический вывод проверок. После integration verification изменения попадают в `main` через squash merge.

Документацию нужно обновлять, если изменяются [data contract](docs/DATA_CONTRACT.md), каналы, units, split, normalization, метрики, конфигурация запуска, архитектурное решение или роли команды. Изменение определения метрик требует записи в `docs/DECISIONS.md`.

## Нельзя коммитить

- raw ERA5, архивы данных и локальные выгрузки;
- generated outputs, checkpoints и большие артефакты;
- `.cdsapirc`, `.env`, токены, ключи и пароли;
- результаты реального CDS download, включая raw pair, archive, request metadata и checksums;
- изменения, не относящиеся к Issue.

## Definition of Done

- у задачи есть один основной владелец;
- код, тесты и документация соответствуют Issue;
- человек проверил Codex diff;
- запущен `make verify`, если изменялся Python-код;
- data contract и metric changes явно отражены в PR;
- нет secrets, raw data, outputs или необъяснённых изменений;
- download changes проверены offline через `make download-dry-run` и tests без CDS credentials;
- все замечания review устранены.
