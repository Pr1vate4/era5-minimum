# GitHub workflow

## Ветки

- `main` — стабильное состояние.
- `dev` — интеграционная разработка.
- Рабочие ветки: `feat/<name>`, `fix/<name>`, `docs/<name>`, `research/<name>`, `infra/<name>`.

## Процесс

`Issue → branch → Codex implementation → human review → local verification → Pull Request into dev → integration verification → merge into main`.

## Правила

1. Прямой push в `main` запрещён.
2. Один PR решает одну задачу.
3. Код Codex всегда проверяется человеком.
4. PR без тестов не объединяется.
5. Raw data, generated outputs и checkpoints не коммитятся.
6. Изменение определения метрик требует обновить `docs/DECISIONS.md`.
7. Изменение [data contract](DATA_CONTRACT.md) требует review от @Pr1vate4 и @VsevolodCod.
8. Предпочтителен squash merge.
9. Commit messages используют Conventional Commits.
10. Нельзя объединять PR с неизвестными или необъяснёнными изменениями.
11. Dependency changes обновляются сначала в `pyproject.toml`; `requirements.txt` остаётся compatibility-file.
