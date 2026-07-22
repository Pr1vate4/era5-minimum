# Team onboarding — 20 minutes

## 1. Проект за одну минуту

ERA5-Minimum исследует минимальный репрезентативный объём ERA5 для нейросетевого сжатия 32–64×. Сейчас готова data foundation и synthetic MVP; честное обучение на реальных данных ещё запрещено, потому что есть лишь четыре sparse timestamps. Факты и blockers: [CURRENT_STATE.md](CURRENT_STATE.md).

## 2. Роли

- @VsevolodCod — ML Lead.
- @Pr1vate4 — Data Platform / Infrastructure / DevOps.
- @devlifeee — Frontend / Product.

Подробный ownership: [TEAM_ROLES.md](TEAM_ROLES.md).

## 3–5. Clone, Python 3.12 и установка

```bash
git clone https://github.com/Pr1vate4/era5-minimum.git
cd era5-minimum
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-dev
```

Подробности и troubleshooting: [DEVELOPMENT.md](DEVELOPMENT.md).

## 6. Tests

```bash
make test
```

## 7. Synthetic smoke

```bash
make smoke
```

Это не real-data training. Общая локальная проверка — `make verify`.

## 8. ERA5 downloader dry-run

```bash
make download-dry-run
```

Команда offline: ей не нужны CDS credentials, сеть или файлы. Real download workflow: [DOWNLOAD_WORKFLOW.md](DOWNLOAD_WORKFLOW.md).

## 9. Где данные

Raw pair живёт локально в `data/raw/era5_single_<date>/`; verified shared copy — только в approved Team Workspace. Raw data не входят в Git. Schema, names и units определяет только [DATA_CONTRACT.md](DATA_CONTRACT.md).

## 10. Где artifacts

Generated outputs, checkpoints, prepared datasets и raw downloads хранятся вне Git: локально в ignored paths либо в approved Team Workspace. Перед upload raw pair проверьте `SHA256SUMS.txt`; подробности: [storage-upload/README_DATA.md](../storage-upload/README_DATA.md).

## 11. Git workflow

Создайте Issue, назначьте одного owner, ответвитесь от `dev`, выполните local verification и откройте PR в `dev`. Имена веток: `feat/`, `fix/`, `docs/`, `research/`, `infra/`. Полный процесс: [WORKFLOW.md](WORKFLOW.md).

## 12. Работа с Codex

Дайте Codex одну Issue: goal, context, constraints и Definition of Done. Проверьте diff человеком, не передавайте credentials и не запускайте непонятные destructive commands. Постоянные правила: [AGENTS.md](../AGENTS.md).

## 13. Pull Request

Заполните PR template, приложите фактический вывод проверок и убедитесь, что нет unrelated changes. Правила: [CONTRIBUTING.md](../CONTRIBUTING.md).

## 14. Запрещённые файлы

Не коммитить raw ERA5, архивы, outputs, checkpoints, `.env`, `.cdsapirc`, keys, tokens и credentials. Полный перечень: [SECURITY.md](../SECURITY.md).

## 15. Первая задача каждого участника

- @Pr1vate4: DATA-002 — получить verified sequential hourly ERA5 range.
- @VsevolodCod: ML-001 — подготовить PCA baseline после принятия DATA-003; до этого review data/metric acceptance criteria.
- @devlifeee: FE-001 — React dashboard на mock JSON, параллельно с data work.

Backlog и зависимости: [TASKS.md](TASKS.md).

## 16. Checklist

- [ ] Python 3.12 environment создан.
- [ ] `make test` и `make smoke` запущены.
- [ ] `make download-dry-run` понятен и не создал файлы.
- [ ] Прочитаны DATA_CONTRACT, WORKFLOW, SECURITY и первая назначенная задача.
- [ ] Создана или выбрана Issue с одним owner.
