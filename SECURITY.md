# Security policy

## Credentials and secrets

- CDS credentials существуют только в личном `~/.cdsapirc`; downloader передаёт их стандартному `cdsapi` и не читает файл вручную.
- GitHub tokens, API keys, passwords и private keys не хранятся в project files, `.env.example`, reports, manifests, Issues или PRs.
- Нельзя commit `.env`, `.cdsapirc`, keys, credentials, raw data, generated outputs или checkpoints.
- Если secret попал в working tree, не добавляйте его в Git, прекратите распространение, сообщите владельцу repository через приватный канал команды и немедленно ротируйте утёкший token. Не публикуйте secret в Issue, чат или PR.

## Data and privacy

- Raw ERA5 и download archives хранятся вне Git и передаются только через approved Team Workspace с checksum verification.
- Не публикуйте персональные данные, user paths, environment dumps или CDS account identifiers в artifacts и документации.

## Codex and command safety

- Человек проверяет каждый Codex-generated diff до merge.
- Не передавайте Codex credentials и не выполняйте неизвестные destructive commands.
- Перед удалением или overwrite подтвердите точную цель; downloader заменяет raw day только с явным `--overwrite` и staging.

## Archive and dependency safety

- ZIP extraction должна отклонять absolute paths, traversal и symbolic links; используйте только безопасную extraction policy downloader.
- Новая dependency требует необходимости, документации, review `pyproject.toml` и проверки CI. Не добавляйте SDK или сервис без отдельной задачи.

## Reporting

Для security report используйте приватный канал владельца repository. Не указывайте реальные email-адреса или sensitive details в публичных материалах.
