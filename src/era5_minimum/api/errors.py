"""
Типизированные ошибки API-слоя ERA5-Minimum.

Ровно те шесть категорий, что перечислены в задаче API-001:
    - artifact not found
    - experiment not found
    - invalid artifact
    - unsupported channel
    - timestamp not found
    - malformed JSON

Исключения не содержат ничего, что нельзя отдавать клиенту напрямую
(без traceback, абсолютных путей, переменных окружения, токенов) —
это требование из ARTIFACT_API_CONTRACT.md, раздел 12, и из задачи
(секция "API не должен возвращать пользователю").

Сопоставление исключение -> HTTP-код закреплено в app.py.
"""

from __future__ import annotations


class ArtifactError(Exception):
    """Базовый класс для всех ошибок артефакт-слоя."""


class ArtifactNotFoundError(ArtifactError):
    """Отсутствующий обязательный артефакт (файл не найден на диске)."""

    def __init__(self, artifact_name: str) -> None:
        self.artifact_name = artifact_name
        super().__init__(f"Artifact not found: {artifact_name}")


class ExperimentNotFoundError(ArtifactError):
    """Неизвестный experiment_id -> HTTP 404."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        super().__init__(f"Experiment not found: {experiment_id}")


class InvalidArtifactError(ArtifactError):
    """Артефакт прочитан, но не проходит Pydantic-валидацию/контракт."""

    def __init__(self, artifact_name: str, reason: str) -> None:
        self.artifact_name = artifact_name
        self.reason = reason
        super().__init__(f"Invalid artifact '{artifact_name}': {reason}")


class UnsupportedChannelError(ArtifactError):
    """Канал не входит в канонический список -> HTTP 422."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        super().__init__(f"Unsupported channel: {channel}")


class TimestampNotFoundError(ArtifactError):
    """Запрошенный timestamp отсутствует в артефакте -> HTTP 422."""

    def __init__(self, timestamp: str) -> None:
        self.timestamp = timestamp
        super().__init__(f"Timestamp not found: {timestamp}")


class MalformedJSONError(ArtifactError):
    """Файл существует, но не парсится как JSON."""

    def __init__(self, artifact_name: str) -> None:
        self.artifact_name = artifact_name
        super().__init__(f"Malformed JSON in artifact: {artifact_name}")
