"""
Слой чтения артефактов с диска.

По заданию (API-001) обязан:
    - открывать JSON-файлы;
    - валидировать данные через Pydantic;
    - возвращать список экспериментов;
    - находить эксперимент по ID;
    - возвращать reconstruction по experiment ID, channel и timestamp;
    - корректно обрабатывать отсутствующие и повреждённые файлы.

Это единственный модуль, который трогает файловую систему. app.py
никогда не открывает файлы напрямую — только через ArtifactRepository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import ValidationError

from era5_minimum.api.errors import (
    ArtifactNotFoundError,
    ExperimentNotFoundError,
    InvalidArtifactError,
    MalformedJSONError,
    TimestampNotFoundError,
    UnsupportedChannelError,
)
from era5_minimum.api.schemas import (
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)


def _read_json(path: Path, artifact_name: str) -> dict | list:
    if not path.exists():
        raise ArtifactNotFoundError(artifact_name)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise MalformedJSONError(artifact_name) from exc


def _raise_invalid(exc: ValidationError, artifact_name: str) -> None:
    raise InvalidArtifactError(artifact_name, str(exc)) from exc


class ArtifactRepository:
    """
    Читает и валидирует набор артефактов из каталога вида demo/mock/.

    Файлы читаются заново на каждый вызов — набор артефактов небольшой
    и статичный, кеширование не требуется на этом этапе (при
    необходимости добавляется здесь, без изменений в app.py).
    """

    def __init__(self, artifacts_root: Path) -> None:
        self.root = Path(artifacts_root)
        self.summary_file = self.root / "summary.json"
        self.experiments_file = self.root / "experiments.json"
        self.sample_efficiency_file = self.root / "sample_efficiency.json"
        self.reconstructions_dir = self.root / "reconstructions"

    # -- summary.json --------------------------------------------------
    def get_summary(self) -> SummaryArtifact:
        raw = _read_json(self.summary_file, "summary.json")
        try:
            return SummaryArtifact.model_validate(raw)
        except ValidationError as exc:
            _raise_invalid(exc, "summary.json")

    # -- experiments.json ------------------------------------------------
    def list_experiments(self) -> List[ExperimentArtifact]:
        """
        Возвращает список экспериментов.

        experiments.json допускается как в виде одного объекта (пример
        в контракте), так и в виде списка объектов — оба варианта
        нормализуются в список экспериментов.
        """
        raw = _read_json(self.experiments_file, "experiments.json")
        items = raw if isinstance(raw, list) else [raw]
        try:
            return [ExperimentArtifact.model_validate(item) for item in items]
        except ValidationError as exc:
            _raise_invalid(exc, "experiments.json")

    def get_experiment(self, experiment_id: str) -> ExperimentArtifact:
        for experiment in self.list_experiments():
            if experiment.id == experiment_id:
                return experiment
        raise ExperimentNotFoundError(experiment_id)

    # -- sample_efficiency.json --------------------------------------------
    def get_sample_efficiency(self) -> SampleEfficiencyArtifact:
        raw = _read_json(
            self.sample_efficiency_file, "sample_efficiency.json"
        )
        try:
            return SampleEfficiencyArtifact.model_validate(raw)
        except ValidationError as exc:
            _raise_invalid(exc, "sample_efficiency.json")

    # -- reconstructions/<experiment_id>.json --------------------------------
    def get_reconstruction(
        self,
        experiment_id: str,
        channel: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> ReconstructionArtifact:
        """
        Возвращает reconstruction по experiment_id, опционально
        отфильтрованный по channel и timestamp.

        Порядок проверок важен для корректных кодов ошибок:
          1. experiment_id должен существовать в experiments.json (404,
             если нет — иначе неизвестный experiment дал бы 500);
          2. файл reconstructions/<experiment_id>.json должен
             существовать и быть валиден (иначе 500);
          3. channel, если передан, должен быть каноническим и совпадать
             с тем, что реально в файле (иначе 422);
          4. timestamp, если передан, должен совпадать с меткой в файле
             (иначе 422).
        """
        self.get_experiment(experiment_id)

        artifact_name = f"reconstructions/{experiment_id}.json"
        path = self.reconstructions_dir / f"{experiment_id}.json"
        raw = _read_json(path, artifact_name)

        try:
            reconstruction = ReconstructionArtifact.model_validate(raw)
        except ValidationError as exc:
            _raise_invalid(exc, artifact_name)

        if reconstruction.experiment_id != experiment_id:
            raise InvalidArtifactError(
                artifact_name,
                "experiment_id inside the file does not match the "
                "requested experiment_id",
            )

        if channel is not None and reconstruction.channel != channel:
            raise UnsupportedChannelError(channel)

        if timestamp is not None:
            actual_iso = reconstruction.timestamp.isoformat().replace(
                "+00:00", "Z"
            )
            if timestamp not in (actual_iso, str(reconstruction.timestamp)):
                raise TimestampNotFoundError(timestamp)

        return reconstruction
