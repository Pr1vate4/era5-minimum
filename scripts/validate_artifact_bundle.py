"""
Artifact bundle validator.

Использование:

    python scripts/validate_artifact_bundle.py demo/mock

Проверяет каталог с артефактами (summary.json, experiments.json,
sample_efficiency.json, reconstructions/*.json) на соответствие
docs/ARTIFACT_API_CONTRACT.md. Выполняет ровно 12 проверок,
перечисленных в задаче API-001:

    1.  наличие обязательных файлов;
    2.  корректность JSON;
    3.  Pydantic-валидация;
    4.  согласованность experiment ID между файлами;
    5.  использование только канонических каналов;
    6.  наличие и корректность units;
    7.  совпадение форм карт;
    8.  соответствие latitude/longitude размеру карт;
    9.  наличие is_demo=true у mock-артефактов;
    10. отсутствие tp6h (и других запрещённых каналов);
    11. существование experiment ID, на которые ссылаются другие файлы;
    12. корректность sample-efficiency структуры.

exit code 0  — bundle валиден;
exit code 1  — найдена ошибка (все ошибки печатаются).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Позволяет запускать скрипт как `python scripts/validate_artifact_bundle.py`
# без предварительной установки пакета era5_minimum.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pydantic import ValidationError  # noqa: E402

from era5_minimum.api.schemas import (  # noqa: E402
    CANONICAL_CHANNEL_UNITS,
    FORBIDDEN_CHANNELS,
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)

class BundleValidator:
    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        self.errors: list[str] = []
        self._experiment_ids: set[str] = set()

    # -- helpers ------------------------------------------------------------
    def _fail(self, message: str) -> None:
        self.errors.append(message)

    def _load_json(self, relative_path: str) -> dict | list | None:
        """Проверки 1 и 2: наличие файла и корректность JSON."""
        path = self.bundle_dir / relative_path
        if not path.exists():
            self._fail(f"[missing-file] {relative_path} does not exist")
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            self._fail(f"[malformed-json] {relative_path}: {exc}")
            return None

    # -- checks ---------------------------------------------------------
    def check_summary(self) -> None:
        raw = self._load_json("summary.json")
        if raw is None:
            return
        try:
            summary = SummaryArtifact.model_validate(raw)
        except ValidationError as exc:
            self._fail(f"[schema] summary.json: {exc}")
            return
        # Проверка 9: is_demo=true у mock-артефактов.
        if not summary.is_demo:
            self._fail("[is-demo] summary.json: is_demo must be true")

    def check_experiments(self) -> None:
        raw = self._load_json("experiments.json")
        if raw is None:
            return
        items = raw if isinstance(raw, list) else [raw]
        for index, item in enumerate(items):
            try:
                experiment = ExperimentArtifact.model_validate(item)
            except ValidationError as exc:
                self._fail(f"[schema] experiments.json[{index}]: {exc}")
                continue
            if not experiment.is_demo:
                self._fail(
                    f"[is-demo] experiments.json[{index}] "
                    f"('{experiment.id}'): is_demo must be true"
                )
            self._experiment_ids.add(experiment.id)

    def check_sample_efficiency(self) -> None:
        """Проверка 12: корректность sample-efficiency структуры."""
        raw = self._load_json("sample_efficiency.json")
        if raw is None:
            return
        try:
            sample_efficiency = SampleEfficiencyArtifact.model_validate(raw)
        except ValidationError as exc:
            self._fail(f"[schema] sample_efficiency.json: {exc}")
            return
        if not sample_efficiency.is_demo:
            self._fail(
                "[is-demo] sample_efficiency.json: is_demo must be true"
            )
        if not sample_efficiency.points:
            self._fail(
                "[sample-efficiency] sample_efficiency.json: "
                "points must not be empty"
            )
        sample_counts = [p.sample_count for p in sample_efficiency.points]
        if any(count <= 0 for count in sample_counts):
            self._fail(
                "[sample-efficiency] sample_efficiency.json: "
                "sample_count values must be positive"
            )
        if len(sample_counts) != len(set(sample_counts)):
            self._fail(
                "[sample-efficiency] sample_efficiency.json: "
                "duplicate sample_count values found"
            )

    def check_reconstructions(self) -> None:
        recon_dir = self.bundle_dir / "reconstructions"
        if not recon_dir.exists():
            self._fail("[missing-file] reconstructions/ does not exist")
            return

        recon_files = sorted(recon_dir.glob("*.json"))
        if not recon_files:
            self._fail(
                "[missing-file] reconstructions/ contains no artifact files"
            )
            return

        for path in recon_files:
            relative_path = f"reconstructions/{path.name}"
            try:
                with path.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except json.JSONDecodeError as exc:
                self._fail(f"[malformed-json] {relative_path}: {exc}")
                continue

            try:
                reconstruction = ReconstructionArtifact.model_validate(raw)
            except ValidationError as exc:
                # Pydantic-валидаторы в схеме уже покрывают проверки 5-8
                # (канон канала, units, формы матриц, latitude/longitude),
                # поэтому все они всплывают здесь как одна ошибка схемы.
                self._fail(f"[schema] {relative_path}: {exc}")
                continue

            # Проверка 9.
            if not reconstruction.is_demo:
                self._fail(
                    f"[is-demo] {relative_path}: is_demo must be true"
                )

            # Проверка 10: явный запрет tp6h / mslp, даже если бы схема
            # почему-то их пропустила.
            if reconstruction.channel in FORBIDDEN_CHANNELS:
                self._fail(
                    f"[forbidden-channel] {relative_path}: channel "
                    f"'{reconstruction.channel}' is forbidden"
                )

            # Проверка 5-6 продублирована здесь явно (не только внутри
            # схемы), чтобы валидатор давал понятную причину даже если
            # схема в будущем изменится.
            if reconstruction.channel not in CANONICAL_CHANNEL_UNITS:
                self._fail(
                    f"[unknown-channel] {relative_path}: channel "
                    f"'{reconstruction.channel}' is not canonical"
                )
            else:
                expected_units = CANONICAL_CHANNEL_UNITS[
                    reconstruction.channel
                ]
                if reconstruction.units != expected_units:
                    self._fail(
                        f"[units-mismatch] {relative_path}: units "
                        f"'{reconstruction.units}' != expected "
                        f"'{expected_units}' for channel "
                        f"'{reconstruction.channel}'"
                    )

            # Проверка 4 и 11: experiment_id должен существовать среди
            # экспериментов из experiments.json.
            if reconstruction.experiment_id not in self._experiment_ids:
                self._fail(
                    f"[unknown-experiment-id] {relative_path}: "
                    f"experiment_id '{reconstruction.experiment_id}' is "
                    "not present in experiments.json"
                )

            # Имя файла должно совпадать с experiment_id внутри него —
            # ещё один аспект согласованности experiment ID (проверка 4).
            if path.stem != reconstruction.experiment_id:
                self._fail(
                    f"[filename-mismatch] {relative_path}: file name "
                    f"does not match experiment_id "
                    f"'{reconstruction.experiment_id}'"
                )

    def run(self) -> int:
        self.check_summary()
        self.check_experiments()
        self.check_sample_efficiency()
        self.check_reconstructions()

        if self.errors:
            print(
                f"Artifact bundle validation FAILED "
                f"({len(self.errors)} error(s)):",
                file=sys.stderr,
            )
            for error in self.errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

        print("Artifact bundle validation OK")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an ERA5-Minimum artifact bundle directory."
    )
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Path to a directory such as demo/mock/",
    )
    args = parser.parse_args()

    bundle_dir: Path = args.bundle_dir
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        print(
            "Artifact bundle validation FAILED (1 error(s)):",
            file=sys.stderr,
        )
        print(
            f"  - [missing-directory] {bundle_dir} is not a directory",
            file=sys.stderr,
        )
        return 1

    validator = BundleValidator(bundle_dir)
    return validator.run()


if __name__ == "__main__":
    raise SystemExit(main())
