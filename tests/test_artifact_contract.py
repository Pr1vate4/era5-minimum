"""
Тесты контракта артефактов: Pydantic-схемы (schemas.py) и слой чтения
(repository.py), без HTTP. Все данные — временные JSON-файлы в
tmp_path, без сети и реальных ERA5/NetCDF данных.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from era5_minimum.api.errors import (
    ArtifactNotFoundError,
    ExperimentNotFoundError,
    InvalidArtifactError,
    MalformedJSONError,
    TimestampNotFoundError,
    UnsupportedChannelError,
)
from era5_minimum.api.repository import ArtifactRepository
from era5_minimum.api.schemas import (
    ExperimentArtifact,
    ReconstructionArtifact,
    SampleEfficiencyArtifact,
    SummaryArtifact,
)

VALID_RECONSTRUCTION = {
    "experiment_id": "demo-pca-001",
    "channel": "t2m",
    "units": "K",
    "timestamp": "2024-01-02T12:00:00Z",
    "latitude": [50.0, 51.0],
    "longitude": [10.0, 11.0, 12.0],
    "original": [[288.1, 288.4, 288.9], [287.6, 287.9, 288.2]],
    "reconstruction": [[288.0, 288.5, 288.8], [287.7, 287.8, 288.3]],
    "absolute_error": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    "is_demo": True,
}


# ---------------------------------------------------------------------------
# Схемы: summary, experiment, sample_efficiency — happy path.
# ---------------------------------------------------------------------------
def test_summary_schema_valid() -> None:
    summary = SummaryArtifact.model_validate(
        {
            "is_demo": True,
            "experiment_count": 1,
            "completed": 1,
            "failed": 0,
            "generated_at": "2024-01-02T15:00:00Z",
        }
    )
    assert summary.experiment_count == 1


def test_experiment_schema_valid() -> None:
    experiment = ExperimentArtifact.model_validate(
        {
            "id": "demo-pca-001",
            "name": "Demo PCA baseline",
            "model": "pca",
            "status": "completed",
            "is_demo": True,
            "compression": {"type": "tensor_ratio", "ratio": 32},
            "training": {
                "sample_count": 168,
                "selection_strategy": "contiguous",
            },
            "metrics": {"normalized_rmse": 0.12, "mae": 0.08},
        }
    )
    assert experiment.compression.ratio == 32
    assert experiment.training.sample_count == 168


def test_sample_efficiency_schema_valid() -> None:
    artifact = SampleEfficiencyArtifact.model_validate(
        {
            "is_demo": True,
            "metric": "normalized_rmse",
            "points": [{"sample_count": 64, "value": 0.21}],
        }
    )
    assert artifact.points[0].sample_count == 64


def test_sample_efficiency_rejects_empty_points() -> None:
    with pytest.raises(ValidationError):
        SampleEfficiencyArtifact.model_validate(
            {"is_demo": True, "metric": "normalized_rmse", "points": []}
        )


# ---------------------------------------------------------------------------
# Reconstruction schema: канал, units, формы карт, is_demo.
# ---------------------------------------------------------------------------
def test_reconstruction_schema_valid() -> None:
    artifact = ReconstructionArtifact.model_validate(VALID_RECONSTRUCTION)
    assert artifact.channel == "t2m"


def test_reconstruction_rejects_unknown_channel() -> None:
    payload = dict(VALID_RECONSTRUCTION, channel="unknown_channel")
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_forbidden_channel_mslp() -> None:
    payload = dict(VALID_RECONSTRUCTION, channel="mslp", units="Pa")
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_forbidden_channel_tp6h() -> None:
    payload = dict(VALID_RECONSTRUCTION, channel="tp6h", units="m")
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_wrong_units_for_channel() -> None:
    payload = dict(VALID_RECONSTRUCTION, units="Pa")  # t2m требует K
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_mismatched_matrix_shapes() -> None:
    payload = dict(VALID_RECONSTRUCTION)
    payload["reconstruction"] = [[1.0, 2.0]]  # форма 1x2 вместо 2x3
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_latitude_length_mismatch() -> None:
    payload = dict(VALID_RECONSTRUCTION, latitude=[50.0])  # ожидается 2
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_rejects_longitude_length_mismatch() -> None:
    payload = dict(VALID_RECONSTRUCTION, longitude=[10.0, 11.0])  # ожид. 3
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


def test_reconstruction_requires_is_demo_true() -> None:
    payload = dict(VALID_RECONSTRUCTION, is_demo=False)
    with pytest.raises(ValidationError):
        ReconstructionArtifact.model_validate(payload)


# ---------------------------------------------------------------------------
# Repository: чтение с диска, поиск эксперимента, обработка ошибок.
# ---------------------------------------------------------------------------
@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "reconstructions").mkdir(parents=True)

    (root / "summary.json").write_text(
        json.dumps(
            {
                "is_demo": True,
                "experiment_count": 1,
                "completed": 1,
                "failed": 0,
                "generated_at": "2024-01-02T15:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (root / "experiments.json").write_text(
        json.dumps(
            {
                "id": "demo-pca-001",
                "name": "Demo PCA baseline",
                "model": "pca",
                "status": "completed",
                "is_demo": True,
                "compression": {"type": "tensor_ratio", "ratio": 32},
                "training": {
                    "sample_count": 168,
                    "selection_strategy": "contiguous",
                },
                "metrics": {"normalized_rmse": 0.12, "mae": 0.08},
            }
        ),
        encoding="utf-8",
    )
    (root / "sample_efficiency.json").write_text(
        json.dumps(
            {
                "is_demo": True,
                "metric": "normalized_rmse",
                "points": [{"sample_count": 64, "value": 0.21}],
            }
        ),
        encoding="utf-8",
    )
    (root / "reconstructions" / "demo-pca-001.json").write_text(
        json.dumps(VALID_RECONSTRUCTION), encoding="utf-8"
    )
    return root


def test_repository_get_summary(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    summary = repo.get_summary()
    assert summary.experiment_count == 1


def test_repository_list_experiments(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    experiments = repo.list_experiments()
    assert len(experiments) == 1
    assert experiments[0].id == "demo-pca-001"


def test_repository_get_experiment_found(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    experiment = repo.get_experiment("demo-pca-001")
    assert experiment.model == "pca"


def test_repository_get_experiment_not_found(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(ExperimentNotFoundError):
        repo.get_experiment("does-not-exist")


def test_repository_get_reconstruction_ok(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    reconstruction = repo.get_reconstruction(
        "demo-pca-001", channel="t2m", timestamp="2024-01-02T12:00:00Z"
    )
    assert reconstruction.channel == "t2m"


def test_repository_get_reconstruction_bad_channel(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(UnsupportedChannelError):
        repo.get_reconstruction("demo-pca-001", channel="msl")


def test_repository_get_reconstruction_bad_timestamp(bundle: Path) -> None:
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(TimestampNotFoundError):
        repo.get_reconstruction(
            "demo-pca-001", timestamp="2099-01-01T00:00:00Z"
        )


def test_repository_missing_artifact_file(bundle: Path) -> None:
    (bundle / "summary.json").unlink()
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(ArtifactNotFoundError):
        repo.get_summary()


def test_repository_malformed_json(bundle: Path) -> None:
    (bundle / "summary.json").write_text("{not valid json", encoding="utf-8")
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(MalformedJSONError):
        repo.get_summary()


def test_repository_invalid_schema(bundle: Path) -> None:
    (bundle / "summary.json").write_text(
        json.dumps({"is_demo": True}), encoding="utf-8"
    )
    repo = ArtifactRepository(artifacts_root=bundle)
    with pytest.raises(InvalidArtifactError):
        repo.get_summary()


# ---------------------------------------------------------------------------
# scripts/validate_artifact_bundle.py: exit codes и вывод ошибок.
# ---------------------------------------------------------------------------
def _run_validator(bundle_dir: Path) -> subprocess.CompletedProcess:
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "validate_artifact_bundle.py"
    )
    return subprocess.run(
        [sys.executable, str(script), str(bundle_dir)],
        capture_output=True,
        text=True,
    )


def test_validator_exit_code_zero_for_valid_bundle(bundle: Path) -> None:
    result = _run_validator(bundle)
    assert result.returncode == 0
    assert "OK" in result.stdout


def test_validator_exit_code_one_for_missing_file(bundle: Path) -> None:
    (bundle / "summary.json").unlink()
    result = _run_validator(bundle)
    assert result.returncode == 1
    assert "summary.json" in result.stderr


def test_validator_exit_code_one_for_forbidden_channel(bundle: Path) -> None:
    broken = dict(VALID_RECONSTRUCTION, channel="tp6h", units="m")
    (bundle / "reconstructions" / "demo-pca-001.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )
    result = _run_validator(bundle)
    assert result.returncode == 1


def test_validator_exit_code_one_for_unknown_experiment_reference(
    bundle: Path,
) -> None:
    broken = dict(VALID_RECONSTRUCTION, experiment_id="ghost-experiment")
    (bundle / "reconstructions" / "ghost-experiment.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )
    result = _run_validator(bundle)
    assert result.returncode == 1
    assert "unknown-experiment-id" in result.stderr


def test_validator_reports_missing_directory() -> None:
    result = _run_validator(Path("/definitely/does/not/exist"))
    assert result.returncode == 1
