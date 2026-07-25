from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from era5_minimum.codec.workflow import run_codec_smoke


def _run_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "codec_cli"
    run_codec_smoke(
        {
            "seed": 31,
            "output_dir": str(output_dir),
            "data": {
                "samples": 12,
                "height": 8,
                "width": 8,
                "validation_samples": 2,
                "test_samples": 2,
            },
            "model": {"latent_channels": 4, "parameter_limit": 2_000_000},
            "codec": {
                "version": "ml-001",
                "grid": "smoke-8x8",
                "quantization_step": 0.25,
                "target_compression_ratio": 32,
            },
            "training": {
                "batch_size": 4,
                "epochs": 1,
                "learning_rate": 1e-3,
                "max_steps": 1,
            },
            "resources": {"max_vram_gb": 24, "max_gpu_hours": 48},
        }
    )
    return output_dir


def _invoke(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_codec_local.py",
            "--run-dir",
            str(run_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_local_evaluator_cli_reads_scientific_report_without_legacy_files(
    tmp_path: Path,
) -> None:
    output_dir = _run_dir(tmp_path)
    (output_dir / "run_summary.json").unlink()
    (output_dir / "metrics_validation.json").unlink()

    result = _invoke(output_dir)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "codec-smoke-31"
    assert payload["actual_serialized_compression_ratio"] > 0.0
    assert payload["exact_roundtrip"] is True
    assert payload["overall_score"] >= 0.0
    assert payload["mean_finite_psnr_db"] is not None
    assert payload["physical_diagnostics"]["mslp_hpa"] >= 0.0
    assert payload["physical_diagnostics"]["tp6h_mm_per_6h"] >= 0.0
    assert payload["physical_diagnostics"]["wind_speed_m_s"] >= 0.0
    assert "preliminary" in payload["limitation"].lower()


def test_local_evaluator_cli_rejects_missing_provenance(tmp_path: Path) -> None:
    output_dir = _run_dir(tmp_path)
    report_path = output_dir / "local_evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["provenance"].pop("git_commit", None)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _invoke(output_dir)

    assert result.returncode != 0
    assert "provenance" in result.stderr.lower()
