from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_latent_probe_script_reads_metrics(tmp_path: Path) -> None:
    payload = {
        "train_pair_count": 8,
        "validation_pair_count": 4,
        "optimizer_steps": 12,
        "relative_improvement_vs_persistence_pct": 5.5,
        "forecast_horizon_hours": 6,
    }
    metrics_path = tmp_path / "probe_metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_latent_probe.py",
            "--metrics",
            str(metrics_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["optimizer_steps"] == 12
    assert output["forecast_horizon_hours"] == 6
