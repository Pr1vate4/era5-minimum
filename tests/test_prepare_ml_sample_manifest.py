from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_prepare_ml_sample_manifest_is_deterministic(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "prepare_ml_sample_manifest.py"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    for output in (first, second):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output",
                str(output),
                "--seed",
                "42",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(result.stdout)
        assert summary["manifest_sha256"]
        assert summary["train_only"] is True

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert sorted(payload["subsets"], key=int) == ["16", "32", "64", "128"]
    assert payload["subsets"]["16"]["parent_size"] is None
    assert payload["subsets"]["128"]["parent_size"] == 64
    assert payload["temporal_embargo_hours"] == 168
    assert payload["seed"] == 42
    assert payload["integrity"]["manifest_sha256"] == json.loads(second.read_text())[
        "integrity"
    ]["manifest_sha256"]
