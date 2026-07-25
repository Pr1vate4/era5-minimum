from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Keep the repository CLI runnable before an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from era5_minimum.codec.evaluation import EVALUATOR_VERSION


def load_local_evaluation(run_dir: Path) -> dict[str, Any]:
    """Load and validate the versioned scientific evaluator artifact."""

    path = Path(run_dir) / "local_evaluation.json"
    if not path.exists():
        raise ValueError(f"local evaluator artifact is missing: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read local evaluator artifact: {error}") from error
    if not isinstance(report, dict):
        raise ValueError("local evaluator artifact must be a JSON object")
    if report.get("evaluator_version") != EVALUATOR_VERSION:
        raise ValueError(
            f"unsupported evaluator_version: {report.get('evaluator_version')!r}; "
            f"expected {EVALUATOR_VERSION!r}"
        )
    if report.get("train_only") is not True:
        raise ValueError("local evaluator report must declare train_only=true")

    required = {
        "run_id",
        "channel_order",
        "per_channel",
        "groups",
        "physical_diagnostics",
        "train_statistics",
        "compression",
        "exact_roundtrip",
        "timings",
        "provenance",
        "limitations",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"local evaluator report is missing required fields: {missing}")
    if len(report["channel_order"]) != 28 or len(report["per_channel"]) != 28:
        raise ValueError("local evaluator report must contain exactly 28 channels")
    if not isinstance(report["limitations"], list) or not report["limitations"]:
        raise ValueError("local evaluator report must contain a limitation")
    if not all(isinstance(value, str) and value for value in report["limitations"]):
        raise ValueError("local evaluator limitations must be non-empty strings")

    train_statistics = report["train_statistics"]
    if (
        not isinstance(train_statistics, dict)
        or train_statistics.get("train_only") is not True
        or not isinstance(train_statistics.get("checksum"), str)
        or len(train_statistics["checksum"]) != 64
    ):
        raise ValueError("train-only statistics provenance is missing or invalid")

    provenance = report["provenance"]
    if (
        not isinstance(provenance, dict)
        or not isinstance(provenance.get("git_commit"), str)
        or len(provenance["git_commit"]) != 40
        or not isinstance(provenance.get("run_id"), str)
        or not isinstance(provenance.get("seed"), int)
        or provenance.get("normalization_train_only") is not True
    ):
        raise ValueError("provenance is missing or incomplete")

    for group in ("surface", "pressure", "overall"):
        if group not in report["groups"]:
            raise ValueError(f"local evaluator report is missing groups.{group}")
    for diagnostic in ("mslp", "tp6h", "geopotential", "u10", "v10", "wind_speed"):
        if diagnostic not in report["physical_diagnostics"]:
            raise ValueError(
                f"local evaluator report is missing physical_diagnostics.{diagnostic}"
            )
    for timing in ("encode_seconds", "decode_seconds"):
        value = report["timings"].get(timing)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"timings.{timing} must be finite")
    compression = report["compression"]
    if not math.isfinite(float(compression["actual_serialized_compression_ratio"])):
        raise ValueError("actual serialized compression ratio must be finite")
    if report["exact_roundtrip"] is not True:
        raise ValueError("exact_roundtrip must be true for a valid local evaluator report")
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"local evaluator report is not JSON-safe: {error}") from error
    return report


def build_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact CLI summary without recomputing train statistics."""

    diagnostics = report["physical_diagnostics"]
    compression = report["compression"]
    finite_psnr = [
        float(row["psnr_db"])
        for row in report["per_channel"]
        if row.get("psnr_db") is not None and math.isfinite(float(row["psnr_db"]))
    ]
    return {
        "evaluator_version": report["evaluator_version"],
        "run_id": report["run_id"],
        "actual_serialized_compression_ratio": compression[
            "actual_serialized_compression_ratio"
        ],
        "tensor_compression_ratio": compression["tensor_compression_ratio"],
        "exact_roundtrip": report["exact_roundtrip"],
        "encode_seconds": report["timings"]["encode_seconds"],
        "decode_seconds": report["timings"]["decode_seconds"],
        "surface_score": report["groups"]["surface"]["nrmse"],
        "pressure_score": report["groups"]["pressure"]["nrmse"],
        "overall_score": report["groups"]["overall"]["nrmse"],
        "mean_finite_psnr_db": (
            sum(finite_psnr) / len(finite_psnr) if finite_psnr else None
        ),
        "physical_diagnostics": {
            "mslp_hpa": diagnostics["mslp"]["rmse_hpa"],
            "tp6h_mm_per_6h": diagnostics["tp6h"]["rmse_mm_per_6h"],
            "wind_speed_m_s": diagnostics["wind_speed"]["rmse_m_s"],
        },
        "limitation": report["limitations"][0],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a local codec run directory")
    parser.add_argument("--run-dir", required=True, help="Path to codec run output directory")
    args = parser.parse_args(argv)
    try:
        report = load_local_evaluation(Path(args.run_dir))
        summary = build_summary(report)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"evaluate_codec_local: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
