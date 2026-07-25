from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local codec run directory")
    parser.add_argument("--run-dir", required=True, help="Path to codec run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "run_id": summary.get("run_id"),
                "actual_compression_ratio": summary.get("actual_compression_ratio"),
                "overall_score": validation.get("overall_score"),
                "codec_eligible": summary.get("codec_eligible"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
