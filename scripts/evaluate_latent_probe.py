from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a latent probe metrics file")
    parser.add_argument("--metrics", required=True, help="Path to probe_metrics.json")
    args = parser.parse_args()

    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "train_pair_count": payload.get("train_pair_count"),
                "validation_pair_count": payload.get("validation_pair_count"),
                "optimizer_steps": payload.get("optimizer_steps"),
                "relative_improvement_vs_persistence_pct": payload.get(
                    "relative_improvement_vs_persistence_pct"
                ),
                "forecast_horizon_hours": payload.get("forecast_horizon_hours"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
