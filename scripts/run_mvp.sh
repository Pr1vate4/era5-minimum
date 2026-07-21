#!/usr/bin/env bash
set -euo pipefail
pytest -q
python -m era5_minimum.experiments --config configs/mvp.yaml
