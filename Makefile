.PHONY: install install-dev test mvp smoke verify download-dry-run clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

mvp:
	python -m era5_minimum.experiments --config configs/mvp.yaml

smoke: mvp

verify: test smoke

download-dry-run:
	python scripts/download_era5.py --date 2024-01-01 --times 00:00 06:00 12:00 18:00 --output-root /tmp/era5-minimum-dry-run --dry-run

clean:
	rm -rf outputs/mvp .pytest_cache .mypy_cache .ruff_cache build dist
