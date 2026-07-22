.PHONY: install install-dev test mvp smoke verify clean

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

clean:
	rm -rf outputs/mvp .pytest_cache .mypy_cache .ruff_cache build dist
