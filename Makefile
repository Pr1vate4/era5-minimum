.PHONY: install test mvp verify clean

install:
	python -m pip install -e .

test:
	pytest -q

mvp:
	python -m era5_minimum.experiments --config configs/mvp.yaml

verify: test mvp

clean:
	rm -rf outputs/mvp .pytest_cache
