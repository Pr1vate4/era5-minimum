DOCKER_COMPOSE ?= docker compose
COMPOSE_FILE ?= compose.yaml
GPU_COMPOSE_FILE ?= compose.gpu.yaml
MONITORING_COMPOSE_FILE ?= compose.monitoring.yaml
API_SERVICE ?= api
TOOLS_SERVICE ?= tools
DATA_SERVICE ?= data-tools
GPU_SERVICE ?= gpu-tools
API_PORT ?= 8000
MODEL_ARCHIVE ?= era5-minimum-n32-model.zip
MODEL_ARTIFACT_DIR ?= artifacts/model-n32

export API_PORT

.PHONY: help install install-dev test mvp smoke verify download-dry-run download-range-dry-run clean \
	runtime-dirs docker-build docker-config docker-gpu-config app-up app-down app-restart \
	app-logs app-ps app-health app-shell tools-shell tools-run data-shell data-run \
	gpu-shell gpu-run gpu-check test-container compile-container verify-container \
	pca-help pca-fit-32x pca-fit-64x artifacts-validate submission-tree \
	submission-validate clean-containers monitoring-config monitoring-up monitoring-down \
	monitoring-restart monitoring-logs monitoring-ps monitoring-prometheus-logs \
	monitoring-grafana-logs monitoring-smoke monitoring-check-prometheus-config \
	monitoring-check-rules monitoring-clean frontend-up frontend-build dev \
	ml-samples cra5-checkpoint-dry-run cra5-smoke cra5-train-n16 cra5-train-n32 \
	cra5-train-n64 cra5-train-n128 experiment-ladder demo demo-ultra demo-full \
	cra5-profile-params cra5-produce-checkpoint cra5-verify-checkpoint \
	weatherbench2-pilot-n128 pca-weatherbench2-n128 weatherbench2-merge-n128 conv-ae-weatherbench2-n128 conv-ae-bitstream-n128 \
	model-unpack

help: ## Show available project and container commands.

	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "%-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install the core package in editable mode.
	python -m pip install -e .

install-dev: ## Install the package with development dependencies.
	python -m pip install -e ".[dev]"

test: ## Run tests and validate the bundled demo artifact contract.
	python -m pytest -q
	python scripts/validate_artifact_bundle.py demo/mock

mvp: ## Run the existing synthetic MVP.
	python -m era5_minimum.experiments --config configs/mvp.yaml

smoke: mvp ## Alias for the synthetic MVP.

verify: test smoke ## Run tests and the synthetic MVP.

download-dry-run: ## Validate one ERA5 download request offline.
	python scripts/download_era5.py --date 2024-01-01 --times 00:00 06:00 12:00 18:00 --output-root /tmp/era5-minimum-dry-run --dry-run

download-range-dry-run: ## Validate a sequential ERA5 range request offline.
	python scripts/download_era5_range.py --start-date 2024-01-01 --end-date 2024-01-03 --output-root /tmp/era5-minimum-range-dry-run --dry-run

ml-samples: ## Generate nested leakage-safe training sample manifests (16/32/64/128).
	python scripts/prepare_ml_sample_manifest.py --output /tmp/era5-minimum-samples.json

cra5-checkpoint-dry-run: ## Show CRA5-159 checkpoint provenance without downloading.
	python scripts/fetch_cra5_checkpoint.py

cra5-smoke: ## Run CRA5 runtime smoke test with synthetic data.
	python scripts/smoke_cra5_runtime.py

demo: ## ONE-COMMAND DEMO (quick mode, ~1 min CPU, ≥64x serialized CR).
	PYTHONPATH=src python scripts/run_demo.py

demo-ultra: ## One-command demo ultra-fast (~15s CPU, ≥32x CR expected).
	PYTHONPATH=src python scripts/run_demo.py --ultra

demo-full: ## One-command demo — largest ≤20M config (h384/l96, ≥32x CR expected).
	PYTHONPATH=src python scripts/run_demo.py --full

cra5-profile-params: ## Profile CRA5 configs under the 20M parameter limit.
	PYTHONPATH=src python scripts/_profile_params.py

cra5-produce-checkpoint: runtime-dirs ## Produce Artemka-valid checkpoint bundle (h256/l64, patch-train).
	PYTHONPATH=src python scripts/produce_valid_checkpoint.py --max-train-batches 4

cra5-verify-checkpoint: runtime-dirs ## Verify checkpoint against the Artemka formal checklist.
	PYTHONPATH=src python scripts/verify_checkpoint.py --name cra5_era5_28ch_best

cra5-train-n16: ## Train CRA5 adapter with 16 samples.
	python scripts/train_cra5_adapter.py --config configs/cra5/cra5_adapter_n16.yaml

cra5-train-n32: ## Train CRA5 adapter with 32 samples.
	python scripts/train_cra5_adapter.py --config configs/cra5/cra5_adapter_n32.yaml

cra5-train-n64: ## Train CRA5 adapter with 64 samples.
	python scripts/train_cra5_adapter.py --config configs/cra5/cra5_adapter_n64.yaml

cra5-train-n128: ## Train CRA5 adapter with 128 samples.
	python scripts/train_cra5_adapter.py --config configs/cra5/cra5_adapter_n128.yaml

experiment-ladder: ## Run full experiment ladder (PCA → ConvAE → CRA5).
	python scripts/run_experiment_ladder.py

clean: ## Remove only generated MVP outputs and local caches.
	rm -rf outputs/mvp .pytest_cache .mypy_cache .ruff_cache build dist

runtime-dirs: ## Create host bind-mount directories with the invoking user's ownership.
	mkdir -p data outputs checkpoints bitstreams artifacts submission

docker-build: runtime-dirs ## Build the CPU API/tools image.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build

docker-config: ## Render and validate the CPU Compose configuration.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) config

docker-gpu-config: ## Render and validate the CPU plus GPU Compose configuration.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) config

app-up: runtime-dirs ## Start the API and Vite frontend together in Docker.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d --build $(API_SERVICE) frontend

app-down: ## Stop the Compose stack without deleting bind-mounted artifacts.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

app-restart: ## Restart the API service.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) restart $(API_SERVICE)

app-logs: ## Follow the latest 200 API log lines.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f --tail=200 $(API_SERVICE)

app-ps: ## Show Compose service status.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

app-health: ## Check the API health endpoint from the host.
	curl -fsS http://localhost:$(API_PORT)/health

frontend-up: ## Start the Vite frontend dev server on VITE_PORT (default: 5173).
	npm run dev

dev: ## Start the local API if needed, then run the Vite frontend.
	@if curl -fsS http://$(LOCAL_API_HOST):$(API_PORT)/health >/dev/null 2>&1; then \
		echo "API is already running at http://$(LOCAL_API_HOST):$(API_PORT)"; \
	else \
		test -x .venv/bin/python || { echo "Missing .venv/bin/python; create the project virtual environment first." >&2; exit 1; }; \
		.venv/bin/python -m uvicorn era5_minimum.api.app:app --host $(LOCAL_API_HOST) --port $(API_PORT) --workers 1 --log-level info > /tmp/era5-minimum-api.log 2>&1 & \
		echo "API started at http://$(LOCAL_API_HOST):$(API_PORT) (log: /tmp/era5-minimum-api.log)"; \
	fi
	npm run dev -- --host $(VITE_HOST)

frontend-build: ## Build the frontend for static hosting.
	npm run build

app-shell: ## Open a shell in the running API container.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) exec $(API_SERVICE) sh

tools-shell: runtime-dirs ## Open a one-off shell for CPU tools.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) sh

tools-run: runtime-dirs ## Run CMD in the one-off CPU tools container.
	@test -n "$(CMD)" || { echo "CMD is required, for example: make tools-run CMD=\"python --version\"" >&2; exit 2; }
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) $(CMD)

data-shell: runtime-dirs ## Open a one-off shell with writable data mount.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile data run --rm $(DATA_SERVICE) sh

data-run: runtime-dirs ## Run CMD in the data-preparation container.
	@test -n "$(CMD)" || { echo "CMD is required, for example: make data-run CMD=\"python --version\"" >&2; exit 2; }
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile data run --rm $(DATA_SERVICE) $(CMD)

gpu-shell: runtime-dirs ## Open a one-off shell with exactly one requested GPU.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) --profile gpu run --rm $(GPU_SERVICE) sh

gpu-run: runtime-dirs ## Run CMD in the one-off GPU tools container.
	@test -n "$(CMD)" || { echo "CMD is required, for example: make gpu-run CMD=\"python --version\"" >&2; exit 2; }
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) --profile gpu run --rm $(GPU_SERVICE) $(CMD)

gpu-check: runtime-dirs ## Check CUDA visibility, one GPU, device name, version, and memory.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(GPU_COMPOSE_FILE) --profile gpu run --rm $(GPU_SERVICE) python -c "import sys, torch; ok = torch.cuda.is_available(); count = torch.cuda.device_count() if ok else 0; print('cuda_available=', ok); print('device_count=', count); print('device_name=', torch.cuda.get_device_name(0) if ok else 'NO_GPU'); print('torch_cuda=', torch.version.cuda); print('total_memory_bytes=', torch.cuda.get_device_properties(0).total_memory if ok else 'NO_GPU'); sys.exit(0 if ok and count == 1 else 'CUDA is unavailable or more than one GPU is visible')"

test-container: runtime-dirs ## Run the full pytest suite inside the CPU tools image.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) pytest -q

compile-container: runtime-dirs ## Compile project Python sources inside the CPU tools image.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) python -m compileall src scripts

verify-container: docker-config compile-container test-container ## Validate Compose, compile, and test in the CPU image.

pca-help: runtime-dirs ## Show the existing Patch PCA CLI help in the CPU tools image.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) python scripts/fit_pca_baseline.py --help

pca-fit-32x: runtime-dirs ## Run the existing 32x Patch PCA config against mounted outputs.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) python scripts/fit_pca_baseline.py --config configs/patch_pca_32x.yaml

pca-fit-64x: runtime-dirs ## Run the existing 64x Patch PCA config against mounted outputs.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) python scripts/fit_pca_baseline.py --config configs/patch_pca_64x.yaml

weatherbench2-pilot-n128: ## Prepare 128 real WeatherBench2 train frames plus 16/16 fixed holdouts at 0.5°.
	python scripts/prepare_weatherbench2_05.py --output-dir data/weatherbench2_28ch_05_n128 --train-size 128 --validation-size 16 --test-size 16 --seed 42

pca-weatherbench2-n128: ## Fit the real 0.5° Patch-PCA reference with train-only statistics.
	python scripts/fit_patch_pca_zarr.py --config configs/patch_pca_weatherbench2_05_n128.yaml

weatherbench2-merge-n128: ## Merge a local pilot plus two verified teammate train shards.
	python scripts/merge_weatherbench2_shards.py --base data/weatherbench2_28ch_05_pilot16_v2 --train-shard data/weatherbench2_28ch_05_n128_shard_a --train-shard data/weatherbench2_28ch_05_n128_shard_b --output-dir data/weatherbench2_28ch_05_n128 --seed 42 --train-size 128

conv-ae-weatherbench2-n128: ## Train the 32× real-data ConvAE after N=128 shards are merged.
	python scripts/train_conv_autoencoder_zarr.py --config configs/conv_ae_weatherbench2_05_n128.yaml

conv-ae-bitstream-n128: ## Measure an actual ConvAE bitstream after real N=128 training.
	python scripts/evaluate_conv_ae_bitstream_zarr.py --checkpoint outputs/conv_ae_weatherbench2_05_n128/model.ckpt --dataset-dir data/weatherbench2_28ch_05_n128 --output-dir outputs/conv_ae_weatherbench2_05_n128_codec

model-unpack: ## Unpack a shared N32 model ZIP from the repository root into ignored artifacts/model-n32/.
	@test -f "$(MODEL_ARCHIVE)" || { echo "Model archive not found: $(MODEL_ARCHIVE)" >&2; exit 2; }
	@test ! -e "$(MODEL_ARTIFACT_DIR)/model.ckpt" || { echo "Refusing to overwrite installed model: $(MODEL_ARTIFACT_DIR)/model.ckpt" >&2; exit 2; }
	@mkdir -p "$(MODEL_ARTIFACT_DIR)"
	@unzip -q "$(MODEL_ARCHIVE)" -d "$(MODEL_ARTIFACT_DIR)"
	@test -f "$(MODEL_ARTIFACT_DIR)/model.ckpt" || { echo "Archive must contain model.ckpt at its root." >&2; exit 2; }
	@test -f "$(MODEL_ARTIFACT_DIR)/normalization_train_only.json" || { echo "Archive must contain normalization_train_only.json at its root." >&2; exit 2; }
	@echo "Installed shared model in $(MODEL_ARTIFACT_DIR)"

artifacts-validate: runtime-dirs ## Validate the bundled demo artifact directory in the CPU image.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile tools run --rm $(TOOLS_SERVICE) python scripts/validate_artifact_bundle.py demo/mock

submission-tree: runtime-dirs ## Show the host submission directory tree.
	find submission -maxdepth 3 -print

submission-validate: ## Report that a final submission validator does not exist yet.
	@echo "submission validator is not implemented yet" >&2
	@exit 2

clean-containers: ## Remove stopped project containers without touching bind-mounted data.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) rm -f

monitoring-config: ## Render and validate the API plus monitoring Compose configuration.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) config

monitoring-up: runtime-dirs ## Start API, Prometheus, and Grafana in the background.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) up -d --build

monitoring-down: ## Stop the monitoring stack without deleting named volumes.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) down

monitoring-restart: ## Restart the API, Prometheus, and Grafana services.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) restart api prometheus grafana

monitoring-logs: ## Follow the latest 200 log lines for the monitoring stack.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) logs -f --tail=200 api prometheus grafana

monitoring-ps: ## Show API and monitoring service status.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) ps

monitoring-prometheus-logs: ## Follow the latest 200 Prometheus log lines.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) logs -f --tail=200 prometheus

monitoring-grafana-logs: ## Follow the latest 200 Grafana log lines.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) logs -f --tail=200 grafana

monitoring-smoke: ## Verify API metrics, Prometheus scraping, and Grafana health.
	python scripts/smoke_monitoring.py

monitoring-check-prometheus-config: ## Validate the Prometheus configuration with promtool.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) run --rm --no-deps --entrypoint promtool prometheus check config /etc/prometheus/prometheus.yml

monitoring-check-rules: ## Validate the ERA5 API Prometheus alert rules with promtool.
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) -f $(MONITORING_COMPOSE_FILE) run --rm --no-deps --entrypoint promtool prometheus check rules /etc/prometheus/rules/era5_api_rules.yml

monitoring-clean: monitoring-down ## Stop monitoring services; persisted monitoring volumes remain intact.
