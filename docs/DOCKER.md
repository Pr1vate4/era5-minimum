# Docker development infrastructure

## Requirements

- Docker Engine and Docker Compose v2;
- for `gpu-*` commands: a compatible NVIDIA driver and NVIDIA Container Toolkit.

The CPU image uses Python 3.12, matching CI. It resolves the existing
`torch>=2.2` constraint from the official CPU-wheel index, so it does not pull
a CUDA runtime stack. There is no Python lock file in this repository, so
other dependencies are installed from the canonical
`pyproject.toml`. The root `Dockerfile` has the `base` and `runtime-cpu`
targets; a CUDA-specific image target does not exist yet because this project
does not define a CUDA-PyTorch image configuration.

## First use

```bash
cp .env.example .env
make docker-config
make docker-build
```

`make runtime-dirs` is run by the operational targets and creates the bind
mount source directories as the invoking host user. Set `APP_UID` and
`APP_GID` in `.env` to that user's numeric IDs before rebuilding if they are
not `1000`; containers then write artifacts without creating root-owned files.

## API

```bash
make app-up
make app-health
make app-logs
```

- API: <http://localhost:8000>
- Health: <http://localhost:8000/health>
- OpenAPI: <http://localhost:8000/docs>
- Prometheus endpoint: <http://localhost:8000/metrics>

Stop it with `make app-down`. The `api` service uses one Uvicorn worker, no
GPU, and the existing `era5_minimum.api.app:app` ASGI object. It is reachable
to a future monitoring override as `http://api:8000`; no monitoring services
are included now.

By default the API serves the image's `demo/mock` artifacts. Its host
`ARTIFACTS_DIR` mount is read-only. To serve a verified mounted artifact bundle
instead, set `ERA5_ARTIFACTS_ROOT=/workspace/artifacts` in local `.env`.

## Bind mounts

| Host variable | Container path | API mode | Tools mode |
| --- | --- | --- | --- |
| `DATA_DIR` | `/workspace/data` | not mounted | read-only; writable only in `data-tools` |
| `OUTPUTS_DIR` | `/workspace/outputs` | not mounted | read-write |
| `CHECKPOINTS_DIR` | `/workspace/checkpoints` | not mounted | read-write |
| `BITSTREAMS_DIR` | `/workspace/bitstreams` | not mounted | read-write |
| `ARTIFACTS_DIR` | `/workspace/artifacts` | read-only | read-write |
| `SUBMISSION_DIR` | `/workspace/submission` | not mounted | read-write |

Zarr, NetCDF/GRIB, NPZ, checkpoints, bitstreams, outputs and submission
artifacts are excluded from the Docker build context. They remain accessible
on the host through bind mounts rather than being copied into an image.

## CPU and data commands

```bash
make tools-shell
make tools-run CMD="python --version"
make data-shell
make data-run CMD="python --version"
make pca-help
make pca-fit-32x
make pca-fit-64x
make artifacts-validate
```

`pca-fit-32x` and `pca-fit-64x` run the existing
`scripts/fit_pca_baseline.py` with the corresponding real project config. The
configs expect prepared NPZ files beneath `/workspace/outputs/real_era5`; they
do not download or generate research data. No evaluator CLI currently exists,
so no evaluator target is provided.

`data-tools` is the only service permitted to write `/workspace/data`; use it
for future Zarr preparation. It does not implement a data loader itself.

## GPU commands and limits

```bash
make docker-gpu-config
make gpu-check
make gpu-shell
make gpu-run CMD="python --version"
```

`compose.gpu.yaml` requests exactly one GPU with the Compose v2
`deploy.resources.reservations.devices` declaration (`count: 1`) and exposes
device `0` as `CUDA_VISIBLE_DEVICES=0`. It is a profile-only one-off service
and does not start alongside `api`. It deliberately uses the current CPU target because
the repository has no declared CUDA image target; `make gpu-check` fails
clearly when the installed PyTorch cannot access CUDA.

Compose does **not** enforce the contest 24 GB peak-VRAM limit. A future
resource monitor must enforce/report that limit in `resource_usage.json`, along
with runtime, CPU memory, optimizer steps, unique timestamps and encode/decode
durations. Prometheus is not the source of record for those results.

The future codec must also remain within the stated contest envelope: one GPU,
at most 20 million trainable parameters, 50,000 optimizer steps and 48 GPU
hours. Its official input is 28 weather channels on either the 0.5° 360 × 720
or 0.25° 721 × 1440 grid. These are operational constraints for future CLI and
resource logging, not claims about the current 8-channel PCA baseline.

## Validation and final delivery

```bash
make test-container
make compile-container
make verify-container
make submission-tree
```

`submission/README.md` documents the future layout for code, container,
checkpoints, bitstreams, manifests, metrics, resources and plots. No fake
submission artifacts or validator are included. Final encoder/decoder commands
must consume CLI arguments, environment variables and `/workspace`-relative
paths, never a developer-specific absolute path.

Use `make docker-config` and `make docker-gpu-config` before running Compose.
Build the final CPU image with `make docker-build`. A future
`compose.monitoring.yaml` can attach to the existing `api` service and its
stable port without replacing it.
