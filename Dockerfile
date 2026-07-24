ARG PYTHON_VERSION=3.12.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ARG APP_UID=1000
ARG APP_GID=1000
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/era5/.local/bin:${PATH}"

WORKDIR /workspace

RUN groupadd --gid "${APP_GID}" era5 \
    && useradd --uid "${APP_UID}" --gid era5 --create-home --shell /usr/sbin/nologin era5


FROM base AS runtime-cpu

COPY pyproject.toml README.md ./
COPY src ./src

# No Python lock file exists yet; pyproject.toml is the canonical dependency source.
# Resolve the existing torch>=2.2 constraint from the official CPU wheel index
# before installing the API and test extras, so this CPU image does not pull a
# CUDA runtime stack.
RUN python -m pip install --no-cache-dir --index-url "${PYTORCH_CPU_INDEX_URL}" "torch>=2.2" \
    && python -m pip install --no-cache-dir ".[api,dev]"

COPY --chown=era5:era5 configs ./configs
COPY --chown=era5:era5 demo ./demo
COPY --chown=era5:era5 scripts ./scripts
COPY --chown=era5:era5 tests ./tests
COPY --chown=era5:era5 docs/ARTIFACT_API_CONTRACT.md ./docs/ARTIFACT_API_CONTRACT.md

RUN mkdir -p data outputs checkpoints bitstreams artifacts submission \
    && chown -R era5:era5 /workspace

USER era5:era5

CMD ["python", "-m", "uvicorn", "era5_minimum.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
