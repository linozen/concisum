# Concisum — LLM summarization & ICD-10 diagnosis API
#
# No GPU required — calls Ollama for LLM inference.
# The sources/ directory (ICD-10 reference data) is baked into the image.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8090

CMD ["uv", "run", "--no-sync", "uvicorn", "src.concisum.api.app:app", "--host", "0.0.0.0", "--port", "8090"]
