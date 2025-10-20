FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY . /app

RUN uv sync --frozen --no-dev
