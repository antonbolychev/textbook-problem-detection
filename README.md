# Textbook Problem Detection Platform

This repository now contains three main areas:

- `research/` – previous prototype scripts and assets for the pipeline.
- `backend/` – FastAPI service for managing uploads, running the pipeline asynchronously via an ARQ worker, and serving artefacts through Redis-backed job tracking.
- `frontend/` – Vite + React + Chakra UI interface for uploading textbooks, tracking progress, and reviewing detected problems with adjustable bounding boxes.

## Quick start

```bash
# build and start all services
docker compose up --build
```

The stack includes:

- **Redis** for job coordination and websocket events.
- **Backend API** (`http://localhost:8000`) exposing REST + websocket endpoints and serving stored artefacts under `/storage`.
- **ARQ worker** responsible for OCR (via `AsyncDatalabClient`), running the pipeline from `research/`, and persisting results.
- **Frontend** (`http://localhost:5173`) providing the user flow.

Set the following environment variables before running if needed:

- `OPENAI_API_KEY`, `OPENAI_API_BASE`
- `DATALAB_API_KEY`, `DATALAB_BASE_URL`

`docker-compose.yaml` forwards these into the backend and worker containers.

## Local development

- Backend: `cd backend && uv sync && uvicorn backend.app.main:app --reload`
- Worker: `cd backend && uv run arq backend.arq_worker.worker.WorkerSettings`
- Frontend: `cd frontend && npm install && npm run dev`

Ensure `storage/` remains shared so artefacts are accessible to both backend and frontend.

## Research assets

All previous experimentation (scripts, notebooks, artefacts) now lives under `research/`. `research/main.py` continues to run the original pipeline unchanged.
