# Backend service

FastAPI application exposing upload/status/result endpoints and an ARQ worker that coordinates OCR with the textbook pipeline from `research/`.

## Useful commands

```bash
uv sync
uv run uvicorn backend.app.main:app --reload
uv run arq backend.arq_worker.worker.WorkerSettings
```

Use the shared `storage/` directory to exchange files with the worker and frontend.
