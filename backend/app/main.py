from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import aiofiles
from arq.connections import ArqRedis, create_pool
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.jobs import list_jobs, load_job, update_status
from app.schemas import (
    JobCreatedResponse,
    JobExtra,
    JobResultResponse,
    JobStatus,
    JobStatusResponse,
)
from app.settings import settings

settings.ensure_storage()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: startup and shutdown."""
    # Startup: Create Redis connection pool
    app.state.redis = await create_pool(settings.redis_settings)
    yield
    # Shutdown: Close Redis connection
    redis: ArqRedis | None = getattr(app.state, "redis", None)
    if redis:
        await redis.aclose()


app = FastAPI(title="Textbook Problem Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount(
    "/storage",
    StaticFiles(directory=settings.storage_root, check_dir=False),
    name="storage",
)


async def redis_dep() -> ArqRedis:
    redis: ArqRedis | None = getattr(app.state, "redis", None)
    if not redis:
        raise HTTPException(status_code=500, detail="Redis connection missing")
    return redis


@app.get("/api/jobs", response_model=list[JobStatusResponse])
async def list_job_statuses(
    status: JobStatus | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    redis: ArqRedis = Depends(redis_dep),
) -> list[JobStatusResponse]:
    statuses = [status] if status is not None else None
    jobs = await list_jobs(redis, statuses=statuses)
    if limit is not None:
        jobs = jobs[:limit]
    return [
        JobStatusResponse.model_validate(job.model_dump())
        for job in jobs
    ]


@app.post("/api/jobs", response_model=JobCreatedResponse)
async def submit_job(
    file: UploadFile = File(...),
    redis: ArqRedis = Depends(redis_dep),
) -> JobCreatedResponse:
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    job_id = uuid4()
    pdf_path = settings.job_pdf_path(str(job_id))

    async with aiofiles.open(pdf_path, "wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            await handle.write(chunk)
    await file.close()

    await update_status(
        redis, job_id, JobStatus.QUEUED, extra=JobExtra(filename=filename)
    )

    await redis.enqueue_job(
        "process_pdf_job",
        job_id=str(job_id),
        filename=filename,
    )
    return JobCreatedResponse(job_id=job_id, status=JobStatus.QUEUED)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(
    job_id: UUID,
    redis: ArqRedis = Depends(redis_dep),
) -> JobStatusResponse:
    record = await load_job(redis, job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse.model_validate(record.model_dump())


@app.get("/api/jobs/{job_id}/result", response_model=JobResultResponse)
async def job_result(
    job_id: UUID,
    redis: ArqRedis = Depends(redis_dep),
) -> JobResultResponse:
    record = await load_job(redis, job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    result_path_str = record.result_path
    if result_path_str:
        result_path = Path(result_path_str)
        if not result_path.is_absolute():
            result_path = settings.storage_root / result_path
    else:
        result_path = settings.job_result_path(str(job_id))
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not available.")
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Result JSON is invalid.") from exc
    return JobResultResponse(job_id=job_id, result=data)


@app.get("/api/jobs/{job_id}/download")
async def download_artifacts(
    job_id: UUID,
) -> FileResponse:
    pdf_path = settings.job_pdf_path(str(job_id))
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(
        pdf_path,
        filename=pdf_path.name,
        media_type="application/pdf",
    )


@app.websocket("/ws/jobs/{job_id}")
async def job_updates(
    websocket: WebSocket,
    job_id: UUID,
) -> None:
    await websocket.accept()
    redis: ArqRedis | None = getattr(app.state, "redis", None)
    if not redis:
        await websocket.close(code=1011)
        return

    initial = await load_job(redis, job_id)
    if initial:
        await websocket.send_text(initial.model_dump_json())

    pubsub = redis.pubsub()
    channel = settings.job_channel(str(job_id))
    await pubsub.subscribe(channel)

    async def sender() -> None:
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                payload = message["data"]
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                await websocket.send_text(payload)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    send_task = asyncio.create_task(sender())
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send_task
