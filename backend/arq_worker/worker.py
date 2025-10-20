from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from arq.connections import ArqRedis, RedisSettings
from arq.typing import WorkerSettingsBase
from arq.worker import run_worker
from datalab_sdk import AsyncDatalabClient
from datalab_sdk.models import ProcessingOptions
from detector.pipeline import run_pipeline
from loguru import logger

from app.jobs import update_status
from app.schemas import JobExtra, JobStatus
from app.settings import settings

settings.ensure_storage()


async def startup(ctx: dict[str, Any]) -> None:  # noqa: ARG001
    logger.info("Starting up arq worker")


def _normalise_ocr_payload(payload: Any) -> Any:
    if payload is None:
        raise ValueError("OCR response is empty.")
    if isinstance(payload, (dict, list)):
        return payload
    if is_dataclass(payload):
        data = asdict(payload)
        if isinstance(data, (dict, list)):
            return data
    for attr in ("model_dump", "dict"):
        fn = getattr(payload, attr, None)
        if callable(fn):
            data = fn()
            if isinstance(data, (dict, list)):
                return data
    if hasattr(payload, "model_dump_json"):
        text = payload.model_dump_json()
        return json.loads(text)
    raise TypeError(f"Unsupported OCR payload type: {type(payload)!r}")


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(settings.storage_root))
    except ValueError:
        return str(path)


async def process_pdf_job(
    ctx: dict[str, Any], job_id: str, filename: str | None = None
) -> None:
    redis: ArqRedis = ctx["redis"]

    await update_status(
        redis, job_id, JobStatus.PROCESSING, extra=JobExtra(filename=filename)
    )

    pdf_path = settings.job_pdf_path(job_id)
    if not pdf_path.exists():
        await update_status(
            redis, job_id, JobStatus.FAILED, message="Uploaded PDF is missing."
        )
        logger.error("PDF not found for job %s at %s", job_id, pdf_path)
        return

    try:
        await update_status(redis, job_id, JobStatus.OCR_RUNNING)
        client = AsyncDatalabClient(api_key=settings.datalab_api_key)
        ocr_payload = await client.ocr(
            str(pdf_path),
            options=ProcessingOptions(),
        )
        await client.close()
        if ocr_payload.success:
            ocr_path = settings.job_ocr_path(job_id)
            ocr_path.write_text(
                json.dumps(asdict(ocr_payload), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await update_status(redis, job_id, JobStatus.OCR_COMPLETE, extra=JobExtra(ocr_path=_relative_path(ocr_path)))
        else:
            raise ValueError(ocr_payload.error)
        await update_status(redis, job_id, JobStatus.PIPELINE_RUNNING)
        working_dir = settings.job_working_dir(job_id)
        result = await run_pipeline(
            pdf_path=pdf_path,
            ocr_result_path=ocr_path,
            working_dir=working_dir,
            visualise=True,
        )
        await update_status(
            redis,
            job_id,
            JobStatus.COMPLETED,
            extra=JobExtra(
                result_path=_relative_path(result.output_json),
                artifacts_dir=_relative_path(working_dir),
                image_paths=[_relative_path(path) for path in result.image_paths],
                visualisations=[_relative_path(path) for path in result.visualisations],
                problem_visualisations=[
                    _relative_path(path) for path in result.problem_visualisations
                ],
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Job %s failed", job_id)
        await update_status(redis, job_id, JobStatus.FAILED, message=str(exc))
        raise


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Shutting down arq worker")

class WorkerSettings(WorkerSettingsBase):
    redis_settings: RedisSettings = settings.redis_settings
    functions = [process_pdf_job]
    on_shutdown = shutdown  # type: ignore[assignment]
    on_startup = startup  # type: ignore[assignment]


if __name__ == "__main__":
    run_worker(WorkerSettings, job_timeout=timedelta(minutes=20))
