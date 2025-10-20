from __future__ import annotations

import json
from loguru import logger
from datetime import UTC, datetime
from typing import Any, Iterable
from uuid import UUID

from arq.connections import ArqRedis

from app.schemas import JobExtra, JobStatus, JobStatusPayload
from app.settings import settings


def _now() -> datetime:
    return datetime.now(UTC)


async def load_job(redis: ArqRedis, job_id: UUID | str) -> JobStatusPayload | None:
    job_id_str = _job_id_str(job_id)
    data = await redis.get(settings.job_status_key(job_id_str))
    if not data:
        return None
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    try:
        return JobStatusPayload.model_validate_json(data)
    except (json.JSONDecodeError, ValueError):
        return None


async def list_jobs(
    redis: ArqRedis,
    *,
    statuses: Iterable[JobStatus] | None = None,
) -> list[JobStatusPayload]:
    pattern = f"{settings.websocket_channel_prefix}:*:status"
    keys = await redis.keys(pattern)

    allowed_statuses: set[JobStatus] | None = None
    if statuses is not None:
        allowed_statuses = set(statuses)

    jobs: list[JobStatusPayload] = []
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        payload = await _load_job_from_key(redis, key)
        if payload is None:
            continue
        if allowed_statuses is not None and payload.status not in allowed_statuses:
            continue
        jobs.append(payload)

    jobs.sort(key=lambda record: record.updated_at, reverse=True)
    return jobs


async def _load_job_from_key(
    redis: ArqRedis,
    key: str,
) -> JobStatusPayload | None:
    data = await redis.get(key)
    if not data:
        return None
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    try:
        return JobStatusPayload.model_validate_json(data)
    except (json.JSONDecodeError, ValueError):
        return None


async def persist_job(redis: ArqRedis, job_id: str, payload: JobStatusPayload) -> None:
    await redis.set(
        settings.job_status_key(job_id),
        payload.model_dump_json(),
        ex=settings.job_ttl_seconds,
    )


async def publish_status(
    redis: ArqRedis, job_id: str, payload: JobStatusPayload
) -> None:
    await redis.publish(
        settings.job_channel(job_id),
        payload.model_dump_json(),
    )


async def update_status(
    redis: ArqRedis,
    job_id: UUID | str,
    status: JobStatus,
    *,
    message: str | None = None,
    extra: JobExtra | None = None,
) -> JobStatusPayload:
    logger.info(f"Updating status for job {job_id} to {status}")
    job_uuid = _job_id(job_id)
    job_id_str = str(job_uuid)
    existing = await load_job(redis, job_uuid)
    now = _now()
    if existing is None:
        record = JobStatusPayload(
            job_id=job_uuid,
            status=status,
            created_at=now,
            updated_at=now,
            extra=JobExtra(),
        )
    else:
        record = existing.model_copy()

    merged_extra = record.extra
    if extra is not None:
        merged_extra = _merge_extra(record.extra, extra)

    update_payload: dict[str, Any] = {
        "status": status,
        "updated_at": now,
        "extra": merged_extra,
    }
    if message is not None:
        update_payload["message"] = message
    if extra is not None and extra.filename is not None:
        update_payload["filename"] = extra.filename
    if extra is not None and extra.result_path is not None:
        update_payload["result_path"] = extra.result_path

    record = record.model_copy(update=update_payload)
    await persist_job(redis, job_id_str, record)
    await publish_status(redis, job_id_str, record)
    return record


def _merge_extra(current: JobExtra, new_extra: JobExtra) -> JobExtra:
    return current.model_copy(update=new_extra.model_dump())


def _job_id(value: UUID | str) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _job_id_str(value: UUID | str) -> str:
    return str(_job_id(value))
