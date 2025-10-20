from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    OCR_RUNNING = "ocr_running"
    OCR_COMPLETE = "ocr_complete"
    PIPELINE_RUNNING = "pipeline_running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JobExtra(BaseModel):
    filename: str | None = None
    ocr_path: str | None = None
    result_path: str | None = None
    artifacts_dir: str | None = None
    image_paths: list[str] = Field(default_factory=list)
    visualisations: list[str] = Field(default_factory=list)
    problem_visualisations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class JobStatusPayload(BaseModel):
    job_id: UUID
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    message: str | None = None
    filename: str | None = None
    result_path: str | None = None
    extra: JobExtra = Field(default_factory=JobExtra)

    model_config = ConfigDict(extra="allow")


class JobCreatedResponse(BaseModel):
    job_id: UUID
    status: JobStatus = Field(default=JobStatus.QUEUED)


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    filename: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    result_path: str | None = None
    message: str | None = None
    extra: JobExtra = Field(default_factory=JobExtra)


class JobResultResponse(BaseModel):
    job_id: UUID
    result: Any
