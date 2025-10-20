from __future__ import annotations

import os
from pathlib import Path

from arq.connections import RedisSettings
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).parent.parent
REPO_ROOT = Path(__file__).parent.parent.parent
MODE = os.getenv("MODE", "local")


class Settings(BaseSettings):
    """Runtime configuration resolved from the environment."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env" if MODE == "local" else REPO_ROOT / ".env.prod",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0

    storage_root: Path = Field(default_factory=lambda: BACKEND_ROOT / "storage")
    job_ttl_seconds: int = 86400

    datalab_api_key: str | None = None
    openai_api_key: str | None = None
    llm_max_concurrency: int = 5

    websocket_channel_prefix: str = "job"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "uploads"

    @property
    def jobs_root(self) -> Path:
        return self.storage_root / "jobs"

    def ensure_storage(self) -> None:
        """Create core storage directories."""
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_pdf_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "source.pdf"

    def job_ocr_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "ocr.json"

    def job_working_dir(self, job_id: str) -> Path:
        working_dir = self.job_dir(job_id) / "artifacts"
        working_dir.mkdir(parents=True, exist_ok=True)
        return working_dir

    def job_result_path(self, job_id: str) -> Path:
        return self.job_working_dir(job_id) / "outputs" / "problem_assignments.json"

    def job_status_key(self, job_id: str) -> str:
        return f"{self.websocket_channel_prefix}:{job_id}:status"

    def job_channel(self, job_id: str) -> str:
        return f"{self.websocket_channel_prefix}:{job_id}:events"

    @property
    def redis_settings(self) -> RedisSettings:
        return RedisSettings(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_db,
        )


settings = Settings()
