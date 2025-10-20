from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Dict, List, Any
from uuid import UUID

import anyio
import pytest
import redis
from fastapi.testclient import TestClient
from arq.connections import create_pool

from app.settings import settings
from arq_worker import worker
from research.detector.pipeline import PipelineResult


@dataclass
class TestEnvironment:
    client: TestClient
    sample_pdf: Path
    __test__ = False


@pytest.fixture
def integration_env(tmp_path, monkeypatch) -> TestEnvironment:
    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
    )
    try:
        redis_client.ping()
    except Exception:  # pragma: no cover - depends on external redis
        pytest.skip("Redis server is required for integration tests.")
    redis_client.flushdb()

    storage_root = tmp_path / "storage"
    monkeypatch.setattr(settings, "storage_root", storage_root)
    settings.ensure_storage()

    work_dir = tmp_path / "work"
    monkeypatch.setattr("research.detector.config.WORK_DIR", work_dir)

    sample_ocr_path = (
        Path(__file__).resolve().parents[1] / "research" / "ocr_result_pass.json"
    )
    sample_pdf = Path(__file__).resolve().parents[1] / "research" / "data" / "math.pdf"

    class FakeDatalabClient:
        async def ocr(self, *_args, **_kwargs):
            return json.loads(sample_ocr_path.read_text(encoding="utf-8"))

        async def aclose(self) -> None:
            pass

    async def fake_get_client(ctx: Dict[str, Any]) -> FakeDatalabClient:
        client = ctx.get("datalab_client")
        if client is None:
            client = FakeDatalabClient()
            ctx["datalab_client"] = client
        return client

    async def fake_run_pipeline(
        *,
        pdf_path: Path,
        ocr_result_path: Path,
        working_dir: Path | None = None,
        visualise: bool = True,
        max_parallel_requests: int | None = None,
    ) -> PipelineResult:
        workdir = Path(working_dir or tmp_path / "artifacts")
        outputs_dir = workdir / "outputs"
        raw_dir = outputs_dir / "llm_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        json_path = outputs_dir / "problem_assignments.json"
        payload = [
            {
                "page": 1,
                "problems": [],
                "lines": [],
                "unassigned_lines": [],
            }
        ]
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return PipelineResult(
            page_assignments=[],
            image_paths=[],
            output_json=json_path,
            visualisations=[],
            problem_visualisations=[],
            raw_outputs_dir=raw_dir,
        )

    monkeypatch.setattr(worker, "_get_datalab_client", fake_get_client)
    monkeypatch.setattr(worker, "run_pipeline", fake_run_pipeline)

    app_main = importlib.import_module("app.main")

    with TestClient(app_main.app) as client:
        yield TestEnvironment(
            client=client,
            sample_pdf=sample_pdf,
        )

    redis_client.flushdb()
    redis_client.close()


def test_backend_full_flow(integration_env: TestEnvironment) -> None:
    env = integration_env
    client = env.client

    with env.sample_pdf.open("rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"file": ("math.pdf", handle, "application/pdf")},
        )
    assert response.status_code == 200
    body = response.json()
    job_id = UUID(body["job_id"])
    assert body["status"] == "queued"

    statuses: List[str] = []

    async def run_worker_async() -> None:
        redis_pool = await create_pool(settings.redis_settings)
        try:
            await worker.process_pdf_job(
                {"redis": redis_pool},
                job_id=str(job_id),
                filename="math.pdf",
            )
        finally:
            await redis_pool.aclose()

    def run_worker() -> None:
        anyio.run(run_worker_async)

    with client.websocket_connect(f"/ws/jobs/{job_id}") as websocket:
        initial = websocket.receive_json()
        assert UUID(initial["job_id"]) == job_id
        statuses.append(initial["status"])

        worker_thread = Thread(target=run_worker, daemon=True)
        worker_thread.start()

        while True:
            message = websocket.receive_json()
            statuses.append(message["status"])
            if message["status"] == "completed":
                break

        worker_thread.join(timeout=30)

    assert "processing" in statuses
    assert statuses[-1] == "completed"

    status_response = client.get(f"/api/jobs/{job_id}")
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "completed"
    assert status_payload["result_path"]

    result_response = client.get(f"/api/jobs/{job_id}/result")
    assert result_response.status_code == 200
    result_payload = result_response.json()
    assert result_payload["result"]

    download_response = client.get(f"/api/jobs/{job_id}/download")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] == "application/pdf"
