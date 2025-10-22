"""
OpenAI helper for extracting problem assignments from textbook pages.

The core abstraction is :class:`LLMProblemExtractor`, which sends the rendered
page image together with OCR lines to the OpenAI multimodal endpoint and asks
for structured JSON describing which lines belong to which math problem.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from loguru import logger
import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from detector import config
from detector.models import TextLine

DEFAULT_SYSTEM_PROMPT = """
You specialise in understanding textbook pages. Your goal is to list every
exercise/problem on the page and tell me which OCR line indices belong to each
problem. Treat contiguous text that belongs to one problem (including headings,
subparts, and answer blanks) as part of that problem.

Return strict JSON with this shape:
{
  "page": <page_number>,
  "problems": [
    {
      "problem_id": "<identifier taken from the page or a synthetic id>",
      "line_indices": [0, 1, 2],
      "problem_text": "<full text of the problem as it appears on the page>"
    }
  ],
  "unassigned_lines": [3, 7]
}

- Use the provided OCR indices; do not invent new numbers.
- If the page has no problems, return an empty list and populate
  ``unassigned_lines`` with every index.
- Prefer identifiers that appear on the page (e.g. ``1.``, ``№ 24``). If none
  exists, synthesise ``auto-1``, ``auto-2``…
- The ``problem_text`` field must concatenate every sentence/line that belongs
  to the problem so downstream checks can rely on it.
"""


class Problem(BaseModel):
    problem_id: str = "auto-unknown"
    line_indices: list[int] = Field(default_factory=list)
    problem_text: str | None = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("problem_id", mode="before")
    @classmethod
    def _normalise_id(cls, value: object) -> str:
        return str(value or "").strip() or "auto-unknown"

    @field_validator("line_indices", mode="before")
    @classmethod
    def _validate_indices(cls, value: Iterable[object] | None) -> list[int]:
        if value is None:
            return []
        try:
            return sorted({int(item) for item in value})  # type: ignore[call-overload]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid line index in {value!r}") from exc

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Problem:
        return cls.model_validate(data)


class PageAssignment(BaseModel):
    page_number: int
    problems: list[Problem] = Field(default_factory=list)
    unassigned_lines: list[int] = Field(default_factory=list)
    raw_response: dict[str, object] | None = None
    raw_response_text: str | None = None

    model_config = ConfigDict(extra="ignore")


class LLMProblemExtractor:
    """Thin OpenAI API client tailored for this pipeline."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_VISION_MODEL
        self.base_url = (base_url or config.OPENAI_API_BASE).rstrip("/")
        self.timeout = timeout or config.OPENAI_TIMEOUT_SECONDS
        self._client: httpx.AsyncClient | None = client
        self._owns_client = client is None
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    async def __aenter__(self) -> LLMProblemExtractor:
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client and self._owns_client:
            await self._client.aclose()
        if self._owns_client:
            self._client = None

    # ------------------------------------------------------------------ public
    async def extract_page(
        self,
        *,
        page_number: int,
        image_path: Path,
        lines: Iterable[TextLine],
    ) -> PageAssignment:
        image_b64 = await asyncio.to_thread(_image_to_base64, image_path)
        prepared_lines = list(lines)
        prompt_text = _format_lines_for_prompt(prepared_lines)
        response_text = await self._request_completion(
            page_number, image_b64, prompt_text
        )
        parsed = self._parse_response(response_text)
        return self._build_assignment(
            page_number, parsed, prepared_lines, response_text
        )

    # ------------------------------ refinement: outlier removal / contiguity
    async def refine_groups(
        self,
        *,
        page_number: int,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """
        Ask the model to refine initial problem groupings by removing outlier
        lines and adding obvious neighbors so that each problem forms a single
        contiguous block within the main text column.

        Returns a dict with shape:
        {
          "page": <int>,
          "corrections": [
            {"problem_id": str, "remove": [int], "add": [int]}
          ]
        }

        The method is best-effort: any API error raises RuntimeError to the
        caller so the pipeline can decide how to proceed.
        """
        client = await self._ensure_client()
        url = "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        instructions = (
            "You are cleaning groups of OCR lines for a textbook page. "
            "For each problem, ensure selected line indices form a single, "
            "contiguous block inside the main text column (content_roi). "
            "Rules: keep only lines primarily inside content_roi; use the "
            "neighbors graph to maintain contiguity; if multiple components "
            "exist, keep the component that contains a header-like line or, "
            "if none, the largest component; you may only add indices that "
            "appear in the lines list and are neighbors of existing indices; "
            "do not invent ids or indices. Respond with strict JSON only: "
            "{\"page\": int, \"corrections\":[{" \
            "problem_id\": str, \"remove\":[int], \"add\":[int]}...]}."
        )

        prompt_text = json.dumps(payload, ensure_ascii=False)
        request_payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt_text},
            ],
        }

        logger.debug(
            f"Sending refine request for page {page_number} (payload size {len(prompt_text)} chars)",
        )
        try:
            response = await client.post(url, json=request_payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network
            body = getattr(exc.response, "text", "") if hasattr(exc, "response") else ""
            raise RuntimeError(f"OpenAI refine request failed: {exc!r} {body}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            text = str(content) if content is not None else "{}"
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Malformed refine response: {data}") from exc

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Be tolerant to minor wrapping errors by extracting the first JSON object
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start == -1 or brace_end == -1:
                raise
            snippet = text[brace_start : brace_end + 1]
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else {}

    # ----------------------------------------------------------------- helpers
    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def _request_completion(
        self, page_number: int, image_b64: str, prompt_text: str
    ) -> str:
        client = await self._ensure_client()
        url = "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"OCR lines for page {page_number}:\n{prompt_text}\n\n"
                                "Please return structured JSON as specified."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }

        logger.debug(
            f"Sending request for page {page_number} with {count_lines(prompt_text)} lines",
        )
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            body = getattr(exc.response, "text", "") if hasattr(exc, "response") else ""
            raise RuntimeError(f"OpenAI request failed: {exc!r} {body}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            return str(content) if content is not None else ""
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Malformed OpenAI response: {data}") from exc

    def _parse_response(self, response_text: str) -> dict[str, object]:
        response_text = response_text.strip()
        if not response_text:
            raise ValueError("Empty response from OpenAI.")

        try:
            parsed = json.loads(response_text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            brace_start = response_text.find("{")
            brace_end = response_text.rfind("}")
            if brace_start == -1 or brace_end == -1:
                raise
            snippet = response_text[brace_start : brace_end + 1]
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else {}

    def _build_assignment(
        self,
        page_number: int,
        parsed: dict[str, object],
        lines: list[TextLine],
        response_text: str,
    ) -> PageAssignment:
        problems_raw = parsed.get("problems", [])
        if not isinstance(problems_raw, list):
            raise ValueError("`problems` field must be a list.")
        problems = []
        seen_indices: set[int] = set()
        for problem_data in problems_raw:
            if not isinstance(problem_data, dict):
                logger.warning("Skipping malformed problem entry: %r", problem_data)
                continue
            problem = Problem.from_dict(problem_data)
            problems.append(problem)
            seen_indices.update(problem.line_indices)

        total_indices = {idx for idx, _ in enumerate(lines)}
        unassigned = sorted(total_indices - seen_indices)

        return PageAssignment(
            page_number=page_number,
            problems=problems,
            unassigned_lines=unassigned,
            raw_response=parsed,
            raw_response_text=response_text,
        )


# ---------------------------------------------------------------------------
# Prompt helpers


def _format_lines_for_prompt(lines: Iterable[TextLine]) -> str:
    formatted_lines: list[str] = []
    for index, line in enumerate(lines):
        text = line.text.replace("\n", " ").strip()
        bbox = ",".join(f"{coord:.0f}" for coord in line.bbox)
        formatted_lines.append(f"{index}: {text or '<empty>'} | bbox={bbox}")
    return "\n".join(formatted_lines)


def _image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return encoded


def count_lines(prompt_text: str) -> int:
    return 0 if not prompt_text else prompt_text.count("\n") + 1
