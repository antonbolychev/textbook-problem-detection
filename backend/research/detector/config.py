"""
Configuration helpers for the textbook problem detection pipeline.

Environment variables override the default values when set to make the project
usable without editing source files (useful for keeping secrets out of VCS).
"""

from __future__ import annotations

from app.settings import BACKEND_ROOT, settings

# API configuration ---------------------------------------------------------

OPENAI_API_KEY = settings.openai_api_key
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_VISION_MODEL = "gpt-4.1"
OPENAI_TIMEOUT_SECONDS = 60
MAX_CONCURRENT_LLM_REQUESTS = settings.llm_max_concurrency

# PDF/image processing ------------------------------------------------------
WORK_DIR = BACKEND_ROOT / "research" / "work"
PDF_OUTPUT_DIR = WORK_DIR / "page_images"
PDF_RENDER_DPI = 220

# Prompt control ------------------------------------------------------------
MAX_OCR_LINES_PER_PROMPT = 160
SYSTEM_PROMPT_PATH = BACKEND_ROOT / "research" / "system_prompt.md"
