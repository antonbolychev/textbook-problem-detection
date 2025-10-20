"""
Utilities for turning PDF textbooks into page images.

The helper first tries to use :mod:`pdf2image` (pure Python) and falls back to
calling ``pdftoppm`` if the Python dependency is missing.  Both branches raise a
clear error when neither option is available so the user knows how to install
the required tooling.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def render_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 220,
    overwrite: bool = False,
) -> list[Path]:
    """
    Render ``pdf_path`` into PNG images inside ``output_dir``.

    Parameters
    ----------
    pdf_path
        Source document to render.
    output_dir
        Destination directory for generated images.
    dpi
        Resolution to request from the renderer.
    overwrite
        When ``False`` existing images are kept and returned as-is so repeated
        runs stay incremental.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = sorted(output_dir.glob(f"{pdf_path.stem}_page_*.png"))
    if existing_images and not overwrite:
        logger.info("Re-using %s pre-rendered images", len(existing_images))
        return existing_images

    try:
        return _render_with_pdf2image(pdf_path, output_dir, dpi=dpi)
    except ModuleNotFoundError:
        logger.debug("pdf2image not installed, falling back to pdftoppm")
        return _render_with_pdftoppm(pdf_path, output_dir, dpi=dpi)


def _render_with_pdf2image(pdf_path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path), dpi=dpi)
    rendered_paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        target = output_dir / f"{pdf_path.stem}_page_{index:03}.png"
        image.save(target, "PNG")
        rendered_paths.append(target)
    return rendered_paths


def _render_with_pdftoppm(pdf_path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
    target_prefix = output_dir / pdf_path.stem
    command = [
        "pdftoppm",
        "-png",
        "-r",
        str(dpi),
        str(pdf_path),
        str(target_prefix),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:  # pragma: no cover - depends on system
        raise RuntimeError(
            "Neither pdf2image nor pdftoppm are available. Install pdf2image "
            "(`pip install pdf2image poppler-utils`) or add pdftoppm to PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess
        raise RuntimeError(
            f"pdftoppm failed with exit code {exc.returncode}: {exc.stderr.decode()}"
        ) from exc

    return sorted(output_dir.glob(f"{pdf_path.stem}-*.png"))
