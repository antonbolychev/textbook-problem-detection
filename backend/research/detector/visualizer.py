"""
Visualization helpers for manual inspection of problem assignments.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from pathlib import Path

from detector.llm import PageAssignment
from detector.models import TextLine

logger = logging.getLogger(__name__)


def annotate_page_assignments(
    *,
    lines: Iterable[TextLine],
    assignment: PageAssignment,
    image_path: Path,
    output_dir: Path,
    image_bbox: Iterable[float] | None = None,
    line_match_status: dict[int, bool] | None = None,
) -> Path:
    """
    Draw the problem identifier next to each OCR line on ``image_path``.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Pillow is required for visualisation. Install it with "
            "`pip install pillow`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    page_bounds = _normalise_image_bbox(image_bbox, image.size)
    scale_x, scale_y = _calculate_scale_factors(page_bounds, image.size)

    index_to_problem = _build_line_problem_lookup(assignment)
    line_match_status = line_match_status or {}
    for index, line in enumerate(lines):
        bbox = _extract_bbox(line)
        if not bbox:
            logger.debug("Skipping line %s without bbox", index)
            continue

        scaled_bbox = _scale_bbox(bbox, page_bounds, scale_x, scale_y, image.size)
        if not scaled_bbox:
            logger.debug("Skipping line %s due to invalid scaled bbox", index)
            continue

        problem_id = index_to_problem.get(index)
        match_status = line_match_status.get(index)
        if problem_id:
            base_colour = _colour_for_problem(problem_id)
            if match_status is False:
                colour = (220, 70, 70)
            else:
                colour = base_colour
        else:
            colour = (180, 180, 180)
        outline: tuple[int, int, int] = tuple(
            max(channel - 20, 0) for channel in colour
        )  # type: ignore[assignment]

        draw.rectangle(scaled_bbox, outline=outline, width=2)
        label = problem_id or "none"
        if match_status is False:
            label = f"{label}!"
        text_position = (scaled_bbox[0] + 4, scaled_bbox[1] + 4)
        text_bbox = draw.textbbox(text_position, label, font=font)
        padded_bbox = (
            text_bbox[0] - 2,
            text_bbox[1] - 2,
            text_bbox[2] + 2,
            text_bbox[3] + 2,
        )
        draw.rectangle(padded_bbox, fill=(0, 0, 0, 160))
        draw.text(text_position, label, fill=colour, font=font)

    output_path = output_dir / f"{image_path.stem}_annotated.png"
    image.save(output_path, format="PNG")
    return output_path


def annotate_problem_groups(
    *,
    lines: Iterable[TextLine],
    assignment: PageAssignment,
    image_path: Path,
    output_dir: Path,
    image_bbox: Iterable[float] | None = None,
    line_match_status: dict[int, bool] | None = None,
) -> Path:
    """
    Draw aggregated bounding boxes for each problem by spanning all assigned lines.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "Pillow is required for visualisation. Install it with "
            "`pip install pillow`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    page_bounds = _normalise_image_bbox(image_bbox, image.size)
    scale_x, scale_y = _calculate_scale_factors(page_bounds, image.size)

    lines_list = list(lines)
    line_match_status = line_match_status or {}

    for problem in assignment.problems:
        scaled_boxes: list[tuple[int, int, int, int]] = []
        mismatch = False
        for index in problem.line_indices:
            if index >= len(lines_list):
                continue
            bbox = _extract_bbox(lines_list[index])
            if not bbox:
                continue
            scaled = _scale_bbox(bbox, page_bounds, scale_x, scale_y, image.size)
            if not scaled:
                continue
            scaled_boxes.append(scaled)
            if line_match_status.get(index) is False:
                mismatch = True

        if not scaled_boxes:
            logger.debug("Problem %s has no drawable boxes", problem.problem_id)
            continue

        min_x = min(box[0] for box in scaled_boxes)
        min_y = min(box[1] for box in scaled_boxes)
        max_x = max(box[2] for box in scaled_boxes)
        max_y = max(box[3] for box in scaled_boxes)

        colour = _colour_for_problem(problem.problem_id)
        if mismatch:
            colour = (220, 70, 70)
        outline: tuple[int, int, int] = tuple(
            max(channel - 20, 0) for channel in colour
        )  # type: ignore[assignment]

        draw.rectangle((min_x, min_y, max_x, max_y), outline=outline, width=3)
        label = problem.problem_id
        if mismatch:
            label = f"{label}!"
        text_position = (min_x + 6, min_y + 6)
        text_bbox = draw.textbbox(text_position, label, font=font)
        padded_bbox = (
            text_bbox[0] - 3,
            text_bbox[1] - 3,
            text_bbox[2] + 3,
            text_bbox[3] + 3,
        )
        draw.rectangle(padded_bbox, fill=(0, 0, 0, 180))
        draw.text(text_position, label, fill=colour, font=font)

    output_path = output_dir / f"{image_path.stem}_problems.png"
    image.save(output_path, format="PNG")
    return output_path


def _build_line_problem_lookup(assignment: PageAssignment) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for problem in assignment.problems:
        for index in problem.line_indices:
            mapping[index] = problem.problem_id
    return mapping


def _extract_bbox(line: TextLine) -> tuple[int, int, int, int] | None:
    if not line.bbox or len(line.bbox) < 4:
        return None
    x0, y0, x1, y1 = line.bbox[:4]
    return int(x0), int(y0), int(x1), int(y1)


def _colour_for_problem(problem_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(problem_id.encode("utf-8")).digest()
    r, g, b = digest[0], digest[5], digest[10]
    return (100 + r % 155, 100 + g % 155, 100 + b % 155)


def _normalise_image_bbox(
    image_bbox: Iterable[float] | None, image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    if image_bbox is None:
        return 0.0, 0.0, float(image_size[0]), float(image_size[1])
    coords = list(image_bbox)
    if len(coords) < 4:
        return 0.0, 0.0, float(image_size[0]), float(image_size[1])
    x0, y0, x1, y1 = coords[:4]
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, float(image_size[0]), float(image_size[1])
    return float(x0), float(y0), float(x1), float(y1)


def _calculate_scale_factors(
    bounds: tuple[float, float, float, float], image_size: tuple[int, int]
) -> tuple[float, float]:
    x0, y0, x1, y1 = bounds
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    return image_size[0] / width, image_size[1] / height


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    bounds: tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    min_x, min_y, _, _ = bounds

    scaled = (
        (x0 - min_x) * scale_x,
        (y0 - min_y) * scale_y,
        (x1 - min_x) * scale_x,
        (y1 - min_y) * scale_y,
    )

    if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
        return None

    clipped = (
        int(max(0, min(image_size[0] - 1, round(scaled[0])))),
        int(max(0, min(image_size[1] - 1, round(scaled[1])))),
        int(max(0, min(image_size[0] - 1, round(scaled[2])))),
        int(max(0, min(image_size[1] - 1, round(scaled[3])))),
    )
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None
