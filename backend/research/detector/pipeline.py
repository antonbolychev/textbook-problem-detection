"""
High-level orchestration for the textbook problem detection workflow.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from detector import config
from detector.llm import LLMProblemExtractor, PageAssignment
from detector.models import OCRResult, TextLine
from detector.pdf_utils import render_pdf_to_images
from detector.visualizer import annotate_page_assignments, annotate_problem_groups

from detector.report_models import LineReport, PageReport, ProblemLineEntry, ProblemReport

from loguru import logger

@dataclass(slots=True)
class PipelineResult:
    page_assignments: list[PageAssignment]
    image_paths: list[Path]
    output_json: Path
    visualisations: list[Path]
    problem_visualisations: list[Path]
    raw_outputs_dir: Path | None = None


@dataclass(slots=True)
class _PageJob:
    order: int
    page_number: int
    image_path: Path
    lines: list[TextLine]


async def run_pipeline(
    *,
    pdf_path: Path,
    ocr_result_path: Path,
    working_dir: Path | None = None,
    visualise: bool = True,
    max_parallel_requests: int | None = None,
) -> PipelineResult:
    working_dir = Path(working_dir) if working_dir else config.WORK_DIR
    pdf_path = Path(pdf_path)
    ocr_result_path = Path(ocr_result_path)
    await asyncio.to_thread(working_dir.mkdir, parents=True, exist_ok=True)

    logger.info(f"Loading OCR result from {ocr_result_path}")
    ocr_result = await asyncio.to_thread(OCRResult.from_file, ocr_result_path)

    logger.info(f"Rendering {ocr_result.page_count} pages from {pdf_path.name}")
    image_dir = working_dir / "page_images"
    image_paths = await asyncio.to_thread(
        render_pdf_to_images,
        pdf_path,
        image_dir,
        dpi=config.PDF_RENDER_DPI,
        overwrite=False,
    )
    if len(image_paths) < len(ocr_result.pages):
        logger.warning(
            f"Rendered {len(image_paths)} images, but OCR has {len(ocr_result.pages)} pages. Results may misalign.",
        )

    jobs: list[_PageJob] = []
    per_page_images: list[Path] = []
    lines_for_pages: list[list[TextLine]] = []
    image_bboxes: list[list[float]] = []

    async with LLMProblemExtractor() as extractor:
        for index, page in enumerate(ocr_result.pages):
            image_for_page = _resolve_image_for_page(image_paths, index)
            per_page_images.append(image_for_page)
            image_bboxes.append(page.image_bbox)
            if len(page.text_lines) > config.MAX_OCR_LINES_PER_PROMPT:
                logger.warning(
                    f"Page {page.page} has {len(page.text_lines)} lines (limit {config.MAX_OCR_LINES_PER_PROMPT}). The prompt will be truncated.",
                )
            subset = list(page.text_lines[: config.MAX_OCR_LINES_PER_PROMPT])
            lines_for_pages.append(subset)
            jobs.append(
                _PageJob(
                    order=index,
                    page_number=page.page or index + 1,
                    image_path=image_for_page,
                    lines=subset,
                )
            )

        concurrency = max_parallel_requests or config.MAX_CONCURRENT_LLM_REQUESTS
        if concurrency <= 0:
            concurrency = 1
        assignments = await _extract_assignments(extractor, jobs, concurrency)

        # Optional refinement pass to remove outliers / add neighbors
        if config.REFINEMENT_ENABLED:
            try:
                assignments = await _refine_assignments(
                    extractor,
                    assignments,
                    lines_for_pages,
                    per_page_images,
                    image_bboxes,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Refinement step failed: {e}")

    output_dir = working_dir / "outputs"
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    json_path = output_dir / "problem_assignments.json"
    page_reports, line_match_maps = _build_page_reports(
        assignments, lines_for_pages, per_page_images, image_bboxes
    )
    page_report_payload = [report.model_dump(mode="json") for report in page_reports]
    await asyncio.to_thread(_write_json, json_path, page_report_payload)

    raw_responses_dir = output_dir / "llm_raw"
    await asyncio.to_thread(raw_responses_dir.mkdir, parents=True, exist_ok=True)
    raw_tasks = [
        asyncio.to_thread(
            _write_raw_response,
            raw_responses_dir,
            assignment,
        )
        for assignment in assignments
        if assignment.raw_response_text
    ]
    if raw_tasks:
        await asyncio.gather(*raw_tasks)

    visualisations: list[Path] = []
    problem_visualisations: list[Path] = []
    if visualise:
        vis_dir = output_dir / "visualisations"
        problems_vis_dir = output_dir / "visualisations_problems"
        for lines, assignment, image, bbox, match_map in zip(
            lines_for_pages,
            assignments,
            per_page_images,
            image_bboxes,
            line_match_maps,
            strict=True,
        ):
            visualisations.append(
                await asyncio.to_thread(
                    annotate_page_assignments,
                    lines=lines,
                    assignment=assignment,
                    image_path=image,
                    output_dir=vis_dir,
                    image_bbox=bbox,
                    line_match_status=match_map,
                )
            )
            problem_visualisations.append(
                await asyncio.to_thread(
                    annotate_problem_groups,
                    lines=lines,
                    assignment=assignment,
                    image_path=image,
                    output_dir=problems_vis_dir,
                    image_bbox=bbox,
                    line_match_status=match_map,
                )
            )

    return PipelineResult(
        page_assignments=assignments,
        image_paths=per_page_images,
        output_json=json_path,
        visualisations=visualisations,
        problem_visualisations=problem_visualisations,
        raw_outputs_dir=raw_responses_dir,
    )


def _resolve_image_for_page(image_paths: list[Path], index: int) -> Path:
    if not image_paths:
        raise ValueError("No images were rendered from the PDF.")
    if index < len(image_paths):
        return image_paths[index]
    logger.warning(
        "Page index %s missing an image; using the last rendered image instead.",
        index,
    )
    return image_paths[-1]


def _calculate_problem_bbox(
    problem: object,  # Problem from PageAssignment
    lines: list[TextLine],
    image_path: Path,
    image_bbox: list[float] | None,
) -> dict[str, float] | None:
    """
    Calculate the aggregated bounding box for a problem by spanning all its assigned lines.
    Returns bbox in format {x, y, width, height} suitable for frontend display.
    """
    try:
        from PIL import Image
    except ModuleNotFoundError:
        logger.warning("PIL not available, skipping bbox calculation")
        return None
    
    try:
        image = Image.open(image_path)
        image_size = image.size
        image.close()
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None
    
    # Normalize image bbox (same logic as visualizer)
    if image_bbox is None or len(image_bbox) < 4:
        page_bounds = (0.0, 0.0, float(image_size[0]), float(image_size[1]))
    else:
        x0, y0, x1, y1 = image_bbox[:4]
        if x1 <= x0 or y1 <= y0:
            page_bounds = (0.0, 0.0, float(image_size[0]), float(image_size[1]))
        else:
            page_bounds = (float(x0), float(y0), float(x1), float(y1))
    
    # Calculate scale factors
    bounds_width = max(page_bounds[2] - page_bounds[0], 1.0)
    bounds_height = max(page_bounds[3] - page_bounds[1], 1.0)
    scale_x = image_size[0] / bounds_width
    scale_y = image_size[1] / bounds_height
    
    # Collect scaled bboxes for all lines in this problem
    scaled_boxes: list[tuple[int, int, int, int]] = []
    for index in problem.line_indices:
        if index >= len(lines):
            continue
        line = lines[index]
        if not line.bbox or len(line.bbox) < 4:
            continue
        
        # Extract and scale bbox
        lx0, ly0, lx1, ly1 = line.bbox[:4]
        scaled = (
            (lx0 - page_bounds[0]) * scale_x,
            (ly0 - page_bounds[1]) * scale_y,
            (lx1 - page_bounds[0]) * scale_x,
            (ly1 - page_bounds[1]) * scale_y,
        )
        
        if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
            continue
        
        # Clip to image bounds
        clipped = (
            int(max(0, min(image_size[0] - 1, round(scaled[0])))),
            int(max(0, min(image_size[1] - 1, round(scaled[1])))),
            int(max(0, min(image_size[0] - 1, round(scaled[2])))),
            int(max(0, min(image_size[1] - 1, round(scaled[3])))),
        )
        
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            scaled_boxes.append(clipped)
    
    if not scaled_boxes:
        return None
    
    # Calculate aggregated bounding box
    min_x = min(box[0] for box in scaled_boxes)
    min_y = min(box[1] for box in scaled_boxes)
    max_x = max(box[2] for box in scaled_boxes)
    max_y = max(box[3] for box in scaled_boxes)
    
    # Return in frontend-friendly format
    return {
        "x": float(min_x),
        "y": float(min_y),
        "width": float(max_x - min_x),
        "height": float(max_y - min_y),
    }


# ---------------------------------------------------------------------------
# Refinement helpers

def _get_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ModuleNotFoundError:  # pragma: no cover - depends on env
        return None
    try:
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None


def _scale_all_line_bboxes(
    *,
    lines: list[TextLine],
    image_path: Path,
    image_bbox: list[float] | None,
) -> tuple[list[tuple[int, int, int, int] | None], tuple[int, int] | None]:
    """Return a list of scaled/clipped bboxes (page pixel coords) for lines.

    Returns a tuple of (scaled_boxes, image_size). Items can be ``None`` when a
    line lacks geometry.
    """
    size = _get_image_size(image_path)
    if size is None:
        return [None for _ in lines], None
    # Normalise page bounds like visualizer
    if image_bbox is None or len(image_bbox) < 4:
        bounds = (0.0, 0.0, float(size[0]), float(size[1]))
    else:
        x0, y0, x1, y1 = image_bbox[:4]
        if x1 <= x0 or y1 <= y0:
            bounds = (0.0, 0.0, float(size[0]), float(size[1]))
        else:
            bounds = (float(x0), float(y0), float(x1), float(y1))
    bw = max(bounds[2] - bounds[0], 1.0)
    bh = max(bounds[3] - bounds[1], 1.0)
    sx, sy = size[0] / bw, size[1] / bh

    result: list[tuple[int, int, int, int] | None] = []
    for line in lines:
        if not line.bbox or len(line.bbox) < 4:
            result.append(None)
            continue
        x0, y0, x1, y1 = line.bbox[:4]
        scaled = (
            (x0 - bounds[0]) * sx,
            (y0 - bounds[1]) * sy,
            (x1 - bounds[0]) * sx,
            (y1 - bounds[1]) * sy,
        )
        if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
            result.append(None)
            continue
        clipped = (
            int(max(0, min(size[0] - 1, round(scaled[0])))),
            int(max(0, min(size[1] - 1, round(scaled[1])))),
            int(max(0, min(size[0] - 1, round(scaled[2])))),
            int(max(0, min(size[1] - 1, round(scaled[3])))),
        )
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result.append(clipped)
        else:
            result.append(None)
    return result, size


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    q = min(max(q, 0.0), 1.0)
    i = (len(v) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    if hi == lo:
        return float(v[lo])
    frac = i - lo
    return float(v[lo] * (1 - frac) + v[hi] * frac)


def _estimate_content_roi(
    boxes: list[tuple[int, int, int, int] | None], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    # Use robust quantiles over x0/x1 to trim margins
    xs0 = [b[0] for b in boxes if b]
    xs1 = [b[2] for b in boxes if b]
    if not xs0 or not xs1:
        return (0, 0, image_size[0], image_size[1])
    x0 = int(_quantile(xs0, 0.05))
    x1 = int(_quantile(xs1, 0.95))
    x0 = max(0, min(x0, image_size[0] - 1))
    x1 = max(x0 + 1, min(x1, image_size[0]))
    return (x0, 0, x1, image_size[1])


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0, (ax1 - ax0)) * max(0, (ay1 - ay0))
    b_area = max(0, (bx1 - bx0)) * max(0, (by1 - by0))
    denom = a_area + b_area - inter
    return inter / denom if denom > 0 else 0.0


def _build_neighbors(
    boxes: list[tuple[int, int, int, int] | None]
) -> dict[int, list[int]]:
    # Simple proximity graph: link lines that vertically overlap or are close
    indices = [i for i, b in enumerate(boxes) if b]
    if not indices:
        return {}
    heights = [boxes[i][3] - boxes[i][1] for i in indices]
    median_h = sorted(heights)[len(heights) // 2]
    threshold = max(6, int(median_h * 1.5))
    neighbors: dict[int, list[int]] = {i: [] for i in indices}
    for i in indices:
        x0, y0, x1, y1 = boxes[i]  # type: ignore[index]
        for j in indices:
            if j <= i:
                continue
            u0, v0, u1, v1 = boxes[j]  # type: ignore[index]
            # vertical closeness or overlap
            vert_gap = max(0, max(y0, v0) - min(y1, v1))
            horiz_overlap = min(x1, u1) - max(x0, u0)
            if vert_gap <= threshold and horiz_overlap > 0:
                neighbors[i].append(j)
                neighbors[j].append(i)
            else:
                # fallback: center distance heuristic
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                dx, dy = (u0 + u1) // 2 - cx, (v0 + v1) // 2 - cy
                if abs(dx) + abs(dy) <= threshold:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
    return neighbors


def _box_in_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    rx0, ry0, rx1, ry1 = roi
    bx0, by0, bx1, by1 = box
    roi_area = max(1, (rx1 - rx0) * (ry1 - ry0))
    ix0, iy0 = max(rx0, bx0), max(ry0, by0)
    ix1, iy1 = min(rx1, bx1), min(ry1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter >= 0.5 * ((bx1 - bx0) * (by1 - by0)) and inter > 0 and roi_area > 0


def _build_refine_payload(
    *,
    page_number: int,
    lines: list[TextLine],
    assignment: PageAssignment,
    image_path: Path,
    image_bbox: list[float] | None,
) -> dict[str, object] | None:
    scaled_boxes, size = _scale_all_line_bboxes(
        lines=lines, image_path=image_path, image_bbox=image_bbox
    )
    if size is None:
        return None
    roi = _estimate_content_roi(
        scaled_boxes, size
    )
    neighbors = _build_neighbors(scaled_boxes)

    def _clean_text(text: str) -> str:
        # strip very simple markup and truncate
        out: list[str] = []
        inside = False
        for ch in text:
            if ch == "<":
                inside = True
                continue
            if ch == ">" and inside:
                inside = False
                continue
            if not inside:
                out.append(ch)
        s = "".join(out).replace("\n", " ").strip()
        if len(s) > config.REFINEMENT_JSON_TRUNCATE_TEXT:
            s = s[: config.REFINEMENT_JSON_TRUNCATE_TEXT] + "…"
        return s

    payload_lines: list[dict[str, object]] = []
    for idx, (line, box) in enumerate(zip(lines, scaled_boxes)):
        entry: dict[str, object] = {"i": idx}
        if box:
            x0, y0, x1, y1 = box
            entry["bbox"] = {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}
            entry["in_roi"] = _box_in_roi(box, roi)
            entry["neighbors"] = neighbors.get(idx, [])
        else:
            entry["bbox"] = None
            entry["in_roi"] = False
            entry["neighbors"] = []
        entry["text"] = _clean_text(line.text or "")
        payload_lines.append(entry)

    payload_groups: list[dict[str, object]] = []
    for problem in assignment.problems:
        payload_groups.append(
            {"problem_id": problem.problem_id, "indices": list(problem.line_indices)}
        )

    return {
        "page": page_number,
        "image_size": [size[0], size[1]],
        "content_roi": {"x0": roi[0], "y0": roi[1], "x1": roi[2], "y1": roi[3]},
        "lines": payload_lines,
        "groups": payload_groups,
    }


async def _refine_assignments(
    extractor: LLMProblemExtractor,
    assignments: list[PageAssignment],
    page_lines: list[list[TextLine]],
    image_paths: list[Path],
    image_bboxes: list[list[float]],
) -> list[PageAssignment]:
    if not assignments:
        return assignments

    for idx, assignment in enumerate(assignments):
        try:
            payload = _build_refine_payload(
                page_number=assignment.page_number,
                lines=page_lines[idx],
                assignment=assignment,
                image_path=image_paths[idx],
                image_bbox=image_bboxes[idx] if idx < len(image_bboxes) else None,
            )
            if payload is None:
                continue
            response = await extractor.refine_groups(
                page_number=assignment.page_number, payload=payload
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Skipping refinement for page {assignment.page_number}: {e}")
            continue

        try:
            corrections = response.get("corrections", [])  # type: ignore[assignment]
            if not isinstance(corrections, list):
                continue
        except Exception:
            continue

        # Apply corrections
        by_id = {p.problem_id: p for p in assignment.problems}
        for item in corrections:
            try:
                pid = str(item.get("problem_id"))
                remove = {int(i) for i in (item.get("remove") or [])}
                add = {int(i) for i in (item.get("add") or [])}
            except Exception:
                continue
            if pid not in by_id:
                continue
            problem = by_id[pid]
            current = set(problem.line_indices)
            # Keep indices within bounds
            max_index = len(page_lines[idx]) - 1
            filtered_add = {i for i in add if 0 <= i <= max_index}
            filtered_remove = {i for i in remove if 0 <= i <= max_index}
            updated = sorted((current - filtered_remove) | filtered_add)
            problem.line_indices = updated  # mutate in-place

    return assignments


def _build_page_reports(
    assignments: list[PageAssignment],
    page_lines: list[list[TextLine]],
    image_paths: list[Path] | None = None,
    image_bboxes: list[list[float]] | None = None,
) -> tuple[list[PageReport], list[dict[int, bool]]]:
    reports: list[PageReport] = []
    match_maps: list[dict[int, bool]] = []
    for idx, (assignment, lines) in enumerate(zip(assignments, page_lines, strict=True)):
        line_text_lookup = {idx: line.text for idx, line in enumerate(lines)}

        line_to_problem: dict[int, str] = {}
        line_match_map: dict[int, bool] = {}
        problem_reports: list[ProblemReport] = []
        for problem in assignment.problems:
            problem_line_entries: list[ProblemLineEntry] = []
            for index in problem.line_indices:
                text = line_text_lookup.get(index, "")
                problem_line_entries.append(
                    ProblemLineEntry(
                        index=index,
                        text=text,
                        matches_problem_text=None,
                    )
                )
                line_to_problem[index] = problem.problem_id
                line_match_map.setdefault(index, False)

            full_text = problem.problem_text or "\n".join(
                entry.text for entry in problem_line_entries if entry.text
            )
            for entry in problem_line_entries:
                matches_text = _line_matches_problem_text(entry.text, full_text)
                entry.matches_problem_text = matches_text
                line_match_map[entry.index] = matches_text

            # Calculate bounding box if image data is available
            bbox = None
            if image_paths and image_bboxes and idx < len(image_paths) and idx < len(image_bboxes):
                bbox = _calculate_problem_bbox(
                    problem=problem,
                    lines=lines,
                    image_path=image_paths[idx],
                    image_bbox=image_bboxes[idx],
                )

            problem_reports.append(
                ProblemReport(
                    problem_id=problem.problem_id,
                    line_indices=problem.line_indices,
                    lines=problem_line_entries,
                    full_text=full_text,
                    bbox=bbox,
                )
            )

        line_reports: list[LineReport] = []
        for index, line in enumerate(lines):
            assigned_problem_id = line_to_problem.get(index)
            matches_problem_text = line_match_map.get(index)
            line_reports.append(
                LineReport(
                    index=index,
                    text=line.text,
                    assigned_problem_id=assigned_problem_id,
                    belongs_to_problem=assigned_problem_id is not None,
                    matches_problem_text=matches_problem_text,
                )
            )

        reports.append(
            PageReport(
                page=assignment.page_number,
                problems=problem_reports,
                lines=line_reports,
                unassigned_lines=assignment.unassigned_lines,
            )
        )
        match_maps.append(line_match_map)
    return reports, match_maps


def _write_json(path: Path, data: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _write_raw_response(directory: Path, assignment: PageAssignment) -> None:
    if not assignment.raw_response_text:
        return
    raw_path = directory / f"page_{assignment.page_number:03}.json"
    raw_path.write_text(assignment.raw_response_text, encoding="utf-8")


async def _extract_assignments(
    extractor: LLMProblemExtractor,
    jobs: list[_PageJob],
    concurrency: int,
) -> list[PageAssignment]:
    if not jobs:
        return []
    semaphore = asyncio.Semaphore(concurrency)
    assignments: list[PageAssignment | None] = [None] * len(jobs)

    async def run_job(job: _PageJob) -> None:
        async with semaphore:
            assignment = await extractor.extract_page(
                page_number=job.page_number,
                image_path=job.image_path,
                lines=job.lines,
            )
        assignments[job.order] = assignment

    await asyncio.gather(*(run_job(job) for job in jobs))
    return [assignment for assignment in assignments if assignment is not None]


def _line_matches_problem_text(line_text: str, problem_text: str | None) -> bool:
    if not line_text.strip():
        return True
    if not problem_text:
        return False
    normalised_line = _normalise_text(line_text)
    normalised_problem = _normalise_text(problem_text)
    return normalised_line in normalised_problem


def _normalise_text(text: str) -> str:
    cleaned = _strip_markup(text)
    return "".join(ch.lower() for ch in cleaned if ch.isalnum())


def _strip_markup(text: str) -> str:
    result: list[str] = []
    inside = False
    for ch in text:
        if ch == "<":
            inside = True
            continue
        if ch == ">" and inside:
            inside = False
            continue
        if not inside:
            result.append(ch)
    return "".join(result)
