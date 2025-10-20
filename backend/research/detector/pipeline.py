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
