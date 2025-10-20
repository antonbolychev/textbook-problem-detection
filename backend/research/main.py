from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from loguru import logger

from detector import run_pipeline

current_dir = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and visualise textbook problems using OpenAI."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=current_dir / Path("data/math.pdf"),
        help="Path to the source PDF (default: data/math.pdf).",
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        default=current_dir / Path("ocr_result_pass.json"),
        help="Path to the OCR JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=current_dir / Path("run_artifacts"),
        help="Directory used for outputs (JSON + visualisations).",
    )
    parser.add_argument(
        "--no-visualise",
        action="store_true",
        help="Skip drawing annotated images.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    logger.info("Starting pipeline…")
    result = await run_pipeline(
        pdf_path=args.pdf,
        ocr_result_path=args.ocr,
        working_dir=args.output_dir,
        visualise=not args.no_visualise,
    )

    logger.info(f"Saved page assignments to {result.output_json}")
    if result.raw_outputs_dir:
        logger.info(f"Stored raw LLM responses in {result.raw_outputs_dir}")
    if result.visualisations:
        logger.info(
            f"Generated {len(result.visualisations)} annotated images in {result.visualisations[0].parent}"
        )
    if result.problem_visualisations:
        logger.info(
            f"Generated {len(result.problem_visualisations)} problem-level overlays in {result.problem_visualisations[0].parent}"
        )
    logger.info("Pipeline finished successfully.")


def main() -> None:
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
