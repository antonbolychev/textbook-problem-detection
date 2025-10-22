from __future__ import annotations

import asyncio
import sys
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path

import mlflow
import tyro
from loguru import logger

from detector import run_pipeline
from mlflow_utils import setup_mlflow, auto_commit_research

current_dir = Path(__file__).resolve().parent


@dataclass
class CLIArgs:
    """CLI parameters for the research pipeline."""

    pdf: Path = current_dir / Path("data/math.pdf")
    ocr: Path = current_dir / Path("ocr_result_pass.json")
    output_dir: Path = current_dir / Path("run_artifacts")
    visualise: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    tracking_uri: str | None = str((current_dir / "mlruns").resolve())
    experiment_name: str = "research"


def _git_commit() -> str:
    """Return current repository commit hash."""
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        .strip()
        .decode("utf-8")
    )


def _normalise_params_for_logging(args: CLIArgs) -> dict[str, str | int | float | bool]:
    """Convert dataclass args to primitives suitable for logging/MLflow."""
    payload: dict[str, str | int | float | bool] = {}
    for key, value in asdict(args).items():
        if isinstance(value, Path):
            payload[key] = str(Path(value).resolve())
        else:
            payload[key] = value  # bool/str/Literal
    return payload


async def _run(args: CLIArgs) -> None:
    """Execute pipeline and log artifacts to MLflow (filesystem-backed in research/mlruns by default)."""
    # Log all CLI params up-front
    params = _normalise_params_for_logging(args)
    logger.info({"event": "cli_params", **params})

    # Determine PDF name for commit context and commit early on experiments branch
    pdf_name = Path(str(params["pdf"])).name
    auto_commit_research(
        branch_name="experiments",
        context_message=f"pre-run pdf={pdf_name}",
    )

    # Configure MLflow tracking and experiment
    setup_mlflow(args.tracking_uri, args.experiment_name)

    # Start MLflow run
    run_name = f"problem-detection:{pdf_name}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        commit = _git_commit()
        mlflow.log_param("git_commit", commit)
        mlflow.set_tag("source.git.commit", commit)

        logger.info("Starting pipeline…")
        result = await run_pipeline(
            pdf_path=Path(params["pdf"]),
            ocr_result_path=Path(params["ocr"]),
            working_dir=Path(params["output_dir"]),
            visualise=bool(params["visualise"]),
        )

        # Log key outputs as MLflow artifacts
        mlflow.log_artifact(str(result.output_json), artifact_path="outputs")

        if result.raw_outputs_dir and result.raw_outputs_dir.exists():
            mlflow.log_artifacts(
                str(result.raw_outputs_dir), artifact_path="outputs/llm_raw"
            )

        if result.visualisations:
            vis_dir = result.visualisations[0].parent
            mlflow.log_artifacts(str(vis_dir), artifact_path="outputs/visualisations")

        if result.problem_visualisations:
            pvis_dir = result.problem_visualisations[0].parent
            mlflow.log_artifacts(
                str(pvis_dir), artifact_path="outputs/visualisations_problems"
            )

        if result.image_paths:
            mlflow.log_artifacts(
                str(result.image_paths[0].parent), artifact_path="page_images"
            )

        # Log inputs for reproducibility
        mlflow.log_artifact(str(Path(params["pdf"])), artifact_path="inputs")
        mlflow.log_artifact(str(Path(params["ocr"])), artifact_path="inputs")

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
    """Entry-point: parse CLI via tyro and run the pipeline."""
    args = tyro.cli(CLIArgs)
    # Apply log level
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
