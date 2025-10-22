from __future__ import annotations

import asyncio
import sys
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Literal
from pathlib import Path
from datetime import datetime

import mlflow
import tyro
from loguru import logger

from detector import run_pipeline

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
    """Return current repository commit hash or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .strip()
            .decode("utf-8")
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _normalise_params_for_logging(args: CLIArgs) -> dict[str, str | int | float | bool]:
    """Convert dataclass args to primitives suitable for logging/MLflow."""
    payload: dict[str, str | int | float | bool] = {}
    for key, value in asdict(args).items():
        if isinstance(value, Path):
            payload[key] = str(Path(value).resolve())
        else:
            payload[key] = value  # bool/str/Literal
    return payload


def _is_uri(value: str) -> bool:
    return "://" in value


def _auto_commit_research(branch_name: str = "experiments", context_message: str = "") -> None:
    """Stage backend/research and commit on the given branch if there are changes.

    - Stages only the research folder.
    - Switches to the target branch (creates if missing), commits only that path.
    - Switches back to the original branch.
    - No-op if there is nothing to commit under research.
    """

    def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    # Discover repo root
    repo_root_proc = _run_git(["rev-parse", "--show-toplevel"], current_dir)
    if repo_root_proc.returncode != 0:
        logger.debug("Not in a git repo; skipping auto-commit.")
        return
    repo_root = Path(repo_root_proc.stdout.strip())

    research_path_rel = Path("backend") / "research"
    research_abs = repo_root / research_path_rel
    if not research_abs.exists():
        logger.debug(f"Research path not found at {research_abs}; skipping auto-commit.")
        return

    # Stage research folder only
    _run_git(["add", "-A", str(research_path_rel)], repo_root)

    # Check if there is anything staged for research
    diff_cached = _run_git(["diff", "--cached", "--name-only", "--", str(research_path_rel)], repo_root)
    staged_files = [ln for ln in diff_cached.stdout.splitlines() if ln.strip()]
    if not staged_files:
        logger.info("No changes in research to commit.")
        return

    # Remember current branch to switch back later
    cur_branch_proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    current_branch = cur_branch_proc.stdout.strip() if cur_branch_proc.returncode == 0 else "HEAD"

    # Ensure target branch exists and switch to it
    branch_exists = _run_git(["rev-parse", "--verify", branch_name], repo_root).returncode == 0
    if branch_exists:
        switch_proc = _run_git(["switch", branch_name], repo_root)
        if switch_proc.returncode != 0:
            # Fallback to checkout for older git
            switch_proc = _run_git(["checkout", branch_name], repo_root)
    else:
        switch_proc = _run_git(["switch", "-c", branch_name], repo_root)
        if switch_proc.returncode != 0:
            switch_proc = _run_git(["checkout", "-b", branch_name], repo_root)

    if switch_proc.returncode != 0:
        logger.warning(
            f"Could not switch to '{branch_name}' branch; committing on current branch '{current_branch}'.\n"
            f"git switch/checkout stderr: {switch_proc.stderr.strip()}"
        )

    # Compose commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"research: auto-commit {timestamp} {context_message}".strip()

    # Commit ONLY the research path, leaving other staged changes untouched
    commit_proc = _run_git(["commit", "-m", msg, "--", str(research_path_rel)], repo_root)
    if commit_proc.returncode != 0:
        # If nothing to commit (race) or other issue, log and exit
        stderr = commit_proc.stderr.strip()
        if "nothing to commit" in stderr.lower():
            logger.info("No changes to commit after staging.")
        else:
            logger.warning(f"Git commit failed: {stderr}")

    # Switch back to previous branch if we managed to switch earlier and there is a named branch to return to
    if switch_proc.returncode == 0 and current_branch not in {"HEAD", branch_name, "(HEAD)"}:
        back_proc = _run_git(["switch", current_branch], repo_root)
        if back_proc.returncode != 0:
            back_proc = _run_git(["checkout", current_branch], repo_root)
        if back_proc.returncode != 0:
            logger.warning(f"Failed to switch back to '{current_branch}': {back_proc.stderr.strip()}")

async def _run(args: CLIArgs) -> None:
    """Execute pipeline and log artifacts to MLflow (filesystem-backed in research/mlruns by default)."""
    # Log all CLI params up-front
    params = _normalise_params_for_logging(args)
    logger.info({"event": "cli_params", **params})

    # Determine PDF name for commit context and commit early on experiments branch
    pdf_name = Path(str(params["pdf"])).name
    try:
        _auto_commit_research(
            branch_name="experiments",
            context_message=f"pre-run pdf={pdf_name}",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Pre-run auto-commit skipped: {e}")

    # Configure tracking URI (local path by default under research/mlruns)
    if args.tracking_uri:
        try:
            tracking = args.tracking_uri
            if not _is_uri(tracking):
                # Treat as filesystem path
                tracking_path = Path(tracking).resolve()
                tracking_path.mkdir(parents=True, exist_ok=True)
                tracking = str(tracking_path)
            mlflow.set_tracking_uri(tracking)
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Failed to set MLflow tracking URI '{args.tracking_uri}': {e}"
            )

    # Ensure experiment exists / is selected
    try:
        mlflow.set_experiment(args.experiment_name)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to set MLflow experiment '{args.experiment_name}': {e}")

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
        try:
            mlflow.log_artifact(str(result.output_json), artifact_path="outputs")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to log output_json: {e}")

        if result.raw_outputs_dir and result.raw_outputs_dir.exists():
            try:
                mlflow.log_artifacts(
                    str(result.raw_outputs_dir), artifact_path="outputs/llm_raw"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to log llm_raw: {e}")

        if result.visualisations:
            vis_dir = result.visualisations[0].parent
            try:
                mlflow.log_artifacts(
                    str(vis_dir), artifact_path="outputs/visualisations"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to log visualisations: {e}")

        if result.problem_visualisations:
            pvis_dir = result.problem_visualisations[0].parent
            try:
                mlflow.log_artifacts(
                    str(pvis_dir), artifact_path="outputs/visualisations_problems"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to log problem visualisations: {e}")

        if result.image_paths:
            try:
                mlflow.log_artifacts(
                    str(result.image_paths[0].parent), artifact_path="page_images"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"Failed to log page images: {e}")

        # Log inputs for reproducibility
        try:
            mlflow.log_artifact(str(Path(params["pdf"])), artifact_path="inputs")
        except Exception:
            pass
        try:
            mlflow.log_artifact(str(Path(params["ocr"])), artifact_path="inputs")
        except Exception:
            pass

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
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(sys.stderr, level=args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
