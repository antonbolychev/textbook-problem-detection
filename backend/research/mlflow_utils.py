from __future__ import annotations

from pathlib import Path
from datetime import datetime

import mlflow
from git import Repo


def _is_uri(value: str) -> bool:
    return "://" in value


def setup_mlflow(tracking_uri: str | None, experiment_name: str) -> None:
    """Set MLflow tracking URI and ensure experiment exists.

    - If `tracking_uri` is a filesystem path, ensure it exists and use it.
    - Otherwise, treat it as a URI and pass through to MLflow.
    - Sets the active experiment to `experiment_name`.
    """
    if tracking_uri is not None:
        tracking = tracking_uri
        if not _is_uri(tracking):
            tracking_path = Path(tracking).resolve()
            tracking_path.mkdir(parents=True, exist_ok=True)
            tracking = str(tracking_path)
        mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(experiment_name)


def auto_commit_research(branch_name: str = "experiments", context_message: str = "") -> None:
    """Auto-commit research changes with minimal branch churn.

    Behaviour:
    - If there are no changes under ``backend/research``: do nothing.
    - If the current branch is ``branch_name`` (default: ``experiments``):
      commit the changes directly on this branch (no new branch).
    - Otherwise, create and switch to a new branch derived from the current
      HEAD named ``<branch_name>-<shortsha>``; if it already exists, append a
      timestamp to keep it unique.
    - Only stage and commit the ``backend/research`` path.
    """

    repo = Repo(search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir or ".").resolve()
    research_rel = Path("backend") / "research"
    research_abs = repo_root / research_rel
    if not research_abs.exists():
        return

    # Quick check: anything to commit under research?
    # Limit porcelain output to the research path only.
    status_lines = repo.git.status("--porcelain", "--", str(research_rel)).splitlines()
    if not status_lines:
        return

    # If already on the target branch, commit in place. Otherwise, create a
    # short-lived experiments branch named with the current short SHA.
    try:
        active_branch = repo.active_branch.name  # type: ignore[attr-defined]
    except Exception:
        active_branch = ""

    if active_branch != branch_name:
        shortsha = repo.git.rev_parse("--short", "HEAD").strip()
        base_branch = f"{branch_name}-{shortsha}"
        new_branch = base_branch
        existing = {h.name for h in repo.heads}
        if new_branch in existing:
            ts_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            new_branch = f"{base_branch}-{ts_id}"
        repo.git.checkout("-b", new_branch)

    # Stage and commit only research path
    repo.git.add("-A", str(research_rel))
    staged = repo.git.diff("--cached", "--name-only", "--", str(research_rel)).splitlines()
    if staged:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"research: auto-commit {timestamp} {context_message}".strip()
        repo.git.commit("-m", msg, "--", str(research_rel))
