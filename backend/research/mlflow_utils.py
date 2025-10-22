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
    """Commit backend/research changes on a fresh `branch_name-<id>` branch before running.

    - Detects pending changes under backend/research (tracked and untracked).
    - Creates and checks out a new branch named `<branch_name>-<YYYYMMDD-HHMMSS>` from current HEAD.
    - Stages and commits only backend/research on that new branch.
    - Stays on the new branch (no stash, no switching back).
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

    # Create and switch to a unique experiments branch at current HEAD using short SHA
    shortsha = repo.git.rev_parse("--short", "HEAD").strip()
    base_branch = f"{branch_name}-{shortsha}"
    new_branch = base_branch
    # Ensure uniqueness if the base name already exists
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
