"""Experiments route — List and retrieve experiment JSON logs."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


@router.get("/api/experiments")
def list_experiments():
    """List all experiment JSON files from the experiments directory.

    Returns:
        {experiments: [{experiment summary}, ...]}

    Experiments are sorted by date (most recent first).
    """
    if not EXPERIMENTS_DIR.exists():
        return {"experiments": []}

    experiments = []
    for filepath in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(filepath.read_text())
            experiments.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return {"experiments": experiments}


@router.get("/api/experiments/{run_id}")
def get_experiment(run_id: str):
    """Get a specific experiment by its run_id.

    Looks for files matching the pattern:
      - pipeline_{run_id}.json
      - {run_id}.json
      - any .json file containing the experiment_id

    Returns the experiment JSON data.
    """
    if not EXPERIMENTS_DIR.exists():
        raise HTTPException(status_code=404, detail="No experiments directory found")

    # Sanitize run_id to prevent path traversal
    safe_run_id = run_id.replace("/", "").replace("\\", "").replace("..", "")

    # Try direct filename matches first
    candidates = [
        EXPERIMENTS_DIR / f"pipeline_{safe_run_id}.json",
        EXPERIMENTS_DIR / f"{safe_run_id}.json",
    ]

    for filepath in candidates:
        if filepath.exists() and filepath.is_file():
            try:
                data = json.loads(filepath.read_text())
                return data
            except (json.JSONDecodeError, OSError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to read experiment file: {exc}",
                ) from exc

    # Fall back to scanning all files for matching experiment_id
    for filepath in EXPERIMENTS_DIR.glob("*.json"):
        try:
            data = json.loads(filepath.read_text())
            exp_id = data.get("experiment_id", "")
            if safe_run_id in exp_id or exp_id == safe_run_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue

    raise HTTPException(
        status_code=404,
        detail=f"Experiment '{run_id}' not found",
    )
