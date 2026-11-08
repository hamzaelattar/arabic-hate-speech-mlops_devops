"""Resolve pipeline paths without allowing inputs to escape project directories."""

from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def project_path(value: str | Path, allowed_dir: Path, *, file: bool = False) -> Path:
    """Accept native absolute paths or project-relative paths inside allowed_dir."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(allowed_dir.resolve()):
        raise ValueError(f"Path must stay inside {allowed_dir}: {value}")
    if file and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def run_key(airflow_run_id: str) -> str:
    if not isinstance(airflow_run_id, str) or not airflow_run_id:
        raise ValueError("Airflow run ID is required")
    return sha256(airflow_run_id.encode("utf-8")).hexdigest()
