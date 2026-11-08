"""Verify a raw CSV against its adjacent DVC pointer without running DVC."""

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from src.pipeline.paths import PROJECT_ROOT, RAW_DIR, project_path


DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ar_dataset.csv"


def calculate_md5(file_path: Path) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(dataset_path: str | Path = DATASET_PATH) -> str:
    """Return the MD5 when the selected dataset matches its DVC pointer."""
    dataset_path = project_path(dataset_path, RAW_DIR, file=True)
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("Dataset must be a CSV file")
    dvc_file = dataset_path.with_name(dataset_path.name + ".dvc")
    if not dvc_file.is_file():
        raise FileNotFoundError(f"DVC metadata not found: {dvc_file}")
    with open(dvc_file, "r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    outputs = metadata.get("outs", []) if isinstance(metadata, dict) else []
    if (
        len(outputs) != 1
        or not isinstance(outputs[0], dict)
        or outputs[0].get("path") != dataset_path.name
    ):
        raise ValueError("Unexpected DVC output configuration")
    expected = outputs[0].get("md5")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", expected):
        raise ValueError("Valid MD5 checksum missing from DVC metadata")
    actual = calculate_md5(dataset_path)
    if actual != expected.lower():
        raise ValueError(
            "Dataset has changed without updating DVC. "
            f"Run: dvc add data/raw/{dataset_path.name}"
        )
    print(f"DVC dataset verification SUCCESS: {dataset_path.name} ({actual})")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    verify_dataset(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
