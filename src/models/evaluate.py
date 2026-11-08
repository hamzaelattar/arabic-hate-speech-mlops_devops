"""Evaluate a trained model or generate predictions for an external dataset."""

import argparse
import json
import os
import re
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.verify_dvc import verify_dataset
from src.pipeline.paths import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    project_path,
)


DATA_DIR = PROCESSED_DIR
MODEL_DIR = MODELS_DIR
REPORT_DIR = REPORTS_DIR
MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"
RUN_ID_PATH = MODEL_DIR / "run_id.txt"

mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI") or (PROJECT_ROOT / "mlruns").as_uri()
)


def evaluate_model(
    data_path: str | Path = DATA_DIR / "test.csv",
    model_path: str | Path = MODEL_PATH,
    report_dir: str | Path = REPORT_DIR,
    run_id_path: str | Path = RUN_ID_PATH,
    mode: str = "train",
    dataset_path: str | Path = RAW_DIR / "ar_dataset.csv",
    dataset_md5: str | None = None,
    airflow_run_id: str | None = None,
    model_id: str | None = None,
    run_id: str | None = None,
) -> str:
    if mode not in {"train", "evaluate"}:
        raise ValueError("mode must be train or evaluate")
    data_path = project_path(data_path, PROCESSED_DIR, file=True)
    try:
        report_dir = project_path(report_dir, ARTIFACTS_DIR)
    except ValueError:
        report_dir = project_path(report_dir, REPORTS_DIR)
    dataset_path = project_path(dataset_path, RAW_DIR, file=True)
    actual_md5 = verify_dataset(dataset_path)
    if dataset_md5 is not None and actual_md5 != dataset_md5:
        raise ValueError("Dataset changed after the Airflow DVC verification task")

    if model_id:
        if mode != "evaluate" or not re.fullmatch(r"m-[0-9a-f]{32}", model_id):
            raise ValueError("model_id must be a valid MLflow logged model ID in evaluate mode")
        model = mlflow.sklearn.load_model(f"models:/{model_id}")
        model_name = model_id
    else:
        model_path = project_path(model_path, MODELS_DIR, file=True)
        model = joblib.load(model_path)
        model_name = model_path.name

    data = pd.read_csv(data_path)
    if "tweet_clean" not in data.columns:
        raise ValueError("Missing required column: tweet_clean")
    if data.empty:
        raise ValueError("Dataset contains no nonempty tweets")
    prediction = model.predict(data["tweet_clean"].fillna(""))
    data["prediction"] = prediction

    labels = pd.Series(pd.NA, index=data.index, dtype="string")
    if "target" in data.columns:
        labels = data["target"].astype("string").str.strip().replace("", pd.NA)
    labeled = labels.notna()
    if mode == "train" and not labeled.all():
        raise ValueError("Training evaluation requires a target for every row")
    unknown = set(labels[labeled]) - set(model.classes_)
    if unknown:
        raise ValueError(f"Unknown target labels: {sorted(unknown)}")

    report_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(report_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    stats = {
        "rows": int(len(data)),
        "labeled_rows": int(labeled.sum()),
        "prediction_counts": {str(k): int(v) for k, v in data["prediction"].value_counts().items()},
    }
    (report_dir / "prediction_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metrics = {}
    if labeled.any():
        truth = labels[labeled]
        predicted = data.loc[labeled, "prediction"]
        metrics = {
            "test_accuracy": accuracy_score(truth, predicted),
            "test_precision_macro": precision_score(truth, predicted, average="macro", zero_division=0),
            "test_recall_macro": recall_score(truth, predicted, average="macro", zero_division=0),
            "test_f1_macro": f1_score(truth, predicted, average="macro", zero_division=0),
        }
        (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")
        (report_dir / "classification_report.txt").write_text(
            classification_report(truth, predicted, zero_division=0), encoding="utf-8"
        )
        matrix = confusion_matrix(truth, predicted, labels=model.classes_)
        pd.DataFrame(matrix, index=model.classes_, columns=model.classes_).to_csv(
            report_dir / "confusion_matrix.csv"
        )

    if mode == "train":
        if run_id is None:
            run_id = project_path(run_id_path, MODELS_DIR, file=True).read_text(encoding="utf-8").strip()
        run_context = mlflow.start_run(run_id=run_id)
    else:
        mlflow.set_experiment("Arabic_Hate_Speech_Detection")
        run_context = mlflow.start_run(run_name="external_evaluation")
    with run_context as run:
        mlflow.set_tags({
            "dataset.name": dataset_path.name,
            "dataset.version_source": "DVC",
            "dataset.dvc_md5": actual_md5,
            "pipeline.mode": mode,
            "airflow.run_id": airflow_run_id or "manual",
            "model.name": model_name,
        })
        if model_id:
            mlflow.log_param("model_id", model_id)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(report_dir), artifact_path="evaluation")
        print(f"Evaluation completed: {len(data)} rows; {int(labeled.sum())} labeled")
        print(f"MLflow Run ID: {run.info.run_id}")
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--run-id-path", default=str(RUN_ID_PATH))
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--dataset", default=str(RAW_DIR / "ar_dataset.csv"))
    parser.add_argument("--dataset-md5")
    parser.add_argument("--airflow-run-id")
    parser.add_argument("--model-id")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    evaluate_model(
        args.data_path, args.model_path, args.report_dir, args.run_id_path,
        args.mode, args.dataset, args.dataset_md5, args.airflow_run_id,
        args.model_id, args.run_id,
    )


if __name__ == "__main__":
    main()
