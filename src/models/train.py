
from pathlib import Path
import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from src.data.verify_dvc import verify_dataset
from src.pipeline.paths import RAW_DIR, PROCESSED_DIR, MODELS_DIR, project_path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# MLflow configuration
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI") or (PROJECT_ROOT / "mlruns").as_uri()
)


def train_model(
    data_dir: str | Path = DATA_DIR,
    model_dir: str | Path = MODEL_DIR,
    dataset_path: str | Path = PROJECT_ROOT / "data" / "raw" / "ar_dataset.csv",
    dataset_md5: str | None = None,
    airflow_run_id: str | None = None,
) -> str:
    data_dir = project_path(data_dir, PROCESSED_DIR)
    model_dir = project_path(model_dir, MODELS_DIR)
    dataset_path = project_path(dataset_path, RAW_DIR, file=True)
    actual_md5 = verify_dataset(dataset_path)
    if dataset_md5 is not None and dataset_md5 != actual_md5:
        raise ValueError("Dataset changed after the Airflow DVC verification task")
    mlflow.set_experiment("Arabic_Hate_Speech_Detection")

    # 1. Load datasets
    train_df = pd.read_csv(project_path(data_dir / "train.csv", PROCESSED_DIR, file=True))
    val_df = pd.read_csv(project_path(data_dir / "val.csv", PROCESSED_DIR, file=True))
    for frame in (train_df, val_df):
        missing = {"tweet_clean", "target"} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    X_train = train_df["tweet_clean"].fillna("")
    y_train = train_df["target"]

    X_val = val_df["tweet_clean"].fillna("")
    y_val = val_df["target"]

    # 2. Model parameters
    max_features = 20000
    max_iter = 1000

    # 3. Create model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 4. Start MLflow experiment
    with mlflow.start_run(run_name="TFIDF_LogReg") as run:
        mlflow.set_tags({
            "dataset.name": dataset_path.name,
            "dataset.version_source": "DVC",
            "dataset.dvc_md5": actual_md5,
            "pipeline.mode": "train",
            "airflow.run_id": airflow_run_id or "manual",
            "model.name": "tfidf_logreg",
        })

        # Record parameters
        mlflow.log_params({
            "model": "LogisticRegression",
            "max_features": max_features,
            "ngram_range": "(1, 2)",
            "max_iter": max_iter,
            "class_weight": "balanced"
        })

        # 5. Train model
        print("Training model...")
        model.fit(X_train, y_train)

        # 6. Validate model
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro F1: {f1:.4f}")

        # 7. Record metrics
        mlflow.log_metrics({
            "val_accuracy": accuracy,
            "val_f1_macro": f1
        })

        # 8. Log model to MLflow
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="tfidf_logreg"
        )
        print("Model ID:", model_info.model_id)
        print("Model URI:", model_info.model_uri)

        # 9. Save local model
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            model,
            model_dir / "tfidf_logreg.joblib"
        )

        # 10. Save MLflow run ID for evaluation
        run_id = run.info.run_id

        (model_dir / "run_id.txt").write_text(
            run_id,
            encoding="utf-8"
        )

        print("Model saved and experiment logged!")
        print(f"MLflow Run ID: {run_id}")
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "raw" / "ar_dataset.csv"))
    parser.add_argument("--dataset-md5")
    parser.add_argument("--airflow-run-id")
    args = parser.parse_args()
    train_model(args.data_dir, args.model_dir, args.dataset, args.dataset_md5, args.airflow_run_id)
