
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "tfidf_logreg.joblib"
RUN_ID_PATH = MODEL_DIR / "run_id.txt"


# MLflow filesystem configuration
mlflow.set_tracking_uri(
    (PROJECT_ROOT / "mlruns").as_uri()
)


def evaluate_model():

    # 1. Load test dataset
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    X_test = test_df["tweet_clean"].fillna("")
    y_test = test_df["target"]

    # 2. Load trained model
    model = joblib.load(MODEL_PATH)

    # 3. Predict
    y_pred = model.predict(X_test)

    # 4. Calculate metrics
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
    }

    print("\n===== TEST EVALUATION =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 5. Classification report
    report = classification_report(
        y_test, y_pred, zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # 6. Confusion matrix
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print(cm)

    # 7. Save reports locally
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "metrics.json"
    report_path = REPORT_DIR / "classification_report.txt"
    cm_path = REPORT_DIR / "confusion_matrix.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report_path.write_text(report, encoding="utf-8")

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
    cm_df.to_csv(cm_path)

    # 8. Connect evaluation to the training run
    if not RUN_ID_PATH.exists():
        raise FileNotFoundError(
            "MLflow run ID not found. Run train.py first."
        )

    run_id = RUN_ID_PATH.read_text(
        encoding="utf-8"
    ).strip()

    with mlflow.start_run(run_id=run_id):

        # Log test metrics
        mlflow.log_metrics(metrics)

        # Log evaluation reports
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

    print("\nEvaluation completed!")
    print("Test metrics and reports logged to MLflow!")


if __name__ == "__main__":
    evaluate_model()
