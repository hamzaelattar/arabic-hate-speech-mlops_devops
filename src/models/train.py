
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# MLflow configuration
mlflow.set_tracking_uri(
    (PROJECT_ROOT / "mlruns").as_uri()
)
mlflow.set_experiment("Arabic_Hate_Speech_Detection")


def train_model():

    # 1. Load datasets
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

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
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            model,
            MODEL_DIR / "tfidf_logreg.joblib"
        )

        # 10. Save MLflow run ID for evaluation
        run_id = run.info.run_id

        (MODEL_DIR / "run_id.txt").write_text(
            run_id,
            encoding="utf-8"
        )

        print("Model saved and experiment logged!")
        print(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    train_model()
