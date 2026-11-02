
from pathlib import Path
import argparse

import mlflow
import mlflow.sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Same MLflow storage as train.py
mlflow.set_tracking_uri(
    (PROJECT_ROOT / "mlruns").as_uri()
)


def load_trained_model(model_uri):

    # Load a specific model from MLflow
    model = mlflow.sklearn.load_model(model_uri)

    print("Model loaded successfully!")
    print("Model URI:", model_uri)

    # Test prediction
    text = "انت شخص رائع"

    prediction = model.predict([text])[0]

    print("Tweet:", text)
    print("Prediction:", prediction)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI"
    )

    args = parser.parse_args()

    load_trained_model(args.model_uri)
