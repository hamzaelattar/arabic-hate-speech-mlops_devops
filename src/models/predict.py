
from pathlib import Path
import argparse

import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "tfidf_logreg.joblib"


def predict_tweet(text):

    # 1. Load trained model
    model = joblib.load(MODEL_PATH)

    # 2. Predict class
    prediction = model.predict([text])[0]

    # 3. Predict probabilities
    probabilities = model.predict_proba([text])[0]

    # 4. Display results
    print("\n===== PREDICTION =====")
    print(f"Tweet: {text}")
    print(f"Predicted class: {prediction}")

    print("\nClass probabilities:")

    for label, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{label}: {probability:.4f}")

    return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arabic Hate Speech Detection"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Cleaned Arabic tweet to classify"
    )

    args = parser.parse_args()

    predict_tweet(args.text)
