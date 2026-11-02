
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="module")
def model():
    # Find the exported MLflow model
    model_files = list(ARTIFACTS_DIR.rglob("MLmodel"))

    assert len(model_files) == 1, (
        "Expected exactly one MLflow model artifact"
    )

    model_dir = model_files[0].parent

    return mlflow.sklearn.load_model(str(model_dir))


def test_model_predictions(model):
    texts = [
        "انت شخص رائع",
        "يجب احترام جميع الناس"
    ]

    predictions = model.predict(texts)

    assert len(predictions) == len(texts)
    assert all(pred in model.classes_ for pred in predictions)


def test_model_probabilities(model):
    probabilities = model.predict_proba(
        ["انت شخص رائع"]
    )

    assert probabilities.shape == (1, 4)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_model_classes(model):
    expected = {
        "gender",
        "origin",
        "other",
        "religion"
    }

    assert set(model.classes_) == expected
