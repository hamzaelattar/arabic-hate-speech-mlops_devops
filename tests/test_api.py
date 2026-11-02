
from fastapi.testclient import TestClient
from src.api.main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_predict():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "انت شخص رائع"}
        )

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "probabilities" in response.json()
