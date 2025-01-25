import pytest
import sys
import os

# Add the root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_test_route(client):
    response = client.get("/test")
    assert response.status_code == 200
    assert response.data == b"hello"

def test_predict_route(client):
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "prediction" in response.json