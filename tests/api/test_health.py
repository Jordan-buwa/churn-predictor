from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert "environment" in data

def test_models_status_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body