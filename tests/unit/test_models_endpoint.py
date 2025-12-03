import pytest
import os
import tempfile
import json
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.routers.models import router

@pytest.fixture
def client():
    """Test client fixture."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)

@pytest.fixture
def temp_model_dir():
    """Create a temporary model directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_model_dir = os.environ.get('MODEL_DIR')
        os.environ['MODEL_DIR'] = temp_dir
        
        # Creating a dummy model structure
        model_path = os.path.join(temp_dir, "random_forest")
        os.makedirs(model_path, exist_ok=True)
        
        metadata = {
            "version": "1.0.0",
            "description": "Test random forest model"
        }
        with open(os.path.join(model_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        yield temp_dir
        
        # Cleanup
        if original_model_dir is not None:
            os.environ['MODEL_DIR'] = original_model_dir
        else:
            del os.environ['MODEL_DIR']

def test_get_models_local(client, temp_model_dir):
    """Test the /models endpoint using local models only."""
    response = client.get("/models")  
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    
    data = response.json()
    assert "models" in data
    assert "source" in data
    assert data["source"] in ["mlflow", "local"]
    assert isinstance(data["models"], list)