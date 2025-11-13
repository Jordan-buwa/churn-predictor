import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.routers import train
from fastapi import FastAPI

# FastAPI test client
app = FastAPI()
app.include_router(train.router)
client = TestClient(app)

# Clear jobs before each test


@pytest.fixture(autouse=True)
def clear_jobs():
    train.training_jobs.clear()

# Mock all file system and subprocess calls


@pytest.fixture(autouse=True)
def mock_external_calls():
    with patch("src.api.routers.train.validate_training_script") as mock_validate, \
            patch("src.api.routers.train.get_script_path") as mock_path, \
            patch("src.api.routers.train.run_training_script") as mock_run, \
            patch("src.api.routers.train.get_allowed_model_types") as mock_allowed:
        # Pretend scripts always exist
        mock_validate.side_effect = lambda x: f"/fake/path/{x}"
        # Map model_type to fake scripts
        mock_path.side_effect = lambda mt: f"{mt}_script.py"
        # Pretend run_training_script does nothing
        mock_run.return_value = None
        # Return allowed model types
        mock_allowed.return_value = ["neural-net", "xgboost", "random-forest"]
        yield
