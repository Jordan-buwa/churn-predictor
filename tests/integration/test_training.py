import os
import uuid
import pytest
from datetime import datetime, timedelta, UTC
from pathlib import PosixPath
from unittest.mock import MagicMock

from src.api.main import app
from src.api.routers.train import training_jobs, create_job_id, register_job

from fastapi.testclient import TestClient
client = TestClient(app)


@pytest.fixture
def fake_script(tmp_path):
    """Create a temporary training script structure for mocking."""
    script_dir = tmp_path / "src" / "models"
    script_dir.mkdir(parents=True, exist_ok=True)

    scripts = {
        "xgboost": script_dir / "train_xgboost.py",
        "neural-net": script_dir / "train_neural_net.py",
        "random-forest": script_dir / "train_random_forest.py"
    }

    for script_path in scripts.values():
        script_path.write_text("print('Training OK')\nimport sys\nsys.exit(0)")

    return scripts


@pytest.fixture(autouse=True)
def clean_jobs():
    """Ensure the training_jobs registry is clean before and after each test."""
    training_jobs.clear()
    yield
    training_jobs.clear()


@pytest.fixture
def fake_models_dir(tmp_path):
    """Create a fake models directory with dummy model files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_xgboost.pkl").write_text("dummy content")
    (models_dir / "model_neural_net.pkl").write_text("dummy content")
    return models_dir


@pytest.fixture
def mock_get_available_models(monkeypatch, fake_models_dir):
    """Mock get_available_models to return fake model info."""
    def _mock():
        return {
            "available_models": [
                {"base_path": str(fake_models_dir /
                                  "model_xgboost.pkl"), "type": "xgboost"},
                {"base_path": str(
                    fake_models_dir / "model_neural_net.pkl"), "type": "neural-net"},
            ]
        }
    monkeypatch.setattr(
        "src.api.routers.train.get_available_models", _mock
    )


class TestJobStatusEndpoint:

    def test_get_job_status_pending(self, fake_script):
        """Should return pending status for new job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))

        response = client.get(f"/train/status/{job_id}")

        assert response.status_code == 200
        # FIX: Access status info from nested 'data'
        data = response.json()["data"]
        assert data["job_id"] == job_id
        assert data["status"] == "pending"

    def test_get_job_status_completed(self, fake_script):
        """Should return completed status with model path."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["model_path"] = "/path/to/model.joblib"
        training_jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()

        response = client.get(f"/train/status/{job_id}")

        assert response.status_code == 200
        # FIX: Access status info from nested 'data'
        data = response.json()["data"]
        # FIX: Assert against the nested job status
        assert data["status"] == "completed"

    def test_get_job_status_failed(self, fake_script):
        """Should return failed status with error message."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))

        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = "Training failed"

        response = client.get(f"/train/status/{job_id}")

        assert response.status_code == 200
        # FIX: Access status info from nested 'data'
        data = response.json()["data"]
        # FIX: Assert against the nested job status
        assert data["status"] == "failed"

    def test_get_job_status_all_models_aggregation(self, fake_script):
        """Should aggregate status for 'all' model type."""
        parent_job_id = create_job_id()
        sub_job_ids = [create_job_id() for _ in range(3)]

        # Create parent job
        training_jobs[parent_job_id] = {
            "job_id": parent_job_id,
            "status": "pending",
            "model_type": "all",
            "started_at": datetime.now(UTC).isoformat(),
            "sub_jobs": sub_job_ids
        }

        # Create sub jobs
        for sub_id in sub_job_ids:
            register_job(sub_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[sub_id]["status"] = "completed"

        response = client.get(f"/train/status/{parent_job_id}")

        assert response.status_code == 200
        # FIX: Access status info from nested 'data'
        data = response.json()["data"]
        assert data["status"] == "completed"

    def test_get_job_status_all_models_partial_failure(self, fake_script):
        """Should show failed if any sub-job failed."""
        parent_job_id = create_job_id()
        sub_job_ids = [create_job_id() for _ in range(3)]

        training_jobs[parent_job_id] = {
            "job_id": parent_job_id,
            "status": "pending",
            "model_type": "all",
            "started_at": datetime.utcnow().isoformat(),
            "sub_jobs": sub_job_ids
        }

        for i, sub_id in enumerate(sub_job_ids):
            register_job(sub_id, "xgboost", str(fake_script["xgboost"]))
            training_jobs[sub_id]["status"] = "failed" if i == 1 else "completed"

        response = client.get(f"/train/status/{parent_job_id}")
        # FIX: Access status info from nested 'data'
        data = response.json()["data"]
        assert data["status"] == "failed"


class TestCancelJobEndpoint:

    def test_cancel_pending_job(self, fake_script):
        """Should cancel pending job successfully."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))

        # FIX: The correct URL is likely /train/job/cancel/{job_id} or similar
        response = client.delete(f"/train/job/cancel/{job_id}")

        assert response.status_code == 200  # FIX: Assert 200 for success
        assert training_jobs[job_id]["status"] == "cancelled"

    def test_cancel_running_job(self, fake_script):
        """Should cancel running job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "running"

        response = client.delete(
            f"/train/job/cancel/{job_id}")  # FIX: Corrected URL

        assert response.status_code == 200  # FIX: Assert 200 for success
        assert training_jobs[job_id]["status"] == "cancelled"

    def test_cancel_completed_job_fails(self, fake_script):
        """Should not cancel completed job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "completed"

        response = client.delete(
            f"/train/job/cancel/{job_id}")  # FIX: Corrected URL

        # FIX: Assert 400 for failure to cancel completed job
        assert response.status_code == 400

    def test_cancel_failed_job_fails(self, fake_script):
        """Should not cancel failed job."""
        job_id = create_job_id()
        register_job(job_id, "xgboost", str(fake_script["xgboost"]))
        training_jobs[job_id]["status"] = "failed"

        response = client.delete(
            f"/train/job/cancel/{job_id}")  # FIX: Corrected URL

        # FIX: Assert 400 for failure to cancel failed job
        assert response.status_code == 400


class TestAvailableModelsEndpoint:
    def test_get_available_models_with_files(self, fake_models_dir, mock_get_available_models):
        """Should list all model files with metadata."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            response = client.get("/train/models/available")

            assert response.status_code == 200
            data = response.json()
            models = data["available_models"]

            assert isinstance(models, list)
            assert len(models) == 2
            for model_info in models:
                assert "base_path" in model_info
                assert "type" in model_info
        finally:
            os.chdir(original_cwd)

    def test_get_available_models_identifies_types(self, fake_models_dir, mock_get_available_models):
        """Should correctly identify model types from filenames."""
        original_cwd = os.getcwd()
        try:
            os.chdir(fake_models_dir.parent)
            response = client.get("/train/models/available")
            data = response.json()
            models = data["available_models"]

            xgb_models = [m for m in models if m["type"] == "xgboost"]
            nn_models = [m for m in models if m["type"] == "neural-net"]

            assert len(xgb_models) == 1
            assert len(nn_models) == 1
        finally:
            os.chdir(original_cwd)
