from src.api.authenticator import get_current_active_user
from src.api.authenticator import get_current_active_user
import os
from unittest.mock import MagicMock, patch
import pandas as pd
from fastapi.testclient import TestClient
from src.api.main import app
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variable for testing
os.environ["ENVIRONMENT"] = "test"

# Initialize FastAPI test client
client = TestClient(app)

# Shared test payload for predictions
customer_data = {
    'unnamed_0': 60001.0,
    'x': 60001.0,
    'customer': 1043968.0,
    'traintest': 1.0,
    'churn': 1.0,
    'churndep': 1.0,
    'revenue': 62.29,
    'mou': 861.0,
    'recchrge': 44.99,
    'directas': 0.0,
    'overage': 78.0,
    'roam': 0.0,
    'changem': -638.0,
    'changer': -27.3,
    'dropvce': 16.33,
    'blckvce': 9.66,
    'unansvce': 34.0,
    'custcare': 1.66,
    'threeway': 0.0,
    'mourec': 438.49,
    'outcalls': 76.33,
    'incalls': 26.33,
    'peakvce': 99.33,
    'opeakvce': 193.33,
    'dropblk': 26.0,
    'callfwdv': 0.0,
    'callwait': 5.0,
    'months': 19.0,
    'uniqsubs': 1.0,
    'actvsubs': 1.0,
    'phones': 1.0,
    'models': 1.0,
    'eqpdays': 556.0,
    'age1': 40.0,
    'age2': 42.0,
    'children': "1",
    'credita': "0",
    'creditaa': "0",
    'prizmrur': "0",
    'prizmub': "0",
    'prizmtwn': "1",
    'refurb': "0",
    'webcap': "1",
    'truck': "1",
    'rv': "0",
    'occprof': "0",
    'occcler': "0",
    'occcrft': "0",
    'occstud': "0",
    'occhmkr': "0",
    'occret': "0",
    'occself': "0",
    'ownrent': "0",
    'marryun': "0",
    'marryyes': "1",
    'mailord': "0",
    'mailres': "0",
    'mailflag': "0",
    'travel': "0",
    'pcown': "1",
    'creditcd': "1",
    'retcalls': "0",
    'retaccpt': "0",
    'newcelly': "0",
    'newcelln': "0",
    'refer': 0.0,
    'incmiss': "0",
    'income': 3.0,
    'mcycle': "0",
    'setprcm': "1",
    'setprc': 0.0,
    'retcall': "0",
}


# 2. ROBUST MOCK USER
mock_admin_user = MagicMock()
mock_admin_user.id = 1
mock_admin_user.username = "test-admin"
mock_admin_user.role = "admin"


def override_get_current_active_user():
    return mock_admin_user


# Assign the override after function definition
app.dependency_overrides[get_current_active_user] = override_get_current_active_user


@patch("src.api.routers.predict.load_model_by_type")
@patch("src.api.routers.predict.get_latest_model")
@patch("src.api.routers.predict.ProductionPreprocessor")
def test_predict_endpoint(mock_preprocessor, mock_get_latest_model, mock_load_model_by_type):
    # Mock preprocessor
    mock_processor_instance = mock_preprocessor.return_value
    mock_df = pd.DataFrame([range(20)])
    mock_processor_instance.preprocess.return_value = mock_df
    mock_processor_instance.get_feature_names.return_value = [
        f"col_{i}" for i in range(20)]

    # Mock latest model path
    mock_get_latest_model.return_value = "/fake/model.pkl"

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_load_model_by_type.return_value = mock_model

    # Send POST request
    response = client.post(
        "/predict/xgboost",
        json=customer_data,
        headers={"Authorization": "Bearer dummy-token-to-trigger-dependency"}
    )

    # Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["data"]["prediction"] == 1

    # 4. CRITICAL FIX: Send POST request with Authorization header
    response = client.post(
        "/predict/xgboost",
        json=customer_data,
        headers={"Authorization": "Bearer dummy-token-to-trigger-dependency"}
    )

    # Assertions
    assert response.status_code == 200  # This should now pass
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["data"]["prediction"] == 1

#  Test /train endpoint


@patch("src.api.routers.train.run_training_script")
def test_train_endpoint(mock_run_script):
    mock_run_script.return_value = None

    # 4. CRITICAL FIX: Send POST request with Authorization header
    response = client.post(
        "/train/xgboost",
        headers={"Authorization": "Bearer dummy-token-to-trigger-dependency"}
    )

    # Assertions
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["message"] == "Training initiated for xgboost"
    assert "job_id" in response_data["data"]
    assert response_data["data"]["model_type"] == "xgboost"
    assert response_data["data"]["status"] == "started"
