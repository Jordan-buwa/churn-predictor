from fastapi import HTTPException, APIRouter, Body, status, Depends
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from src.api.authenticator import get_current_active_user
from src.api.db import UserCreate, UserRead, UserRole
from src.api.utils.database import get_db_connection
from src.data_pipeline.preprocess import ProductionPreprocessor
from typing import Dict, Any
import os
import joblib
import json
import logging
from src.api.utils.config import APIConfig, get_model_path, get_allowed_model_types
from src.api.utils.response_models import PredictionResponse
from src.api.utils.error_handlers import (
    DataNotFoundError, PreprocessingError,
    handle_model_error, handle_data_error, raise_if_model_not_found
)
from src.api.utils.models_types import ModelType, validate_model_type
from src.api.ml_models import load_single_model

if os.getenv("ENVIRONMENT") == "test":
    from unittest.mock import MagicMock
    current_active_user = MagicMock(id="test-user")
else:
    from src.api.authenticator import current_active_user

from ..main import PREDICTION_CLASS_COUNT, AVG_PREDICTION_PROBABILITY, MODEL_PREDICTION_COUNT, CURRENT_BATCH_AVG_PROB

router = APIRouter(prefix="/predict")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialize configuration
config = APIConfig()


def get_latest_model(model_type: str):
    """Get the latest model path using centralized configuration."""
    try:
        if not validate_model_type(model_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type. Must be one of: {ModelType.get_all_types()}"
            )

        model_path = get_model_path(model_type)
        raise_if_model_not_found(model_path, model_type)
        return model_path
    except Exception as e:
        handle_model_error(model_type, e)


def load_model_by_type(model_type: str):
    """Centralized loader using ml_models with metadata-aware path selection."""
    model = load_single_model(model_type)
    if model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to load model: {model_type}")
    return model


def map_dropdowns(payload: dict) -> dict:
    # --- PRIZM ---
    prizm = payload.pop("prizm_cluster", None)
    payload["prizmrur"] = 1 if prizm == "rural" else 0
    payload["prizmub"] = 1 if prizm == "urban" else 0
    payload["prizmtwn"] = 1 if prizm == "town" else 0

    # --- OCCUPATION ---
    occ = payload.pop("occupation", None)
    occ_map = {
        "professional": "occprof",
        "clerical": "occcler",
        "craft": "occcrft",
        "student": "occstud",
        "homemaker": "occhmkr",
        "retired": "occret",
        "self-employed": "occself"
    }
    # Initialize all to 0
    for col in occ_map.values():
        payload[col] = 0
    if occ and occ in occ_map:
        payload[occ_map[occ]] = 1

    # --- MARITAL ---
    marital = payload.pop("marital_status", "unknown")
    payload["marryyes"] = 1 if marital == "married" else 0
    payload["marryun"] = 1 if marital == "unmarried" else 0

    return payload


def null_to_nan(payload: dict) -> dict:
    """
    Recursively convert JSON `null` (Python `None`) to `np.nan`.
    Leaves numbers/strings untouched.
    """
    for key, value in payload.items():
        if value is None:
            payload[key] = np.nan
        elif isinstance(value, dict):
            null_to_nan(value)
    return payload


@router.post("/{model_type}", response_model=PredictionResponse)
def predict_from_payload(
    model_type: str,
    payload: Dict[str, Any] = Body(..., example={
        "revenue": 45.3, "mou": 120.5, "months": 12, "credita": "A"
    }),
    current_user=Depends(get_current_active_user)
):
    """
    Accept raw customer data -> predict churn and update Prometheus metrics.
    """

    # Adding role-based access control
    allowed_roles = [UserRole.ADMIN, UserRole.MANAGER, UserRole.SUPERVISOR]
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your role does not have access to prediction services"
        )

    # Log prediction request for auditing
    logger.info(
        f"Prediction request - User: {current_user.username}, Model: {model_type}")

    payload = map_dropdowns(payload)
    payload = null_to_nan(payload)
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {ModelType.get_all_types()}"
        )

    try:
        raw_data = payload

        # Check if churn value is provided
        provided_churn = raw_data.get('churn')
        customer_id = raw_data.get('customer_id', 'ad-hoc')

        # If churn is provided, return it as the prediction
        if provided_churn is not None:
            # Convert to float for consistency
            churn_value = float(provided_churn)
            prediction_data = {
                "model_type": model_type,
                "prediction": churn_value,
                "model_path": "provided_in_input",
                "customer_id": customer_id,
                "preprocessing_applied": False,
                "confidence": 1.0,
                "note": "Using provided churn value from input"
            }

            return PredictionResponse(
                status="success",
                message="Using provided churn value as prediction",
                data=prediction_data
            )

        # --- Standard Prediction Flow ---

        df = pd.DataFrame([raw_data])

        if os.getenv("ENVIRONMENT") != "test":
            artifact_path = config.preprocessing_artifacts_path
            if not os.path.exists(artifact_path):
                raise PreprocessingError(
                    f"Preprocessing artifacts not found at {artifact_path}")
        else:
            artifact_path = "/tmp/dummy_artifacts.json"

        processor = ProductionPreprocessor(artifacts_path=artifact_path)
        df_processed = processor.preprocess(df)
        feature_names = processor.get_feature_names()
        features_dict = df_processed[feature_names].iloc[0].to_dict()
        X = np.array([list(features_dict.values())], dtype=np.float32)

        # Load model
        model_path = get_latest_model(model_type)
        model = load_model_by_type(model_type)

        is_neural_net = model_type.replace("-", "_") == ModelType.NEURAL_NET

        # Prediction and Probability Calculation
        if is_neural_net:
            import torch
            tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                # Neural nets usually output probabilities [0.0 - 1.0]
                output = model(tensor).cpu().numpy()
                probability_score = output.flatten()[0].item()
                pred_class = int(np.round(probability_score))
        else:
            # Standard Scikit-learn models
            pred_class = model.predict(X)[0]

            if hasattr(model, 'predict_proba'):
                # Use [:, 1] to get the probability of the positive class (1)
                probability_score = model.predict_proba(X)[0][1]
            else:
                # Fallback for models without predict_proba
                probability_score = float(pred_class)

        # --- Prometheus Metrics Update ---
        n_predictions = len(X)

        # M1: PREDICTION CLASS COUNT (Counter, Labelled by class)
        PREDICTION_CLASS_COUNT.labels(
            model_name=model_type,
            prediction_class=str(pred_class)
        ).inc()

        # M2: AVERAGE PREDICTION PROBABILITY (Gauge)
        AVG_PREDICTION_PROBABILITY.labels(
            model_name=model_type).set(probability_score)

        # M3: MODEL PREDICTION COUNT (Counter, labelled by model)
        MODEL_PREDICTION_COUNT.labels(model_name=model_type).inc(n_predictions)

        # M4: CURRENT BATCH AVG PROB (Gauge - tracks the latest score)
        CURRENT_BATCH_AVG_PROB.labels(
            model_name=model_type).set(probability_score)

        logging.info(
            f"Metrics updated for {model_type}. Prob: {probability_score:.4f}, "
            f"Class: {pred_class}"
        )
        # --- End Prometheus Metrics Update ---

        prediction_data = {
            "model_type": model_type,
            "prediction": float(pred_class),
            "model_path": model_path,
            "customer_id": raw_data.get("customer_id", "ad-hoc"),
            "preprocessing_applied": True,
            "confidence": float(probability_score),
        }

        return PredictionResponse(
            message="Prediction successful",
            data=prediction_data
        )

    except Exception as e:
        logger.error(f"Prediction failed for model {model_type}: {str(e)}")
        if "preprocessing" in str(e).lower():
            raise PreprocessingError(f"Failed to preprocess data: {str(e)}")
        else:
            handle_model_error(model_type, e)


@router.get("/{model_type}/customer/{customer_id}", response_model=PredictionResponse)
def predict_from_db_customer(
    model_type: str,
    customer_id: str,
    current_user=Depends(get_current_active_user)
):
    """Predict churn for a specific customer from database."""

    allowed_roles = [UserRole.ADMIN, UserRole.MANAGER, UserRole.SUPERVISOR]
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your role does not give you access to database"
        )
    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {get_allowed_model_types()}"
        )

    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT features FROM customer_data WHERE customer_id = %s",
                (customer_id,)
            )
            row = cur.fetchone()
            if not row:
                raise DataNotFoundError(f"customer_id: {customer_id}")

            features_dict = row["features"]
            if isinstance(features_dict, str):
                try:
                    features_dict = json.loads(features_dict)
                except json.JSONDecodeError:
                    raise PreprocessingError(
                        f"Invalid features format for customer {customer_id}")

            # Convert to 2D list for model
            X = [list(features_dict.values())]

            # Load model
            model_path = get_latest_model(model_type)
            model = load_model_by_type(model_type)

            is_neural_net = model_type.replace(
                "-", "_") == ModelType.NEURAL_NET

            # Predict
            if is_neural_net:
                import torch
                tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    probability_score = model(tensor).cpu().numpy()[0]
                    pred_class = int(np.round(probability_score))
            else:
                pred_class = model.predict(X)[0]
                if hasattr(model, 'predict_proba'):
                    probability_score = model.predict_proba(X)[0][1]
                else:
                    probability_score = float(pred_class)

            # --- Prometheus Metrics Update ---
            PREDICTION_CLASS_COUNT.labels(
                model_name=model_type,
                prediction_class=str(int(pred_class))
            ).inc()
            AVG_PREDICTION_PROBABILITY.labels(
                model_name=model_type).set(probability_score)
            MODEL_PREDICTION_COUNT.labels(model_name=model_type).inc(1)
            CURRENT_BATCH_AVG_PROB.labels(
                model_type=model_type).set(probability_score)
            # --- End Prometheus Metrics Update ---

            prediction_data = {
                "model_type": model_type,
                "prediction": float(pred_class),
                "model_path": model_path,
                "customer_id": customer_id,
                "feature_count": len(features_dict),
                "preprocessing_applied": False,
                "confidence": float(probability_score)
            }

            return PredictionResponse(
                message=f"Prediction successful for customer {customer_id}",
                data=prediction_data
            )

    except Exception as e:
        logger.error(f"Prediction failed for customer {customer_id}: {str(e)}")
        if "not found" in str(e).lower():
            handle_data_error(f"customer_id: {customer_id}", e)
        else:
            handle_model_error(model_type, e)


@router.get("/{model_type}/batch/{batch_id}", response_model=PredictionResponse)
def predict_from_db_batch(
    model_type: str,
    batch_id: str,
    limit: int = 100,
    current_user=Depends(get_current_active_user)
):
    """Return predictions for the *first N* records of a batch."""

    allowed_roles = [UserRole.ADMIN, UserRole.MANAGER, UserRole.SUPERVISOR]
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your role does not give you access to database"
        )

    if model_type not in get_allowed_model_types():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"model_type must be one of: {get_allowed_model_types()}"
        )

    try:
        with get_db_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                """SELECT customer_id, features
                   FROM customer_data
                   WHERE batch_id = %s
                   ORDER BY created_at
                   LIMIT %s""",
                (batch_id, limit)
            )
            rows = cur.fetchall()
            if not rows:
                raise DataNotFoundError(f"batch_id: {batch_id}")

        results = []
        model_path = get_latest_model(model_type)
        model = load_model_by_type(model_type)

        # Prepare for bulk prediction
        feature_list = []
        for r in rows:
            # Assuming features are already preprocessed JSON/dict
            features_dict = r["features"]
            if isinstance(features_dict, str):
                features_dict = json.loads(features_dict)
            feature_list.append(list(features_dict.values()))

        X = np.array(feature_list, dtype=np.float32)

        # Bulk Predict
        if model_type.replace("-", "_") == ModelType.NEURAL_NET:
            import torch
            tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                predictions = model(tensor).cpu().numpy().flatten()
                # Assuming output is probability, convert to class
                pred_classes = (predictions >= 0.5).astype(int)
                probabilities = predictions
        else:
            # Standard Scikit-learn models
            pred_classes = model.predict(X)
            if hasattr(model, 'predict_proba'):
                # Use [:, 1] for probability of the positive class (1)
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = pred_classes.astype(float)

        # Package results
        for i, r in enumerate(rows):
            results.append({
                "customer_id": r["customer_id"],
                "prediction": float(pred_classes[i]),
                "confidence": float(probabilities[i])
            })

        prediction_data = {
            "model_type": model_type,
            "batch_id": batch_id,
            "predictions": results,
            "model_path": model_path,
            "prediction_count": len(results),
            "preprocessing_applied": False
        }

        return PredictionResponse(
            message=f"Batch prediction successful for {len(results)} customers",
            data=prediction_data
        )

    except Exception as e:
        logger.error(f"Batch prediction failed for batch {batch_id}: {str(e)}")
        if "not found" in str(e).lower():
            handle_data_error(f"batch_id: {batch_id}", e)
        else:
            handle_model_error(model_type, e)
