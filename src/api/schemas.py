from pydantic import BaseModel, Field
from typing import Optional, Dict
import os
from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pwdlib import PasswordHash
load_dotenv()

API_KEY_SECRET = os.getenv("API_KEY_SECRET", "secret_key")

pwd_hash = PasswordHash.recommended()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_hash.verify(plain_password, hashed_password)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY_SECRET:
        raise HTTPException(
            status_code=403, detail="Invalid or missing API Key")
    return api_key


class TrainingRequest(BaseModel):
    """
    Schema for initiating a model training request.
    """
    model_type: str = Field(
        ..., description="The type of model to train (e.g., 'xgboost', 'random_forest', or 'all').")
    retrain: bool = Field(
        False, description="Whether to ignore existing models and force a retrain.")
    use_cv: bool = Field(
        True, description="Whether to use cross-validation during training.")
    hyperparameters: Optional[Dict] = Field(
        None, description="Optional dictionary of hyperparameters to override defaults.")

    class Config:
        # Example for FastAPI documentation
        json_schema_extra = {
            "example": {
                "model_type": "xgboost",
                "retrain": False,
                "use_cv": True,
                "hyperparameters": {"n_estimators": 500}
            }
        }
