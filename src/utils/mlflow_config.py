import os
import mlflow
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class AzureMLFlowConfig:
    """Centralized MLflow configuration for Azure ML"""

    def __init__(self):
        pass

    def setup_mlflow(self):
        """Setup MLflow tracking URI based on environment"""
        try:
            self._setup_local_mlflow()
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow, using local: {str(e)}")
            mlflow.set_tracking_uri("http://localhost:8080")

    def _setup_local_mlflow(self):
        """Setup local MLflow tracking"""
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"))

        logger.info("MLflow configured for local tracking")

    def get_experiment_name(self, base_name: str) -> str:
        """Get experiment name with environment prefix"""
        env = os.getenv("ENVIRONMENT", "local")
        return f"{env}-{base_name}"


# Global instance
mlflow_config = AzureMLFlowConfig()


def setup_mlflow():
    """Convenience function to setup MLflow"""
    mlflow_config.setup_mlflow()
