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
        self.workspace_name = os.getenv("AZURE2_ML_WORKSPACE_NAME")
        self.resource_group = os.getenv("AZURE2_RESOURCE_GROUP")
        self.subscription_id = os.getenv("AZURE2_SUBSCRIPTION_ID")
        self.is_azure_ml = all([self.workspace_name, self.resource_group, self.subscription_id])
    
    def setup_mlflow(self):
        """Setup MLflow tracking URI based on environment"""
        try:
            if self.is_azure_ml:
                self._setup_azure_ml()
            else:
                self._setup_local_mlflow()
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow, using local: {str(e)}")
            mlflow.set_tracking_uri("http://localhost:8080")
    
    def _setup_azure_ml(self):
        """Setup Azure ML tracking"""
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            workspace_name=self.workspace_name,
        )
        
        # Get MLflow tracking URI from Azure ML workspace
        workspace = ml_client.workspaces.get(self.workspace_name)
        mlflow_tracking_uri = workspace.mlflow_tracking_uri
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info("MLflow configured for Azure ML")
    
    def _setup_local_mlflow(self):
        """Setup local MLflow tracking"""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"))
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