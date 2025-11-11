import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from .models_types import ModelType

load_dotenv()

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass


class APIConfig:
    """Centralized configuration manager for API endpoints."""
    
    def __init__(self):
        self.repo_root = Path(__file__).resolve().parent.parent.parent.parent
        self.config_dir = self.repo_root / "config"
        self._config_cache = {}
        #self._validate_paths()
        
        # Azure ML configuration
        self.azure_ml_workspace = os.getenv("AZURE_ML_WORKSPACE_NAME")
        self.azure_ml_enabled = bool(self.azure_ml_workspace)
    
    @property
    def use_azure_ml(self) -> bool:
        """Check if Azure ML should be used"""
        return self.azure_ml_enabled and os.getenv("ENVIRONMENT") == "production"
    
    def _validate_paths(self):
        """Validate that all required paths exist"""
        required_dirs = [
            self.repo_root,
            self.config_dir,
            Path(self.model_dir),
            Path(self.cache_dir),
            Path(self.logs_dir),
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                raise ConfigurationError(f"Required directory not found: {directory}")
        
        # Validate critical files
        required_files = [
            Path(self.preprocessing_artifacts_path),
            Path(self.test_data_path),
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise ConfigurationError(f"Required file not found: {file_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from environment or config files."""
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value
        return self._config_cache.get(key, default)
    
    def load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self._config_cache.update(config)
        return config
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        validation_config = self.load_yaml_config("config_api_data-val.yaml")
        registered_models = validation_config.get("registered_models", {})
        return registered_models.get(model_type, {})
    
    @property
    def model_dir(self) -> str:
        """Get models directory path."""
        return self.get("MODEL_DIR", str(self.repo_root / "models"))
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory path."""
        return str(self.repo_root / "src" / "api" / "cache")
    
    @property
    def logs_dir(self) -> str:
        """Get logs directory path."""
        return str(self.repo_root / "src" / "api" / "logs")
    
    @property
    def preprocessing_artifacts_path(self) -> str:
        """Get preprocessing artifacts file path."""
        return self.get("PREPROCESSING_ARTIFACTS_PATH", 
                       str(self.repo_root / "src" / "data_pipeline" / "preprocessing_artifacts.json"))
    
    @property
    def test_data_path(self) -> str:
        """Get test data file path."""
        return self.get("TEST_DATA_PATH", str(self.repo_root / "test_input.json"))
    
    @property
    def predict_input_path(self) -> str:
        """Get predict input file path."""
        return self.get("PREDICT_INPUT_PATH", str(self.repo_root / "src" / "api" / "routers" / "predict_input.json"))
# Global configuration instance
config = APIConfig()

# Convenience functions for common configurations
def get_model_path(model_type: str) -> str:
    """Get the latest model file path for a given model type.

    Prefers structured storage under `models/<type>/versions` with `metadata.json`.
    Falls back to scanning top-level legacy files if needed.
    """
    base_dir = Path(config.model_dir)
    normalized = model_type.replace("-", "_")

    # Structured directory per model type
    structured_base = base_dir / normalized
    versions_dir = structured_base / "versions"
    metadata_path = structured_base / "metadata.json"

    # Try metadata first
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            latest_path = meta.get("latest_path")
            if latest_path and Path(latest_path).exists():
                return latest_path
        except Exception:
            pass

    # Fallback to versions directory
    model_extensions = {
        "neural_net": [".pth", ".pt", ".h5"],
        "xgboost": [".joblib", ".pkl"], 
        "random_forest": [".joblib", ".pkl"],
    }
    ext = model_extensions.get(normalized, [".joblib", ".pth", ".pkl"])

    candidates = []
    if versions_dir.exists():
        for f in versions_dir.iterdir():
            if f.is_file() and f.suffix in ext:
                candidates.append(f)
        if candidates:
            latest_model = max(candidates, key=lambda f: f.stat().st_mtime)
            return str(latest_model)

    # Legacy top-level scan
    if base_dir.exists():
        legacy_files = [
            f for f in base_dir.iterdir()
            if f.is_file() and f.suffix in ext and normalized.replace("-", "") in f.name.lower().replace("-", "")
        ]
        if legacy_files:
            latest_model = max(legacy_files, key=lambda f: f.stat().st_mtime)
            return str(latest_model)

    raise FileNotFoundError(f"No trained {model_type} model found in {base_dir}")

def get_allowed_model_types() -> list:
    """Get list of allowed model types."""
    return ModelType.get_all_types() 