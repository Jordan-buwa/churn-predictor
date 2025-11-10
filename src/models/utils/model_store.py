import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import joblib

try:
    import torch
except ImportError:
    torch = None

# Reuse standardized model types from API utilities
try:
    from src.api.utils.models_types import ModelType, normalize_model_type
except Exception:
    # Fallback if API utilities are not importable in some contexts
    class ModelType:
        XGBOOST = "xgboost"
        RANDOM_FOREST = "random_forest"
        NEURAL_NET = "neural_net"

    def normalize_model_type(model_type: str) -> str:
        return model_type.replace("-", "_")


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", "models")).resolve()


def _extension_for(model_type: str) -> str:
    mt = normalize_model_type(model_type)
    if mt == ModelType.NEURAL_NET:
        return ".pth"
    return ".joblib"


def _ensure_structure(model_type: str) -> Dict[str, Path]:
    base = _model_dir() / normalize_model_type(model_type)
    versions = base / "versions"
    schemas = base / "schemas"
    base.mkdir(parents=True, exist_ok=True)
    versions.mkdir(parents=True, exist_ok=True)
    schemas.mkdir(parents=True, exist_ok=True)
    return {"base": base, "versions": versions, "schemas": schemas}


def _write_metadata(base_dir: Path, metadata: Dict[str, Any]) -> None:
    meta_path = base_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _read_metadata(base_dir: Path) -> Dict[str, Any]:
    meta_path = base_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_model_artifacts(
    model: Any,
    model_type: str,
    metrics: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
    version_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Save model, optional schema, and write/update metadata in a standardized structure.

    Structure:
    models/<model_type>/
      versions/<model_type>_churn_vYYYYMMDD_HHMMSS.ext
      schemas/<version>_schema.json
      metadata.json { latest_version, latest_path, versions: [...] }

    Returns a dict with paths and version info.
    """
    mt = normalize_model_type(model_type)
    paths = _ensure_structure(mt)
    ext = _extension_for(mt)

    # Version naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = version_hint or f"{mt}_churn_v{timestamp}"
    model_filename = f"{version}{ext}"
    model_path = paths["versions"] / model_filename

    # Persist model
    if mt == ModelType.NEURAL_NET:
        if torch is None:
            raise RuntimeError("torch is not available to save neural_net model")
        # Save the entire model for easy retrieval
        torch.save(model, model_path)
        save_format = "torch_object"
    else:
        joblib.dump(model, model_path)
        save_format = "joblib"

    # Write schema if provided
    schema_path = None
    if schema is not None:
        schema_path = paths["schemas"] / f"{version}_schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)

    # Update metadata
    existing = _read_metadata(paths["base"]) or {}
    versions_list = existing.get("versions", [])
    versions_list.append({
        "version": version,
        "path": str(model_path),
        "created_at": datetime.utcnow().isoformat(),
        "format": save_format,
        "metrics": metrics or {},
        "schema_path": str(schema_path) if schema_path else None
    })

    metadata = {
        "model_type": mt,
        "latest_version": version,
        "latest_path": str(model_path),
        "versions": versions_list,
    }
    _write_metadata(paths["base"], metadata)

    # Also maintain a convenience copy named latest.ext (non-symlink for portability)
    latest_copy = paths["base"] / f"latest{ext}"
    try:
        shutil.copy2(model_path, latest_copy)
    except Exception:
        # Best-effort; ignore copy failures
        pass

    return {
        "version": version,
        "model_path": str(model_path),
        "schema_path": str(schema_path) if schema_path else None,
        "metadata_path": str(paths["base"] / "metadata.json"),
        "latest_copy": str(latest_copy)
    }


def get_latest_model_path(model_type: str) -> Optional[str]:
    """Retrieve latest model path using metadata, fallback to latest file in versions."""
    mt = normalize_model_type(model_type)
    paths = _ensure_structure(mt)

    meta = _read_metadata(paths["base"]) or {}
    latest = meta.get("latest_path")
    if latest and Path(latest).exists():
        return latest

    # Fallback: find newest in versions
    candidates = list(paths["versions"].glob(f"*{_extension_for(mt)}"))
    if not candidates:
        return None
    latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest_file)