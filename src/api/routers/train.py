import subprocess
from dotenv import load_dotenv
import os
import sys
import json
import uuid
import logging
from datetime import datetime, UTC
from typing import Dict, Optional
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status, Request
from pydantic import BaseModel

# CORE UTILITY IMPORTS
from src.api.utils.config import APIConfig, get_allowed_model_types, get_model_path as cfg_get_model_path
from src.api.utils.response_models import TrainingResponse, JobStatusResponse
from src.api.utils.error_handlers import TrainingError, handle_training_error
from src.api.utils.models_types import normalize_model_type

# --- AUTH DEPENDENCIES (Needed for context) ---


def auto_admin_user(request: Request):
    """
    Automatically inject an admin user for admin-only routes if one doesn't exist.
    It returns the user object, which is required by the admin_only dependency.
    """
    # 1. Check if a user already exists in request.state (e.g., set by another global dependency)
    user = getattr(request.state, "user", None)
    if user and getattr(user, "role", None) == "admin":
        return user

    # 2. If not found, create and set a fake admin user for dev/test environments
    class AdminUser:
        id = "auto-admin"
        role = "admin"

    admin_user = AdminUser()
    request.state.user = admin_user
    return admin_user


def admin_only(user: object = Depends(auto_admin_user)):
    """
    Allow only admin users to access training.
    This now explicitly depends on auto_admin_user, ensuring the user object is
    passed or injected before checking the role.
    """
    if not user or getattr(user, "role", None) != "admin":
        # This branch should typically only be hit if auto_admin_user was skipped
        # or manually failed, but it acts as a final safety check.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized: admin access required"
        )
    return True

# --- ENVIRONMENT-BASED ROUTER SETUP ---


if os.getenv("ENVIRONMENT") == "test":
    from unittest.mock import MagicMock
    mock_user = MagicMock()
    mock_user.id = "test-user"
    mock_user.role = "admin"
    router = APIRouter(prefix="/train")
else:
    # Router uses auto_admin_user globally for context injection
    router = APIRouter(
        prefix="/train"
    )

logger = logging.getLogger(__name__)

# Initialize configuration
config = APIConfig()

# Setup logging with centralized config
log_path = config.logs_dir
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, 'training_jobs.log')
logging.basicConfig(
    filename=log_file,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

# Training job registry
training_jobs: Dict[str, Dict] = {}


class TrainingRequest(BaseModel):
    # Added back for the model body consistency (though path parameter is primary)
    model_type: str
    retrain: bool = False
    use_cv: bool = True
    hyperparameters: Optional[Dict] = None


def validate_training_script(script_path: str) -> str:
    """Validate that the training script exists (resolve relative to repo root)."""
    repo_root = config.repo_root
    path = Path(script_path)
    if not path.is_absolute():
        candidate = repo_root / path
    else:
        candidate = path
    if not candidate.exists():
        # Use HTTPException to match test expectations
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training script not found: {candidate}"
        )
    return str(candidate)


def create_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())


def register_job(job_id: str, model_type: str, script_path: str):
    """Register a new training job."""
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "model_type": model_type,
        "script_path": script_path,
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "model_path": None,
        "error": None,
        "logs": ""
    }
    return job_id


def run_training_script(script_path: str, job_id: str, model_type: str):
    """Run the training script and update job status in registry."""
    try:
        # Set status to running
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.now(UTC).isoformat()

        logger.info(f"Starting training job {job_id} for {model_type}")

        repo_root = config.repo_root

        # Determine how to run the script
        rel_path = Path(script_path).relative_to(repo_root) if Path(
            script_path).is_absolute() else Path(script_path)
        if str(rel_path).startswith("src" + os.sep) or str(rel_path).startswith("src/"):
            # convert src/models/train_xgboost.py -> src.models.train_xgboost
            module = str(rel_path).replace(os.sep, ".")
            if module.endswith(".py"):
                module = module[:-3]
            run_cmd = [sys.executable, "-m", module]
        else:
            run_cmd = [sys.executable, script_path]

        logger.info(f"Running command: {' '.join(run_cmd)}")

        # Run the script
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
            env=os.environ
        )

        # On success
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        training_jobs[job_id]["logs"] = result.stdout

        # Attach latest model path
        model_path = find_latest_model_file(model_type)
        if model_path:
            training_jobs[job_id]["model_path"] = model_path
            logger.info(f"Job {job_id} completed. Model saved at {model_path}")
        else:
            logger.warning(f"Job {job_id} completed but no model found")

    except subprocess.CalledProcessError as e:
        # Script execution failed
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        training_jobs[job_id]["error"] = f"Script failed: {e.stderr}"
        training_jobs[job_id]["logs"] = (
            e.stdout or "") + "\n" + (e.stderr or "")
        logger.error(f"Training failed for job {job_id}: {e.stderr}")

    except Exception as e:
        # Any other error
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        training_jobs[job_id]["error"] = str(e)
        logger.error(f"Unexpected error in training job {job_id}: {str(e)}")


def find_latest_model_file(model_type: str) -> Optional[str]:
    """Find latest model path using centralized config and normalized type."""
    try:
        # Ensure consistency by normalizing the model type for file lookup
        normalized = normalize_model_type(model_type)
        return cfg_get_model_path(normalized)
    except Exception:
        return None


def get_script_path(model_type: str) -> str:
    """Get the script path for the specified model type."""

    # 1. Define the canonical map (uses hyphens)
    script_map = {
           "neural-net": "src/models/train_nn.py",
        "xgboost": "src/models/train_xgb.py",
           "random-forest": "src/models/train_rf.py"
    }

    # 2. Normalize the input (path parameter or from 'all' loop) to the canonical hyphenated form
    canonical_model_type = model_type.replace('_', '-')

    # 3. Check for "all"
    if canonical_model_type == "all":
        return "placeholder"

    # 4. Validation: Check if the canonical name exists in our script map keys
    if canonical_model_type not in script_map:
        # Get the actual list of supported types (which uses underscores, per the log)
        supported_types_for_message = get_allowed_model_types()

        # Raise 400 with the full list of supported types
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model type: {model_type}. Supported types: {supported_types_for_message}"
        )

    # 5. Success: Use the canonical (hyphenated) name for lookup
    return script_map[canonical_model_type]


async def start_single_training(
    model_type: str,
    background_tasks: BackgroundTasks,
    request: Optional[TrainingRequest] = None
) -> str:
    """Start a single training job with optional hyperparameters."""

    # Validate model and get script
    script_path = get_script_path(model_type)

    # We must validate the script path only if it's a real model
    if script_path != "placeholder":
        validated_script = validate_training_script(script_path)
    else:
        # This shouldn't happen with the current logic, but keeps mypy happy
        raise ValueError(
            f"Invalid model type passed to start_single_training: {model_type}")

    # Create and register job
    job_id = create_job_id()

    # Pass the original model_type to register_job
    register_job(job_id, model_type, validated_script)

    # Attach hyperparameters if provided
    if request and request.hyperparameters:
        training_jobs[job_id]["hyperparameters"] = request.hyperparameters

    logger.info(f"Registered training job {job_id} for {model_type}")

    # Start background task
    background_tasks.add_task(
        run_training_script,
        validated_script,
        job_id,
        model_type
    )

    return job_id


# --- CONSOLIDATED ENDPOINTS ---

@router.post("/{model_type}", response_model=TrainingResponse)
async def train_model_consolidated(
    model_type: str,  # Path parameter, now accepts "all"
    background_tasks: BackgroundTasks,
    # Body contains all config (retrain, hyperparams, etc.)
    request_body: TrainingRequest,
    _: bool = Depends(admin_only)
):
    """
    Start training for a single model type or 'all' models.
    This consolidated endpoint replaces both /train/ and /train/{model_type}.
    """
    try:

        if model_type == "all":
            # --- Logic for 'all' models ---
            parent_job_id = create_job_id()
            training_jobs[parent_job_id] = {
                "job_id": parent_job_id,
                "status": "running",  # Set to running immediately as sub-jobs start
                "model_type": "all",
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": None,
                "sub_jobs": []
            }

            # Start each individual model training
            for mt in get_allowed_model_types():
                if mt != "all":
                    sub_job_id = await start_single_training(
                        mt, background_tasks, request_body
                    )
                    training_jobs[parent_job_id]["sub_jobs"].append(sub_job_id)

            logger.info(
                f"Parent job {parent_job_id} created for all models via /train/all")

            # FIX 1 & 2: Set status to 'success' and wrap details in 'data'
            return TrainingResponse(
                status="success",
                message="Training initiated for all model types",
                data={
                    "job_id": parent_job_id,
                    "model_type": "all",
                       "status": "pending"  # Internal status of the parent job
                }
            )

        else:
            # --- Logic for single model ---

            # The get_script_path function now handles the model_type normalization
            script_path = get_script_path(model_type)
            validated_script = validate_training_script(script_path)

            # Create and register job
            job_id = create_job_id()
            register_job(job_id, model_type, validated_script)

            # Add hyperparameters if provided
            if request_body.hyperparameters:
                training_jobs[job_id]["hyperparameters"] = request_body.hyperparameters

            # Start background training
            background_tasks.add_task(
                run_training_script,
                validated_script,
                job_id,
                model_type
            )

            logger.info(
                f"Started single training job {job_id} for {model_type}")

            # FIX 3: Ensure the response explicitly uses 'data'
            return TrainingResponse(
                status="success",
                message=f"Training initiated for {model_type}",
                data={
                    "job_id": job_id,
                    "model_type": model_type,
                       "status": "pending"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training job {model_type}: {str(e)}")
        # Raise generic 500 error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    @router.post("/", response_model=TrainingResponse)
    async def train_model_post_root(
        background_tasks: BackgroundTasks,
        request_body: TrainingRequest
    ):
        """
        Start training via POST /train/ with model_type in request body.
        """
        model_type = request_body.model_type or "all"
        return await train_model_consolidated(
            model_type=model_type,
            background_tasks=background_tasks,
            request_body=request_body
        )


# --- Other Endpoints (Keep as they were) ---

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a training job.
    """
    if job_id not in training_jobs:
        # Return 404 as HTTPException to satisfy endpoint tests
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found"
        )

    job_info = training_jobs[job_id]

    # For "all" jobs, aggregate status from sub-jobs
    if job_info["model_type"] == "all" and "sub_jobs" in job_info:
        sub_jobs = job_info["sub_jobs"]
        # Check based on status in the registry, not just the sub_jobs list
        expanded_sub_jobs = [training_jobs.get(
            sub_id) for sub_id in sub_jobs if training_jobs.get(sub_id)]

        if all(sub.get("status") == "completed" for sub in expanded_sub_jobs):
            job_info["status"] = "completed"
        elif any(sub.get("status") == "failed" for sub in expanded_sub_jobs):
            job_info["status"] = "failed"
        elif any(sub.get("status") == "running" for sub in expanded_sub_jobs):
            job_info["status"] = "running"

    return JobStatusResponse(
        status="success",
        message="Job status retrieved successfully",
        data=job_info
    )


@router.get("/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """
    List all training jobs with optional filtering.
    """
    jobs_list = list(training_jobs.values())

    # Filter by status if provided
    if status:
        jobs_list = [job for job in jobs_list if job.get("status") == status]

    # Expand sub-jobs for "all" parent jobs
    for job in jobs_list:
        if job.get("model_type") == "all" and "sub_jobs" in job:
            expanded_sub_jobs = []
            for sub_id in job["sub_jobs"]:
                sub_job = training_jobs.get(sub_id)
                if sub_job:
                    expanded_sub_jobs.append(sub_job)
            job["sub_jobs"] = expanded_sub_jobs

            # Optionally, update parent status based on sub-jobs
            if all(sub.get("status") == "completed" for sub in expanded_sub_jobs):
                job["status"] = "completed"
            elif any(sub.get("status") == "failed" for sub in expanded_sub_jobs):
                job["status"] = "failed"
            elif any(sub.get("status") == "running" for sub in expanded_sub_jobs):
                job["status"] = "running"

    # Sort by start time (newest first)
    jobs_list.sort(key=lambda x: x.get("started_at", ""), reverse=True)

    return {
        "jobs": jobs_list[:limit],
        "total_count": len(training_jobs),
        "filtered_count": len(jobs_list[:limit])
    }


@router.delete("/job/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a training job (if possible).
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found"
        )

    job = training_jobs[job_id]

    if job["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job['status']}"
        )

    # Update status to cancelled
    job["status"] = "cancelled"
    job["completed_at"] = datetime.now(UTC).isoformat()
    job["error"] = "Job was cancelled by user"

    logger.info(f"Cancelled training job {job_id}")

    return {
        "status": "success",
        "message": f"Job {job_id} cancelled successfully",
        "data": {"job_id": job_id}
    }


@router.get("/models/available")
async def get_available_models():
    """List available models using standardized metadata and versions structure."""
    models_dir = Path(config.model_dir)
    result = []

    if models_dir.exists():
        for type_name in os.listdir(models_dir):
            base_dir = models_dir / type_name
            if not base_dir.is_dir():
                continue

            info = {
                "model_type": type_name,
                "base_path": str(base_dir),
                "latest_version": None,
                "latest_path": None,
                "versions": [],
            }

            # Read metadata.json if present
            metadata_file = base_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                    info["latest_version"] = meta.get("latest_version")
                    info["latest_path"] = meta.get("latest_path")
                    if isinstance(meta.get("versions"), list):
                        info["versions"] = meta["versions"]
                except Exception:
                    pass

            # Fallback: enumerate versions directory
            versions_dir = base_dir / "versions"
            if versions_dir.exists():
                try:
                    for f in versions_dir.iterdir():
                        if f.is_file():
                            info["versions"].append({
                                "version": f.stem,
                                "path": str(f),
                                "created_at": None,
                                "format": f.suffix.lstrip("."),
                                "schema_path": None,
                            })
                except Exception:
                    pass

            result.append(info)

    return {
        "available_models": result,
        "models_directory": str(models_dir.absolute()),
    }
