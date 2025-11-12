from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status
from pydantic import BaseModel
import subprocess
from dotenv import load_dotenv
import os
import sys
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path

from src.api.utils.config import APIConfig, get_allowed_model_types
from src.api.utils.response_models import TrainingResponse, JobStatusResponse
from src.api.utils.error_handlers import TrainingError, handle_training_error

# Load environment variables
load_dotenv()

# Setup authentication based on environment
if os.getenv("ENVIRONMENT") == "test":
    from unittest.mock import MagicMock
    mock_user = MagicMock()
    mock_user.id = "test-user"
    router = APIRouter(prefix="/train")
else:
    from src.api.routers.auth import current_active_user
    router = APIRouter(prefix="/train", dependencies=[Depends(current_active_user)])

router = APIRouter(prefix="/train")
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
    level=logging.INFO
)

# Training job registry (use Redis or database in production)
training_jobs: Dict[str, Dict] = {}

class TrainingRequest(BaseModel):
    model_type: str  # "neural-net", "xgboost", "random-forest", "all"
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
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "model_path": None,
        "error": None,
        "logs": ""
    }
    return job_id

def run_training_script(script_path: str, job_id: str, model_type: str, retrain: bool = False, use_cv: bool = True, hyperparameters: Optional[Dict] = None):
    """Run the training script and update job status."""
    try:
        # Update job status to running
        training_jobs[job_id]["status"] = "running"
        training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Starting training job {job_id} for {model_type}")
        
        # Ensure subprocess runs with repository root on PYTHONPATH so `import src` works
        repo_root = config.repo_root
        
        # Build command arguments
        cmd_args = []
        
        # Add flags based on training parameters
        if retrain:
            cmd_args.append("--retrain")
        if not use_cv:
            cmd_args.append("--no-cv")
        
        # Add hyperparameters if provided
        if hyperparameters:
            for key, value in hyperparameters.items():
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
        
        # If the script lives under src/, prefer running it as a module to preserve package imports
        rel_path = Path(script_path).relative_to(repo_root) if Path(script_path).is_absolute() else Path(script_path)
        
        if str(rel_path).startswith("src" + os.sep) or str(rel_path).startswith("src/"):
            # convert src/models/train_xgboost.py -> src.models.train_xgboost
            module = str(rel_path).replace(os.sep, ".")
            if module.endswith(".py"):
                module = module[:-3]
            run_cmd = [sys.executable, "-m", module] + cmd_args
        else:
            run_cmd = [sys.executable, script_path] + cmd_args
        
        logger.info(f"Running command: {' '.join(run_cmd)} (cwd={repo_root})")
        
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
            env=os.environ
        )
        
        # Update job status on success
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["logs"] = result.stdout
        
        # Find the latest model file
        model_path = find_latest_model_file(model_type)
        if model_path:
            training_jobs[job_id]["model_path"] = model_path
            logger.info(f"Training completed for job {job_id}. Model saved: {model_path}")
        else:
            logger.warning(f"Training completed for job {job_id} but no model file found")
            
    except subprocess.CalledProcessError as e:
        # Update job status on failure
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = f"Script execution failed: {e.stderr}"
        training_jobs[job_id]["logs"] = (e.stdout or "") + "\n" + (e.stderr or "")
        logger.error(f"Training failed for job {job_id}: {e.stderr}\nCmd: {getattr(e, 'cmd', None)}")
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        training_jobs[job_id]["error"] = str(e)
        logger.error(f"Unexpected error in training job {job_id}: {str(e)}")

def find_latest_model_file(model_type: str) -> Optional[str]:
    """Find latest model path using centralized config and normalized type."""
    try:
        from src.api.utils.models_types import normalize_model_type
        from src.api.utils.config import get_model_path as cfg_get_model_path
        normalized = normalize_model_type(model_type)
        return cfg_get_model_path(normalized)
    except Exception:
        return None

def get_script_path(model_type: str) -> str:
    """Get the script path for the specified model type."""
    allowed_types = get_allowed_model_types()
    
    script_map = {
        "neural-net": "src/models/train_nn.py",
        "xgboost": "src/models/train_xgb.py", 
        "random-forest": "src/models/train_rf.py"
    }
    
    if model_type not in allowed_types and model_type != "all":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model type: {model_type}. Supported types: {allowed_types}"
        )
    
    if model_type in script_map:
        return script_map[model_type]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No training script configured for model type: {model_type}"
        )

@router.post("/", response_model=TrainingResponse)
async def train_model(
    request_body: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start training for a specific model type or all models.
    
    Args:
        request_body: Training configuration including model type and parameters
        background_tasks: FastAPI background tasks
    """
    try:
        model_type = request_body.model_type
        
        if model_type == "all":
            # Start training for all model types
            job_id = create_job_id()
            training_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "model_type": "all",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "sub_jobs": [],
                "error": None,
                "logs": ""
            }
            
            # Start individual training jobs for each model type
            allowed_types = get_allowed_model_types()
            for mt in allowed_types:
                if mt != "all":  # Skip the "all" option
                    sub_job_id = await start_single_training(
                        mt, background_tasks, request_body
                    )
                    training_jobs[job_id]["sub_jobs"].append(sub_job_id)
            
            logger.info(f"Started training job {job_id} for all model types with {len(training_jobs[job_id]['sub_jobs'])} sub-jobs")
            
            return TrainingResponse(
                job_id=job_id,
                status="started", 
                message="Training initiated for all model types",
                model_type="all"
            )
        else:
            # Single model training
            script_path = get_script_path(model_type)
            validated_script = validate_training_script(script_path)
            
            job_id = create_job_id()
            register_job(job_id, model_type, validated_script)
            
            # Add training parameters to job info
            training_jobs[job_id]["retrain"] = request_body.retrain
            training_jobs[job_id]["use_cv"] = request_body.use_cv
            if request_body.hyperparameters:
                training_jobs[job_id]["hyperparameters"] = request_body.hyperparameters
            
            # Start training in background
            background_tasks.add_task(
                run_training_script, 
                validated_script, 
                job_id, 
                model_type,
                request_body.retrain,
                request_body.use_cv,
                request_body.hyperparameters
            )
            
            logger.info(f"Started training job {job_id} for {model_type}")
            
            return TrainingResponse(
                job_id=job_id,
                status="started",
                message=f"Training initiated for {model_type}",
                model_type=model_type
            )
        
    except HTTPException:
        # Bubble up HTTP errors directly
        raise
    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def start_single_training(
    model_type: str, 
    background_tasks: BackgroundTasks,
    request: TrainingRequest
) -> str:
    """Start training for a single model type."""
    script_path = get_script_path(model_type)
    validated_script = validate_training_script(script_path)
    
    job_id = create_job_id()
    register_job(job_id, model_type, validated_script)
    
    # Add training parameters to job info
    training_jobs[job_id]["retrain"] = request.retrain
    training_jobs[job_id]["use_cv"] = request.use_cv
    if request.hyperparameters:
        training_jobs[job_id]["hyperparameters"] = request.hyperparameters
    
    background_tasks.add_task(
        run_training_script, 
        validated_script, 
        job_id, 
        model_type,
        request.retrain,
        request.use_cv,
        request.hyperparameters
    )
    
    logger.info(f"Started sub-training job {job_id} for {model_type}")
    return job_id

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a training job.
    
    Args:
        job_id: The ID of the training job to check
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found"
        )
    
    job_info = training_jobs[job_id]
    
    # For "all" jobs, aggregate status from sub-jobs
    if job_info["model_type"] == "all" and "sub_jobs" in job_info:
        sub_jobs = job_info["sub_jobs"]
        if not sub_jobs:
            job_info["status"] = "pending"
        elif all(training_jobs.get(sub_id, {}).get("status") == "completed" for sub_id in sub_jobs):
            job_info["status"] = "completed"
        elif any(training_jobs.get(sub_id, {}).get("status") == "failed" for sub_id in sub_jobs):
            job_info["status"] = "failed"
        elif any(training_jobs.get(sub_id, {}).get("status") == "running" for sub_id in sub_jobs):
            job_info["status"] = "running"
        elif all(training_jobs.get(sub_id, {}).get("status") == "pending" for sub_id in sub_jobs):
            job_info["status"] = "pending"
    
    return JobStatusResponse(
        status="success",
        message="Job status retrieved successfully",
        data=job_info
    )

@router.get("/jobs")
async def list_jobs(limit: int = 10, status_filter: Optional[str] = None):
    """
    List all training jobs with optional filtering.
    
    Args:
        limit: Maximum number of jobs to return
        status_filter: Filter by job status ("pending", "running", "completed", "failed")
    """
    jobs_list = list(training_jobs.values())
    
    if status_filter:
        jobs_list = [job for job in jobs_list if job.get("status") == status_filter]
    
    # Sort by start time (newest first)
    jobs_list.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    
    return {
        "jobs": jobs_list[:limit],
        "total_count": len(jobs_list),
        "filtered_count": len(jobs_list[:limit])
    }

@router.delete("/job/cancel/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a training job (if possible).
    
    Note: This is a basic implementation. For full cancellation support,
    you would need to implement process management.
    """
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found"
        )
    
    job = training_jobs[job_id]
    
    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job['status']}"
        )
    
    # Update status to cancelled
    job["status"] = "cancelled"
    job["completed_at"] = datetime.utcnow().isoformat()
    job["error"] = "Job was cancelled by user"
    
    # Also cancel sub-jobs for "all" jobs
    if job["model_type"] == "all" and "sub_jobs" in job:
        for sub_job_id in job["sub_jobs"]:
            if sub_job_id in training_jobs and training_jobs[sub_job_id]["status"] in ["pending", "running"]:
                training_jobs[sub_job_id]["status"] = "cancelled"
                training_jobs[sub_job_id]["completed_at"] = datetime.utcnow().isoformat()
                training_jobs[sub_job_id]["error"] = "Parent job was cancelled"
    
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