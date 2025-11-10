# Train API Documentation

## Overview

The **Train API** provides endpoints to trigger, monitor, and manage model training tasks.  
It allows you to start training jobs for specific model types such as **XGBoost**, **Random Forest**, or **Neural Network**, all managed asynchronously through FastAPI background tasks.

Training scripts are defined under:

```text
src/models/
|-- train.py
|-- train_rf.py
|-- train_xgb.py
```

## Endpoints Summary

| Method | Endpoint | Description |
|--------|-----------|-------------|
| **POST** | `/train/{model_type}` | Start training for a specific model type |
| **POST** | `/train` | Start training with configuration (supports "all" models) |
| **GET** | `/train/status/{job_id}` | Get the current status of a training job |
| **GET** | `/train/jobs` | List recent or filtered training jobs |
| **DELETE** | `/train/job/{job_id}` | Cancel a training job.(**Note: This is a soft cancel, it updates the job status but does not terminate the running script process**) |
| **GET** | `/train/models/available` | Get list of available trained model files |

---

## Request Models

### **TrainingRequest**
| Field | Type | Default | Description |
|--------|------|----------|-------------|
| `model_type` | `str` | required | `"neural-net"`, `"xgboost"`, `"random-forest"`, or `"all"` |
| `retrain` | `bool` | `False` | Whether to overwrite existing model |
| `use_cv` | `bool` | `True` | Use cross-validation during training |
| `hyperparameters` | `dict` | `None` | Optional hyperparameter overrides |

### **Training Response**
| Field | Type | Description |
|--------|------|-------------|
| `job_id` | `str` | Unique identifier for the job |
| `status` | `str` | `"started"`, `"running"`, `"completed"`, `"failed"` |
| `message` | `str` | Training status message |
| `model_type` | `str` | Type of model being trained |

---

## POST `/train/{model_type}`

Trigger training for a **specific** model type.

### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/train/xgboost"
```
### Example Response
```json
{
  "status": "success",
  "message": "Training initiated for xgboost",
  "data": {
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "model_type": "xgboost",
    "status": "started"
  }
}
```
## POST `/train` (Full Configuration)
Trigger training with full configuration **(single or all models)**.
### Example Request
```bash
curl -X POST "http://127.0.0.1:8000/train" \
-H "Content-Type: application/json" \
-d '{
  "model_type": "all",
  "retrain": true,
  "use_cv": true,
  "hyperparameters": {
    "learning_rate": 0.01,
    "max_depth": 5
  }
}'
```
### Example Response
```json
{
  "job_id": "789a4567-e89b-12d3-a456-426614174999",
  "status": "started",
  "message": "Training initiated for all model types",
  "model_type": "all"
}
```
## GET `/train/status/{job_id}`
### Retrieve the status of a training job.
### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/train/status/123e4567-e89b-12d3-a456-426614174000" 
```
### Example Response
```json
{
  "status": "success",
  "message": "Job status retrieved successfully",
  "data": {
    "job_id": "123e4567-e89b-12d3-a456-426614174000",
    "status": "running",
    "model_type": "xgboost",
    "started_at": "2025-11-07T12:30:00Z",
    "completed_at": null,
    "model_path": null,
    "error": null,
    "logs": ""
  }
}
```
## GET `/train/jobs`
### List all recent training jobs with optional filtering.
### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/train/jobs?limit=5&status=running"
```
### Example Response
```json
{
  "jobs": [
    {
      "job_id": "123e4567-e89b-12d3-a456-426614174000",
      "status": "running",
      "model_type": "xgboost"
    }
  ],
  "total_count": 10,
  "filtered_count": 1
}
```
## DELETE `/train/job/cancel/{job_id}`
### Cancel a training job (soft cancel).
### Example Request
```bash
curl -X DELETE "http://127.0.0.1:8000/train/job/cancel/123e4567-e89b-12d3-a456-426614174000"
```
### Example Response
```json
{
  "status": "success",
  "message": "Job 123e4567-e89b-12d3-a456-426614174000 cancelled successfully",
  "data": {
    "job_id": "123e4567-e89b-12d3-a456-426614174000"
  }
}
```
## GET `/train/models/available`

### List available trained models using standardized metadata and versions.
### Example Request
```bash
curl -X GET "http://127.0.0.1:8000/train/models/available"
```
### Example Response
```json
{
  "available_models": [
    {
      "model_type": "xgboost",
      "base_path": "models/xgboost",
      "latest_version": "xgboost_churn_v20251110_153245",
      "latest_path": "models/xgboost/versions/xgboost_churn_v20251110_153245.joblib",
      "versions": [
        {
          "version": "xgboost_churn_v20251110_153245",
          "path": "models/xgboost/versions/xgboost_churn_v20251110_153245.joblib",
          "created_at": "2025-11-10T15:32:45Z",
          "format": "joblib",
          "schema_path": "models/xgboost/schemas/xgboost_churn_v20251110_153245_schema.json"
        }
      ]
    },
    {
      "model_type": "neural_net",
      "base_path": "models/neural_net",
      "latest_version": "neural_net_churn_v20251110_153300",
      "latest_path": "models/neural_net/versions/neural_net_churn_v20251110_153300.pth",
      "versions": [
        {
          "version": "neural_net_churn_v20251110_153300",
          "path": "models/neural_net/versions/neural_net_churn_v20251110_153300.pth",
          "created_at": "2025-11-10T15:33:00Z",
          "format": "pth",
          "schema_path": "models/neural_net/schemas/neural_net_churn_v20251110_153300_schema.json"
        }
      ]
    }
  ],
  "models_directory": "/home/user/project/models"
}
```

#### Notes
- Data is sourced from `metadata.json` under `models/<type>/` when available, with fallback scanning of `models/<type>/versions/`.
- `latest_version` and `latest_path` indicate the current default used by prediction endpoints.
- `schema_path` points to the saved training schema stored under `models/<type>/schemas/`.
