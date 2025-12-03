from fastapi import APIRouter, Depends, HTTPException, Response, status
from datetime import timedelta
import logging

from src.monitoring.drift import DriftDetector
from src.api.db import UserRole
from src.api.authenticator import get_current_active_user

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/prometheus_metrics", tags=["prometheus"])

# Initializing DriftDetector once
detector = DriftDetector()

@router.get("/")
def get_drift_metrics(
    current_user = Depends(get_current_active_user),
    mode: str = "row_by_row",  # "row_by_row" or "batch"
    lookback_hours: int = 1
):
    """
    Return Prometheus-compatible drift metrics.

    - Only admin users allowed.
    - mode: "row_by_row" or "batch"
    - lookback_hours: only relevant for batch mode (last N hours)
    """
    # Admin check 
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can access drift metrics"
        )

    # Run drift check before returning model metrics
    try:
        lookback = timedelta(hours=lookback_hours)
        # user_id is required for admin check inside DriftDetector
        result = detector.run_drift_check(user_id=current_user.id, mode=mode, lookback=lookback)
        logger.info(f"Drift check completed: {result['drifted_features']}/{result['total_features']} features drifted")
    except Exception as e:
        logger.exception(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Drift check failed: {e}")

    # Return Prometheus metrics
    payload, content_type = detector.prometheus_metrics_response()
    return Response(content=payload, media_type=content_type)
