"""
Monitoring and alerting endpoints for JTBD Validator.
Provides health checks, metrics, and alert management.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from service.logging_config import metrics, check_alerts, logger


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    checks: Dict[str, bool]


class MetricsResponse(BaseModel):
    metrics: Dict[str, Any]
    alerts: list[str]
    timestamp: str


class AlertConfig(BaseModel):
    error_rate_threshold: float = 0.1  # 10%
    cost_threshold: float = 10.0  # $10
    response_time_threshold: float = 5000.0  # 5 seconds


# Alert configuration
alert_config = AlertConfig()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint for monitoring systems.
    Returns overall health status and individual component checks.
    """
    checks = {
        "api": True,
        "llm_configured": _check_llm_config(),
        "logs_writable": _check_log_directory(),
    }

    # Overall status is healthy only if all checks pass
    status = "healthy" if all(checks.values()) else "degraded"

    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks
    }


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Get current metrics and active alerts.
    Returns request counts, error rates, costs, and performance metrics.
    """
    metrics_data = metrics.get_stats()
    alerts = check_alerts(metrics_data)

    return {
        "metrics": metrics_data,
        "alerts": alerts,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/reset-metrics")
def reset_metrics():
    """
    Reset all metrics counters (use with caution).
    Typically used at the start of a new monitoring period.
    """
    metrics.request_counts.clear()
    metrics.error_counts.clear()
    metrics.total_tokens = 0
    metrics.total_cost = 0.0
    metrics.response_times.clear()

    logger.info("Metrics have been reset")

    return {
        "status": "ok",
        "message": "All metrics reset to zero",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/alerts/config")
def get_alert_config():
    """Get current alert configuration thresholds."""
    return alert_config.dict()


@router.put("/alerts/config")
def update_alert_config(config: AlertConfig):
    """
    Update alert configuration thresholds.
    Allows dynamic adjustment of when alerts fire.
    """
    global alert_config
    alert_config = config

    logger.info(f"Alert configuration updated: {config.dict()}")

    return {
        "status": "ok",
        "config": alert_config.dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


def _check_llm_config() -> bool:
    """Check if LLM is properly configured."""
    # Check if Gemini API key is set
    gemini_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("JTBD_DSPY_MODEL")

    if not gemini_key or not model:
        return False

    return True


def _check_log_directory() -> bool:
    """Check if log directory is writable."""
    from pathlib import Path
    log_dir = Path(__file__).resolve().parent.parent / "logs"

    try:
        # Try to create directory if it doesn't exist
        log_dir.mkdir(exist_ok=True)

        # Try to write a test file
        test_file = log_dir / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()

        return True
    except Exception as e:
        logger.error(f"Log directory check failed: {e}")
        return False
