"""
Logging and monitoring configuration for JTBD Validator.
Provides sanitized request logging, error tracking, and cost monitoring.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# Configure structured logging
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Main application log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("jtbd_validator")

# Security log for suspicious activity
security_logger = logging.getLogger("jtbd_validator.security")
security_handler = logging.FileHandler(LOG_DIR / "security.log")
security_handler.setFormatter(logging.Formatter(
    '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
))
security_logger.addHandler(security_handler)
security_logger.setLevel(logging.WARNING)

# Cost tracking log
cost_logger = logging.getLogger("jtbd_validator.costs")
cost_handler = logging.FileHandler(LOG_DIR / "costs.log")
cost_handler.setFormatter(logging.Formatter(
    '%(asctime)s - COST - %(message)s'
))
cost_logger.addHandler(cost_handler)
cost_logger.setLevel(logging.INFO)


class MetricsCollector:
    """Collects metrics for monitoring and alerting."""

    def __init__(self):
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.response_times: list[float] = []

    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record a request with timing."""
        self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1
        self.response_times.append(duration)

        if status_code >= 400:
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1

    def record_llm_usage(self, tokens: int, cost: float):
        """Track LLM token usage and costs."""
        self.total_tokens += tokens
        self.total_cost += cost
        cost_logger.info(f"tokens={tokens}, cost=${cost:.4f}, total_cost=${self.total_cost:.4f}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_response = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        return {
            "request_counts": self.request_counts,
            "error_counts": self.error_counts,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_response_time_ms": round(avg_response * 1000, 2)
        }


# Global metrics instance
metrics = MetricsCollector()


def sanitize_for_logging(data: Any, max_length: int = 200) -> str:
    """
    Sanitize data for safe logging.
    - Truncates long strings
    - Removes sensitive patterns (API keys, tokens, emails)
    - Handles nested structures
    """
    if data is None:
        return "null"

    if isinstance(data, str):
        # Remove potential sensitive data patterns
        sanitized = data
        # Redact email addresses
        import re
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', sanitized)
        # Redact API keys (patterns like AIza..., sk-..., etc.)
        sanitized = re.sub(r'\b(AIza[0-9A-Za-z_-]{35}|sk-[0-9A-Za-z]{32,})\b', '[API_KEY]', sanitized)

        # Truncate if too long
        if len(sanitized) > max_length:
            return sanitized[:max_length] + f"...[{len(sanitized)-max_length} chars truncated]"
        return sanitized

    if isinstance(data, dict):
        return {k: sanitize_for_logging(v, max_length) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        if len(data) > 10:
            return f"[list with {len(data)} items, first 3: {[sanitize_for_logging(x, max_length) for x in data[:3]]}]"
        return [sanitize_for_logging(x, max_length) for x in data]

    return str(data)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests with sanitization."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log incoming request (sanitized)
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # Log request
        logger.info(f"{method} {path} from {client_ip}")

        # Check for suspicious patterns
        if self._is_suspicious(request):
            security_logger.warning(
                f"Suspicious request detected: {method} {path} from {client_ip}"
            )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{method} {path} failed after {duration:.2f}s: {type(e).__name__}")
            metrics.record_request(path, duration, 500)
            raise

        # Log response
        duration = time.time() - start_time
        status_code = response.status_code

        logger.info(
            f"{method} {path} completed: {status_code} in {duration:.2f}s"
        )

        # Record metrics
        metrics.record_request(path, duration, status_code)

        # Alert on errors
        if status_code >= 500:
            security_logger.error(f"Server error: {method} {path} returned {status_code}")

        return response

    def _is_suspicious(self, request: Request) -> bool:
        """Detect potentially suspicious requests."""
        suspicious_patterns = [
            "../", "..\\",  # Path traversal
            "<script", "javascript:",  # XSS attempts
            "' OR '", "1=1",  # SQL injection patterns
            "eval(", "exec(",  # Code injection
        ]

        # Check URL path
        path = request.url.path.lower()
        for pattern in suspicious_patterns:
            if pattern.lower() in path:
                return True

        return False


def check_alerts(metrics_data: Dict[str, Any]) -> list[str]:
    """
    Check metrics and return list of alerts.
    """
    alerts = []

    # Alert on high error rate
    total_requests = metrics_data.get("total_requests", 0)
    total_errors = metrics_data.get("total_errors", 0)
    if total_requests > 0:
        error_rate = total_errors / total_requests
        if error_rate > 0.1:  # More than 10% errors
            alerts.append(f"⚠️ High error rate: {error_rate*100:.1f}% ({total_errors}/{total_requests})")

    # Alert on high costs
    total_cost = metrics_data.get("total_cost", 0)
    if total_cost > 10.0:  # More than $10
        alerts.append(f"💰 High API costs: ${total_cost:.2f}")

    # Alert on slow responses
    avg_response = metrics_data.get("avg_response_time_ms", 0)
    if avg_response > 5000:  # More than 5 seconds
        alerts.append(f"🐌 Slow response times: {avg_response:.0f}ms average")

    return alerts
