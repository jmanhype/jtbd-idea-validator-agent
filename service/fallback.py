"""
Graceful degradation and fallback mechanisms for LLM failures.
Provides circuit breakers, retry logic, and fallback responses.
"""
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
from enum import Enum

from service.logging_config import logger, security_logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for LLM API calls.
    Prevents cascade failures by temporarily blocking requests after repeated failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        Raises CircuitBreakerOpen exception if circuit is open.
        """
        if self.state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is OPEN. Too many failures. "
                    f"Try again in {self._time_until_retry():.0f}s"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker CLOSED - service recovered")

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Blocking requests for {self.recovery_timeout}s"
            )
            security_logger.error("LLM service circuit breaker opened due to repeated failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _time_until_retry(self) -> float:
        """Calculate time remaining until retry is allowed."""
        if self.last_failure_time is None:
            return 0
        elapsed = time.time() - self.last_failure_time
        return max(0, self.recovery_timeout - elapsed)

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "time_until_retry": self._time_until_retry() if self.state == CircuitState.OPEN else 0
        }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker instance
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    success_threshold=2
)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0
):
    """
    Decorator for retrying failed LLM calls with exponential backoff.

    Usage:
        @retry_with_exponential_backoff(max_retries=3)
        def my_llm_call():
            return llm.invoke(...)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise

                    # Log retry attempt
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

        return wrapper
    return decorator


def get_fallback_response(endpoint: str, error: str) -> Dict[str, Any]:
    """
    Generate fallback response when LLM is unavailable.
    Provides graceful degradation with helpful error messages.
    """
    fallback_responses = {
        "/deconstruct": {
            "assumptions": [
                {
                    "push": "Service temporarily unavailable",
                    "pull": "Please try again in a moment",
                    "anxiety": "We're working to restore service",
                    "inertia": "Your request has been logged"
                }
            ],
            "error": error,
            "fallback": True
        },
        "/jobs": {
            "jobs": [],
            "error": error,
            "fallback": True,
            "message": "Unable to generate jobs at this time. Please try again later."
        },
        "/moat": {
            "layers": [],
            "error": error,
            "fallback": True,
            "message": "Moat analysis temporarily unavailable. Please try again later."
        },
        "/judge": {
            "scorecard": {
                "criteria": [],
                "total": 0,
                "error": error,
                "fallback": True
            },
            "message": "Scoring temporarily unavailable. Please try again later."
        }
    }

    response = fallback_responses.get(endpoint, {
        "error": error,
        "fallback": True,
        "message": "Service temporarily unavailable. Please try again later."
    })

    logger.warning(f"Returning fallback response for {endpoint}: {error}")
    return response


def with_circuit_breaker_and_retry(endpoint: str):
    """
    Decorator that combines circuit breaker and retry logic.
    Returns fallback response if all attempts fail.

    Usage:
        @with_circuit_breaker_and_retry("/deconstruct")
        def deconstruct_handler(req):
            return call_llm(req)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check circuit breaker
                return llm_circuit_breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpen as e:
                # Circuit is open, return fallback immediately
                logger.error(f"Circuit breaker is open for {endpoint}: {e}")
                return get_fallback_response(endpoint, str(e))
            except Exception as e:
                # Other errors - try with retry
                @retry_with_exponential_backoff(max_retries=2, initial_delay=0.5)
                def retry_call():
                    return func(*args, **kwargs)

                try:
                    return retry_call()
                except Exception as retry_error:
                    # All retries failed, return fallback
                    logger.error(f"All retries failed for {endpoint}: {retry_error}")
                    return get_fallback_response(endpoint, str(retry_error))

        return wrapper
    return decorator


def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get current circuit breaker status for monitoring."""
    return llm_circuit_breaker.get_status()
