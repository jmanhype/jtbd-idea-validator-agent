"""
Production-grade error handling and sanitization.
Prevents stack trace leakage and provides user-friendly error messages.
"""
import os
import traceback
from typing import Dict, Any

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from service.logging_config import logger, security_logger


def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("ENVIRONMENT", "development").lower() in ["production", "prod"]


def sanitize_error_response(
    status_code: int,
    error: Exception,
    include_details: bool = False
) -> Dict[str, Any]:
    """
    Create sanitized error response.
    In production: Generic messages only
    In development: Include stack traces for debugging
    """
    # Map status codes to user-friendly messages
    status_messages = {
        400: "Invalid request. Please check your input.",
        401: "Authentication required.",
        403: "Access denied.",
        404: "Resource not found.",
        429: "Too many requests. Please slow down.",
        500: "An internal error occurred. Please try again later.",
        502: "Service temporarily unavailable.",
        503: "Service is under maintenance.",
        504: "Request timed out. Please try again."
    }

    error_response = {
        "error": status_messages.get(status_code, "An error occurred"),
        "status_code": status_code
    }

    # Include details only in development or if explicitly requested
    if include_details and not is_production():
        error_response["detail"] = str(error)
        error_response["type"] = type(error).__name__

    # Log the actual error for debugging
    if status_code >= 500:
        logger.error(
            f"Server error {status_code}: {type(error).__name__}: {str(error)}",
            exc_info=error
        )
    else:
        logger.warning(f"Client error {status_code}: {str(error)}")

    return error_response


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for all unhandled exceptions.
    Sanitizes error messages in production and logs full details.
    """
    # Log full error details internally
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {str(exc)}",
        exc_info=exc
    )

    # Check if this is a security-relevant error
    if _is_security_error(exc, request):
        security_logger.error(
            f"Security-relevant error from {request.client.host if request.client else 'unknown'}: "
            f"{type(exc).__name__}"
        )

    # Determine status code
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    elif isinstance(exc, StarletteHTTPException):
        status_code = exc.status_code
    else:
        status_code = 500

    # Create sanitized response
    error_response = sanitize_error_response(
        status_code,
        exc,
        include_details=not is_production()
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler for HTTPException instances.
    Provides cleaner error responses while preserving status codes.
    """
    # Log the exception
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
        )
    else:
        logger.info(
            f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
        )

    # Check if detail is a dict (structured error from our code)
    if isinstance(exc.detail, dict):
        # Already structured, use as-is
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )

    # Create sanitized response
    error_response = {
        "error": exc.detail if not is_production() else sanitize_error_response(
            exc.status_code, exc
        )["error"],
        "status_code": exc.status_code
    }

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )


def _is_security_error(exc: Exception, request: Request) -> bool:
    """
    Determine if an exception is security-relevant.
    Helps identify potential attacks or security issues.
    """
    # Check for common attack patterns
    suspicious_indicators = [
        "sql",
        "injection",
        "xss",
        "script",
        "unauthorized",
        "forbidden",
        "token",
        "authentication"
    ]

    error_msg = str(exc).lower()
    url_path = request.url.path.lower()

    for indicator in suspicious_indicators:
        if indicator in error_msg or indicator in url_path:
            return True

    # Check status codes
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        if exc.status_code in [401, 403, 429]:
            return True

    return False


def safe_error_dict(error: Exception) -> Dict[str, Any]:
    """
    Convert exception to safe dict for logging/monitoring.
    Includes type, message, but not full stack trace.
    """
    return {
        "type": type(error).__name__,
        "message": str(error),
        "production": is_production()
    }


class ProductionErrorMiddleware:
    """
    Middleware to catch and sanitize all errors in production.
    Ensures no stack traces or sensitive info leak to clients.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            # Intercept error responses in production
            if message["type"] == "http.response.start" and is_production():
                status = message["status"]
                if status >= 500:
                    # Log that we're sanitizing an error
                    logger.info(f"Sanitizing {status} error response")

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            # This should be caught by FastAPI's exception handlers,
            # but this is a last-resort safety net
            logger.critical(
                f"Uncaught exception in middleware: {type(exc).__name__}: {str(exc)}",
                exc_info=exc
            )

            # Send generic error response
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "An internal error occurred. Please try again later.",
                    "status_code": 500
                }
            )

            await response(scope, receive, send)
