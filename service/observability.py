"""Observability utilities for the JTBD DSPy sidecar service.

This module wires OpenTelemetry tracing with an OTLP exporter and provides
request-scoped metadata propagation (Request-ID header + span attributes).
"""

from __future__ import annotations

import os
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor


_REQUEST_ID_CTX: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_TRACING_CONFIGURED = False


def get_request_id() -> Optional[str]:
    """Return the request-id associated with the current context, if any."""

    return _REQUEST_ID_CTX.get()


def _configure_provider() -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return

    service_name = os.getenv("OTEL_SERVICE_NAME", "jtbd-dspy-sidecar")
    deploy_env = os.getenv("DEPLOY_ENV", "dev")
    endpoint = os.getenv("OTLP_ENDPOINT")
    headers = os.getenv("OTLP_HEADERS")

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": deploy_env,
            "modaic.agent_id": os.getenv("MODAIC_AGENT_ID", ""),
            "modaic.agent_rev": os.getenv("MODAIC_AGENT_REV", ""),
        }
    )

    provider = TracerProvider(resource=resource)
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach and propagate a request identifier via headers and span attributes."""

    header_name = "Request-ID"

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        existing = request.headers.get(self.header_name)
        request_id = existing or uuid.uuid4().hex
        token = _REQUEST_ID_CTX.set(request_id)

        # Attach request-id to the span if one exists.
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("request.id", request_id)
            client_host = request.client.host if request.client else ""
            span.set_attribute("http.client_ip", client_host)
            if request.headers.get("content-length"):
                try:
                    span.set_attribute(
                        "http.request_content_length",
                        int(request.headers.get("content-length", "0")),
                    )
                except ValueError:
                    span.set_attribute("http.request_content_length", 0)

        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            _REQUEST_ID_CTX.reset(token)

        response.headers[self.header_name] = request_id
        if span and span.is_recording():
            span.set_attribute("request.status_code", response.status_code)
        return response


def instrument_app(app: FastAPI) -> None:
    """Initialize tracing and attach middleware to a FastAPI app."""

    _configure_provider()

    if not getattr(app.state, "_otlp_instrumented", False):
        FastAPIInstrumentor.instrument_app(app)
        app.state._otlp_instrumented = True

    # Avoid duplicate middleware registration.
    if not any(getattr(m, "cls", None) is RequestIdMiddleware for m in app.user_middleware):
        app.add_middleware(RequestIdMiddleware)
