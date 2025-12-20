"""Authentication middleware for JTBD MCP Server.

Pure ASGI middleware for bearer token authentication.
Uses timing-safe comparison to prevent timing attacks.

Note: SSE and MCP message endpoints are skipped because:
1. BaseHTTPMiddleware breaks SSE streaming
2. Auth for these endpoints is handled by supergateway's --oauth2Bearer flag
"""

import os
import secrets
import json
from typing import Callable, Awaitable

from starlette.types import ASGIApp, Receive, Send, Scope


class BearerTokenMiddleware:
    """Pure ASGI middleware to validate Bearer token from Authorization header.

    This is a pure ASGI middleware (not BaseHTTPMiddleware) to avoid breaking
    SSE streaming responses.

    Configuration:
        JTBD_API_KEY: The expected API key (default: "dev-key-jtbd")

    Usage:
        app = BearerTokenMiddleware(app)
    """

    # Paths that bypass this middleware entirely
    SKIP_AUTH_PATHS = {"/health", "/ready", "/sse"}
    SKIP_AUTH_PREFIXES = ("/messages",)

    def __init__(self, app: ASGIApp, api_key: str | None = None):
        """Initialize the middleware.

        Args:
            app: The ASGI application
            api_key: Optional API key override (defaults to env var JTBD_API_KEY)
        """
        self.app = app
        self.api_key = api_key or os.getenv("JTBD_API_KEY", "dev-key-jtbd")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Process the request and validate authentication."""
        # Only process HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Skip auth for health check and MCP endpoints
        if path in self.SKIP_AUTH_PATHS:
            await self.app(scope, receive, send)
            return

        # Skip auth for paths with matching prefixes
        if any(path.startswith(prefix) for prefix in self.SKIP_AUTH_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Get Authorization header from scope
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode("utf-8")

        # Check for Bearer token
        if not auth_header.startswith("Bearer "):
            await self._send_unauthorized(
                send,
                "Missing or invalid Authorization header. Expected: Bearer <token>",
            )
            return

        # Extract token
        token = auth_header[7:]  # Remove "Bearer " prefix

        # Validate token using timing-safe comparison
        if not secrets.compare_digest(token.encode(), self.api_key.encode()):
            await self._send_unauthorized(send, "Invalid API key")
            return

        # Token is valid, proceed with request
        await self.app(scope, receive, send)

    async def _send_unauthorized(self, send: Send, message: str) -> None:
        """Send a 401 Unauthorized response."""
        body = json.dumps({
            "error": "Unauthorized",
            "message": message,
        }).encode("utf-8")

        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })
