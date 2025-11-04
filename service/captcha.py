"""
CAPTCHA integration for detecting and blocking automated abuse.
Implements both hCaptcha and reCAPTCHA v3 support with fallback to challenge-based protection.
"""
import os
import time
from typing import Optional, Dict
from collections import defaultdict

import httpx
from fastapi import HTTPException, Request

from service.logging_config import security_logger


class RateLimitTracker:
    """
    Simple in-memory rate limit tracker.
    Tracks request frequency per IP to detect suspicious behavior.
    """

    def __init__(self):
        # Track requests per IP: {ip: [(timestamp, endpoint), ...]}
        self.requests: Dict[str, list[tuple[float, str]]] = defaultdict(list)
        # Track failed attempts per IP
        self.failures: Dict[str, int] = defaultdict(int)
        # IPs requiring CAPTCHA verification
        self.captcha_required: set[str] = set()

    def record_request(self, ip: str, endpoint: str):
        """Record a request from an IP."""
        now = time.time()
        # Keep only last 5 minutes of history
        self.requests[ip] = [
            (ts, ep) for ts, ep in self.requests[ip]
            if now - ts < 300  # 5 minutes
        ]
        self.requests[ip].append((now, endpoint))

    def record_failure(self, ip: str):
        """Record a failed request or validation."""
        self.failures[ip] += 1

        # Require CAPTCHA after 5 failures
        if self.failures[ip] >= 5:
            self.captcha_required.add(ip)
            security_logger.warning(
                f"IP {ip} marked for CAPTCHA verification after {self.failures[ip]} failures"
            )

    def is_suspicious(self, ip: str) -> bool:
        """
        Check if an IP shows suspicious behavior.
        Returns True if:
        - More than 20 requests in last minute
        - More than 100 requests in last 5 minutes
        - Multiple failed validations
        """
        now = time.time()
        requests = self.requests.get(ip, [])

        # Count requests in last minute
        last_minute = sum(1 for ts, _ in requests if now - ts < 60)
        if last_minute > 20:
            security_logger.warning(f"IP {ip} exceeded rate limit: {last_minute} req/min")
            return True

        # Count requests in last 5 minutes
        last_5min = len(requests)
        if last_5min > 100:
            security_logger.warning(f"IP {ip} exceeded rate limit: {last_5min} req/5min")
            return True

        # Check failure count
        if self.failures.get(ip, 0) >= 3:
            return True

        return False

    def requires_captcha(self, ip: str) -> bool:
        """Check if an IP requires CAPTCHA verification."""
        return ip in self.captcha_required or self.is_suspicious(ip)

    def clear_captcha_requirement(self, ip: str):
        """Clear CAPTCHA requirement after successful verification."""
        self.captcha_required.discard(ip)
        self.failures[ip] = 0


# Global rate limit tracker
rate_tracker = RateLimitTracker()


async def verify_hcaptcha(token: str, remote_ip: str) -> bool:
    """
    Verify hCaptcha token.
    Returns True if verification succeeds.
    """
    secret = os.getenv("HCAPTCHA_SECRET_KEY")
    if not secret:
        security_logger.warning("hCaptcha secret key not configured")
        return False

    url = "https://hcaptcha.com/siteverify"
    data = {
        "secret": secret,
        "response": token,
        "remoteip": remote_ip
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, timeout=5.0)
            result = response.json()

            if result.get("success"):
                security_logger.info(f"hCaptcha verification succeeded for IP {remote_ip}")
                return True
            else:
                security_logger.warning(
                    f"hCaptcha verification failed for IP {remote_ip}: {result.get('error-codes')}"
                )
                return False

    except Exception as e:
        security_logger.error(f"hCaptcha verification error: {e}")
        return False


async def verify_recaptcha_v3(token: str, remote_ip: str, action: str = "submit") -> bool:
    """
    Verify reCAPTCHA v3 token.
    Returns True if verification succeeds with score >= 0.5.
    """
    secret = os.getenv("RECAPTCHA_SECRET_KEY")
    if not secret:
        security_logger.warning("reCAPTCHA secret key not configured")
        return False

    url = "https://www.google.com/recaptcha/api/siteverify"
    data = {
        "secret": secret,
        "response": token,
        "remoteip": remote_ip
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, timeout=5.0)
            result = response.json()

            if result.get("success"):
                score = result.get("score", 0)
                expected_action = result.get("action", "")

                # Verify action matches
                if expected_action != action:
                    security_logger.warning(
                        f"reCAPTCHA action mismatch: expected {action}, got {expected_action}"
                    )
                    return False

                # Require score >= 0.5 (0.0 = bot, 1.0 = human)
                if score >= 0.5:
                    security_logger.info(
                        f"reCAPTCHA v3 verification succeeded for IP {remote_ip} (score: {score})"
                    )
                    return True
                else:
                    security_logger.warning(
                        f"reCAPTCHA v3 score too low for IP {remote_ip}: {score}"
                    )
                    return False
            else:
                security_logger.warning(
                    f"reCAPTCHA verification failed for IP {remote_ip}: {result.get('error-codes')}"
                )
                return False

    except Exception as e:
        security_logger.error(f"reCAPTCHA verification error: {e}")
        return False


async def check_captcha_required(request: Request, captcha_token: Optional[str] = None):
    """
    Middleware function to check if CAPTCHA is required.
    Raises HTTPException if CAPTCHA is required but not provided or invalid.

    Usage in endpoints:
        @app.post("/endpoint")
        async def endpoint(request: Request, captcha_token: Optional[str] = None):
            await check_captcha_required(request, captcha_token)
            # ... rest of endpoint logic
    """
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    # Record the request
    rate_tracker.record_request(client_ip, endpoint)

    # Check if CAPTCHA is required
    if rate_tracker.requires_captcha(client_ip):
        if not captcha_token:
            security_logger.warning(
                f"CAPTCHA required but not provided for IP {client_ip} on {endpoint}"
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded. CAPTCHA verification required.",
                    "captcha_required": True
                }
            )

        # Verify CAPTCHA token
        # Try hCaptcha first, fall back to reCAPTCHA
        verified = False

        if os.getenv("HCAPTCHA_SECRET_KEY"):
            verified = await verify_hcaptcha(captcha_token, client_ip)

        if not verified and os.getenv("RECAPTCHA_SECRET_KEY"):
            verified = await verify_recaptcha_v3(captcha_token, client_ip, action="api_request")

        if verified:
            # Clear CAPTCHA requirement on successful verification
            rate_tracker.clear_captcha_requirement(client_ip)
            security_logger.info(f"CAPTCHA verification successful for IP {client_ip}")
        else:
            # Record failure
            rate_tracker.record_failure(client_ip)
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "CAPTCHA verification failed",
                    "captcha_required": True
                }
            )


def get_rate_limit_stats() -> Dict:
    """Get statistics about rate limiting and CAPTCHA requirements."""
    return {
        "total_tracked_ips": len(rate_tracker.requests),
        "ips_requiring_captcha": len(rate_tracker.captcha_required),
        "total_failures": sum(rate_tracker.failures.values()),
        "suspicious_ips": [
            ip for ip in rate_tracker.requests.keys()
            if rate_tracker.is_suspicious(ip)
        ]
    }
