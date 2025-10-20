"""OpenAI-compatible chat completions endpoint with optional SSE streaming."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import AsyncGenerator, Dict, List, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from opentelemetry import trace

from service.agent_loader import get_agent


router = APIRouter(prefix="/v1")
_TRACER = trace.get_tracer("jtbd.openai")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: Optional[str] = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


def _assert_bearer_token(authorization: Optional[str]) -> None:
    required = os.getenv("API_BEARER_TOKEN")
    if not required:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ", 1)[1]
    if token != required:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _last_user_content(messages: List[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return ""


@router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None),
):
    _assert_bearer_token(authorization)

    request_id = getattr(request.state, "request_id", None) or "unknown"
    user_text = _last_user_content(req.messages)
    created = int(time.time())
    completion_id = f"cmpl_{created}"

    if not req.stream:
        with _TRACER.start_as_current_span("chat.completions") as span:
            span.set_attribute("request.id", request_id)
            span.set_attribute("model", req.model)
            response_text = get_agent()(user_text)

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": response_text},
                }
            ],
            "usage": None,
        }

    async def event_stream() -> AsyncGenerator[dict, None]:
        with _TRACER.start_as_current_span("chat.completions.stream") as span:
            span.set_attribute("request.id", request_id)
            span.set_attribute("model", req.model)
            text = get_agent()(user_text)

        first = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        yield {"event": "message", "data": json.dumps(first)}

        chunk_size = max(1, int(os.getenv("STREAM_CHUNK_SIZE", "60")))
        for idx in range(0, len(text), chunk_size):
            delta_text = text[idx : idx + chunk_size]
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield {"event": "message", "data": json.dumps(payload)}
            await asyncio.sleep(0)

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield {"event": "message", "data": json.dumps(final)}
        yield {"event": "done", "data": "[DONE]"}

    headers = {"Request-ID": request_id, "Content-Type": "text/event-stream"}
    return EventSourceResponse(event_stream(), headers=headers)

