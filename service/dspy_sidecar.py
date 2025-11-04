import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from plugins.llm_dspy import configure_lm
from service.agent_loader import call_agent_envelope, get_agent, reload_agent
from service.observability import instrument_app
from service.openai_adapter import router as openai_router

# Import new security and monitoring modules
from service.logging_config import RequestLoggingMiddleware, logger
from service.monitoring import router as monitoring_router
from service.captcha import check_captcha_required, get_rate_limit_stats
from service.fallback import with_circuit_breaker_and_retry, get_circuit_breaker_status
from service.error_handling import (
    global_exception_handler,
    http_exception_handler,
    ProductionErrorMiddleware
)


configure_lm()
app = FastAPI(title="JTBD DSPy Sidecar")
instrument_app(app)

# Add security and monitoring middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ProductionErrorMiddleware)

# Register exception handlers
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)

# Include monitoring router
app.include_router(monitoring_router)


_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def read_index() -> str:
    if not _FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html is missing")
    return index_path.read_text(encoding="utf-8")


class DeconstructReq(BaseModel):
    idea: str = Field(..., min_length=10, max_length=10000)
    hunches: list[str] = Field(default=[], max_items=20)
    captcha_token: Optional[str] = None

    @validator('idea')
    def validate_idea(cls, v):
        if not v or v.isspace():
            raise ValueError('Idea cannot be empty or whitespace only')
        return v.strip()

    @validator('hunches', each_item=True)
    def validate_hunch_length(cls, v):
        if len(v) > 500:
            raise ValueError('Each hunch must be ≤ 500 characters')
        return v.strip()


@app.post("/deconstruct")
async def deconstruct(request: Request, req: DeconstructReq):
    # Check if CAPTCHA is required for this IP
    await check_captcha_required(request, req.captcha_token)

    # Call with circuit breaker and retry logic
    @with_circuit_breaker_and_retry("/deconstruct")
    def execute_deconstruct():
        out = call_agent_envelope("deconstruct", {"idea": req.idea, "hunches": req.hunches})
        if "error" in out:
            raise HTTPException(status_code=500, detail=out["error"])
        return {"assumptions": out.get("assumptions", [])}

    return execute_deconstruct()


class JobsReq(BaseModel):
    context: dict = Field(..., max_items=50)
    constraints: list[str] = Field(default=[], max_items=20)
    captcha_token: Optional[str] = None

    @validator('constraints', each_item=True)
    def validate_constraint_length(cls, v):
        if len(v) > 500:
            raise ValueError('Each constraint must be ≤ 500 characters')
        return v.strip()


@app.post("/jobs")
async def jobs(request: Request, req: JobsReq):
    # Check if CAPTCHA is required for this IP
    await check_captcha_required(request, req.captcha_token)

    # Call with circuit breaker and retry logic
    @with_circuit_breaker_and_retry("/jobs")
    def execute_jobs():
        out = call_agent_envelope("jobs", {"context": req.context, "constraints": req.constraints})
        if "error" in out:
            raise HTTPException(status_code=500, detail=out["error"])
        return {"jobs": out.get("jobs", [])}

    return execute_jobs()


class MoatReq(BaseModel):
    concept: str = Field(..., min_length=10, max_length=5000)
    triggers: str = Field(default="", max_length=2000)
    captcha_token: Optional[str] = None

    @validator('concept', 'triggers')
    def validate_not_empty(cls, v):
        if v and not v.isspace():
            return v.strip()
        return v


@app.post("/moat")
async def moat(request: Request, req: MoatReq):
    # Check if CAPTCHA is required for this IP
    await check_captcha_required(request, req.captcha_token)

    # Call with circuit breaker and retry logic
    @with_circuit_breaker_and_retry("/moat")
    def execute_moat():
        out = call_agent_envelope("moat", {"concept": req.concept, "triggers": req.triggers})
        if "error" in out:
            raise HTTPException(status_code=500, detail=out["error"])
        return {"layers": out.get("layers", [])}

    return execute_moat()


class JudgeReq(BaseModel):
    summary: str = Field(..., min_length=20, max_length=10000)
    captcha_token: Optional[str] = None

    @validator('summary')
    def validate_summary(cls, v):
        if not v or v.isspace():
            raise ValueError('Summary cannot be empty or whitespace only')
        return v.strip()


@app.post("/judge")
async def judge(request: Request, req: JudgeReq):
    # Check if CAPTCHA is required for this IP
    await check_captcha_required(request, req.captcha_token)

    # Call with circuit breaker and retry logic
    @with_circuit_breaker_and_retry("/judge")
    def execute_judge():
        out = call_agent_envelope("judge", {"summary": req.summary})
        if "error" in out:
            raise HTTPException(status_code=500, detail=out["error"])
        return {"scorecard": out.get("scorecard", {})}

    return execute_judge()


class QueryReq(BaseModel):
    query: str = Field(..., min_length=5, max_length=5000)
    captcha_token: Optional[str] = None

    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


@app.post("/agent/query")
async def agent_query(request: Request, req: QueryReq):
    # Check if CAPTCHA is required for this IP
    await check_captcha_required(request, req.captcha_token)

    # Call with circuit breaker and retry logic
    @with_circuit_breaker_and_retry("/agent/query")
    def execute_query():
        result = get_agent()(req.query)
        try:
            return {"result": json.loads(result)}
        except json.JSONDecodeError:
            return {"result": result}

    return execute_query()


@app.post("/admin/reload")
def admin_reload():
    source = reload_agent()
    return {"status": "ok", "source": source}


@app.get("/status")
def get_status():
    """
    Get current service status including circuit breaker and rate limiting info.
    Useful for monitoring and debugging.
    """
    return {
        "service": "JTBD DSPy Sidecar",
        "status": "running",
        "circuit_breaker": get_circuit_breaker_status(),
        "rate_limits": get_rate_limit_stats()
    }


app.include_router(openai_router)
