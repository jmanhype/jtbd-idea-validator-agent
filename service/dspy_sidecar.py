import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from plugins.llm_dspy import configure_lm
from service.agent_loader import call_agent_envelope, get_agent, reload_agent
from service.observability import instrument_app
from service.openai_adapter import router as openai_router


configure_lm()
app = FastAPI(title="JTBD DSPy Sidecar")
instrument_app(app)


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
    idea: str
    hunches: list[str]


@app.post("/deconstruct")
def deconstruct(req: DeconstructReq):
    out = call_agent_envelope("deconstruct", {"idea": req.idea, "hunches": req.hunches})
    if "error" in out:
        raise HTTPException(status_code=500, detail=out["error"])
    return {"assumptions": out.get("assumptions", [])}


class JobsReq(BaseModel):
    context: dict
    constraints: list[str]


@app.post("/jobs")
def jobs(req: JobsReq):
    out = call_agent_envelope("jobs", {"context": req.context, "constraints": req.constraints})
    if "error" in out:
        raise HTTPException(status_code=500, detail=out["error"])
    return {"jobs": out.get("jobs", [])}


class MoatReq(BaseModel):
    concept: str
    triggers: str = ""


@app.post("/moat")
def moat(req: MoatReq):
    out = call_agent_envelope("moat", {"concept": req.concept, "triggers": req.triggers})
    if "error" in out:
        raise HTTPException(status_code=500, detail=out["error"])
    return {"layers": out.get("layers", [])}


class JudgeReq(BaseModel):
    summary: str


@app.post("/judge")
def judge(req: JudgeReq):
    out = call_agent_envelope("judge", {"summary": req.summary})
    if "error" in out:
        raise HTTPException(status_code=500, detail=out["error"])
    return {"scorecard": out.get("scorecard", {})}


class QueryReq(BaseModel):
    query: str


@app.post("/agent/query")
def agent_query(req: QueryReq):
    result = get_agent()(req.query)
    try:
        return {"result": json.loads(result)}
    except json.JSONDecodeError:
        return {"result": result}


@app.post("/admin/reload")
def admin_reload():
    source = reload_agent()
    return {"status": "ok", "source": source}


app.include_router(openai_router)
