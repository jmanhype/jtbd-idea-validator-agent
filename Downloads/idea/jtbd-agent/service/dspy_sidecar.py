import json
from fastapi import FastAPI
from pydantic import BaseModel
from plugins.llm_dspy import configure_lm, Deconstruct, Jobs, Moat, judge_with_arbitration

configure_lm()
app = FastAPI(title="JTBD DSPy Sidecar")

class DeconstructReq(BaseModel):
    idea: str
    hunches: list[str]

@app.post("/deconstruct")
def deconstruct(req: DeconstructReq):
    items = Deconstruct()(idea=req.idea, hunches=req.hunches)
    return {"assumptions": [i.model_dump() for i in items]}

class JobsReq(BaseModel):
    context: dict
    constraints: list[str]

@app.post("/jobs")
def jobs(req: JobsReq):
    jobs = Jobs()(context=req.context, constraints=req.constraints)
    return {"jobs": [j.model_dump() for j in jobs]}

class MoatReq(BaseModel):
    concept: str
    triggers: str = ""

@app.post("/moat")
def moat(req: MoatReq):
    layers = Moat()(concept=req.concept, triggers=req.triggers)
    return {"layers": [l.model_dump() for l in layers]}

class JudgeReq(BaseModel):
    summary: str

@app.post("/judge")
def judge(req: JudgeReq):
    sc = judge_with_arbitration(summary=req.summary)
    return {"scorecard": sc.model_dump()}
