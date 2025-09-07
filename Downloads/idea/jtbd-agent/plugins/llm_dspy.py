import os, json, hashlib, random
import dspy
from typing import List, Dict, Tuple
from contracts.assumption_v1 import AssumptionV1
from contracts.job_v1 import JobV1
from contracts.scorecard_v1 import ScorecardV1, Criterion
from contracts.innovation_layer_v1 import InnovationLayerV1

TEMPERATURE = float(os.getenv("JTBD_LLM_TEMPERATURE", "0.2"))
SEED = int(os.getenv("JTBD_LLM_SEED", "42"))
USE_DOUBLE_JUDGE = os.getenv("JTBD_DOUBLE_JUDGE", "1") == "1"  # default ON

def _uid(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]

def configure_lm():
    """Configure DSPy global LLM. Edit model name here to your provider choice."""
    model = os.getenv("JTBD_DSPY_MODEL", "gpt-4o-mini")
    
    # Check if it's a Claude model
    if "claude" in model.lower():
        try:
            lm = dspy.Anthropic(model=model, max_tokens=4000, temperature=TEMPERATURE)
        except Exception:
            # Fallback to generic LM
            lm = dspy.LM(model=model, max_tokens=4000, temperature=TEMPERATURE)
    else:
        # Try OpenAI first
        try:
            lm = dspy.OpenAI(model=model, max_tokens=4000, temperature=TEMPERATURE, seed=SEED)
        except Exception:
            # Fallback to a generic LM
            lm = dspy.LM(model=model, max_tokens=4000, temperature=TEMPERATURE)
    dspy.configure(lm=lm)

# ---------------- Signatures ----------------
class DeconstructSig(dspy.Signature):
    """Extract assumptions and classify levels.
    Return JSON list of objects: [{text, level(1..3), confidence, evidence:[]}]"""
    idea: str = dspy.InputField()
    hunches: List[str] = dspy.InputField()
    assumptions_json: str = dspy.OutputField()

class JobsSig(dspy.Signature):
    """Generate 5 distinct JTBD statements with Four Forces (push/pull/anxiety/inertia) each.
    Return JSON list: [{statement, forces:{push:[], pull:[], anxiety:[], inertia:[]}}]"""
    context: str = dspy.InputField()
    constraints: str = dspy.InputField()
    jobs_json: str = dspy.OutputField()

class MoatSig(dspy.Signature):
    """Apply Doblin/10-types + timing/ops/customer/value triggers to strengthen concept.
    Return JSON list: [{type, trigger, effect}]"""
    concept: str = dspy.InputField()
    triggers: str = dspy.InputField()
    layers_json: str = dspy.OutputField()

class JudgeScoreSig(dspy.Signature):
    """Score on 5 criteria with short rationales. Return JSON:
    {criteria:[{name, score(0..10), rationale}], total} """
    summary: str = dspy.InputField()
    scorecard_json: str = dspy.OutputField()

# ---------------- Modules ----------------
class Deconstruct(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(DeconstructSig)
    def forward(self, idea: str, hunches: List[str]):
        out = self.p(idea=idea, hunches=hunches)
        data = json.loads(out.assumptions_json)
        # post-process: bound / defaults
        items = []
        for obj in data[:8]:
            text = obj.get("text","").strip()
            if not text: continue
            level = int(obj.get("level", 2))
            level = 1 if level < 1 else 3 if level > 3 else level
            conf = float(obj.get("confidence", 0.6))
            conf = max(0.0, min(1.0, conf))
            items.append(AssumptionV1(
                assumption_id=f"assump:{_uid(text)}", text=text, level=level, confidence=conf,
                evidence=[e for e in obj.get("evidence", []) if isinstance(e, str)]
            ))
        return items

class Jobs(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(JobsSig)
    def forward(self, context: Dict[str,str], constraints: List[str]):
        out = self.p(context=json.dumps(context), constraints=json.dumps(constraints))
        arr = json.loads(out.jobs_json)
        jobs = []
        seen = set()
        for obj in arr[:12]:
            stmt = obj.get("statement","").strip()
            if not stmt or stmt in seen: continue
            seen.add(stmt)
            forces = obj.get("forces",{}) or {}
            for k in ["push","pull","anxiety","inertia"]:
                forces.setdefault(k, [])
            jobs.append(JobV1(job_id=f"job:{_uid(stmt)}", statement=stmt, forces=forces))
            if len(jobs) >= 5: break
        return jobs

class Moat(dspy.Module):
    def __init__(self): super().__init__(); self.p = dspy.Predict(MoatSig)
    def forward(self, concept: str, triggers: str):
        out = self.p(concept=concept, triggers=triggers)
        arr = json.loads(out.layers_json)
        layers = []
        for obj in arr[:6]:
            t = str(obj.get("type","")).strip()
            tr = str(obj.get("trigger","")).strip()
            ef = str(obj.get("effect","")).strip()
            if not t or not tr or not ef: continue
            layers.append(InnovationLayerV1(layer_id=f"layer:{_uid(t+tr+ef)}", type=t, trigger=tr, effect=ef))
        return layers

CRITERIA = ["Underserved Opportunity","Strategic Impact","Market Scale","Solution Differentiability","Business Model Innovation"]


import pickle
JUDGE_COMPILED_PATH = os.getenv("JTBD_JUDGE_COMPILED")
_compiled_judge = None
if JUDGE_COMPILED_PATH and os.path.exists(JUDGE_COMPILED_PATH):
    try:
        with open(JUDGE_COMPILED_PATH, "rb") as f:
            _compiled_judge = pickle.load(f)
    except Exception:
        _compiled_judge = None

class JudgeScore(dspy.Module):
    def __init__(self): super().__init__(); self.p = _compiled_judge or dspy.Predict(JudgeScoreSig)
    def forward(self, summary: str):
        out = self.p(summary=summary)
        data = json.loads(out.scorecard_json)
        crits = []
        for item in data.get("criteria", []):
            name = item.get("name")
            if name not in CRITERIA: continue
            score = float(item.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            rationale = item.get("rationale","")
            crits.append(Criterion(name=name, score=score, rationale=rationale))
        # Fill any missing criteria to maintain schema shape
        present = {c.name for c in crits}
        for name in CRITERIA:
            if name not in present:
                crits.append(Criterion(name=name, score=5.0, rationale="defaulted"))
        total = round(sum(c.score for c in crits)/len(crits), 2)
        return ScorecardV1(target_id="target:final", criteria=crits, total=total)

# --------------- Double-judge arbitration (optional) ---------------
def judge_with_arbitration(summary: str) -> ScorecardV1:
    if not USE_DOUBLE_JUDGE:
        return JudgeScore()(summary=summary)
    j1 = JudgeScore()(summary=summary)
    j2 = JudgeScore()(summary=summary)
    # Simple tie-breaker: take the criterion-wise average if they differ by <=1.5, else choose the lower.
    merged = []
    for name in CRITERIA:
        c1 = next(c for c in j1.criteria if c.name==name)
        c2 = next(c for c in j2.criteria if c.name==name)
        diff = abs(c1.score - c2.score)
        score = (c1.score + c2.score)/2.0 if diff <= 1.5 else min(c1.score, c2.score)
        rationale = f"arb: {c1.rationale} | {c2.rationale}"
        merged.append(Criterion(name=name, score=round(score,1), rationale=rationale))
    total = round(sum(c.score for c in merged)/len(merged), 2)
    return ScorecardV1(target_id="target:final", criteria=merged, total=total)
