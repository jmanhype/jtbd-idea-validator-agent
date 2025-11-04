"""
DSPy-based LLM orchestration for JTBD validation.

This module defines DSPy Signatures and Modules for structured business idea
analysis including assumption extraction, JTBD generation, moat analysis,
and scoring with dual-judge arbitration.
"""
import os
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional
import dspy
from json_repair import repair_json
from contracts.assumption_v1 import AssumptionV1
from contracts.job_v1 import JobV1
from contracts.scorecard_v1 import ScorecardV1, Criterion
from contracts.innovation_layer_v1 import InnovationLayerV1

TEMPERATURE = float(os.getenv("JTBD_LLM_TEMPERATURE", "0.2"))
SEED = int(os.getenv("JTBD_LLM_SEED", "42"))
USE_DOUBLE_JUDGE = os.getenv("JTBD_DOUBLE_JUDGE", "1") == "1"  # default ON

def _uid(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]

def configure_lm() -> None:
    """
    Configure DSPy global LLM with provider-specific settings.

    Detects and initializes the appropriate LLM provider based on the
    JTBD_DSPY_MODEL environment variable. Falls back to generic LM
    if provider-specific initialization fails.

    Environment Variables:
        JTBD_DSPY_MODEL: Model identifier (default: "gpt-4o-mini")
        JTBD_LLM_TEMPERATURE: Sampling temperature (default: 0.2)
        JTBD_LLM_SEED: Random seed for reproducibility (default: 42)

    Raises:
        ImportError: If required provider library is not installed
        ValueError: If model configuration is invalid
    """
    model = os.getenv("JTBD_DSPY_MODEL", "gpt-4o-mini")

    # Check if it's a Claude model
    if "claude" in model.lower():
        try:
            lm = dspy.Anthropic(model=model, max_tokens=4000, temperature=TEMPERATURE)
        except (ImportError, ValueError) as e:
            # Fallback to generic LM if Anthropic library missing or invalid config
            print(f"Warning: Anthropic initialization failed ({e}), falling back to generic LM")
            lm = dspy.LM(model=model, max_tokens=4000, temperature=TEMPERATURE)
    else:
        # Try OpenAI first
        try:
            lm = dspy.OpenAI(model=model, max_tokens=4000, temperature=TEMPERATURE, seed=SEED)
        except (ImportError, ValueError) as e:
            # Fallback to a generic LM if OpenAI library missing or invalid config
            print(f"Warning: OpenAI initialization failed ({e}), falling back to generic LM")
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
    """Score business idea on exactly these 5 criteria (0-10 scale) with rationales.
    Return JSON: {"criteria":[{"name":"Underserved Opportunity","score":7.0,"rationale":"Clear need exists..."}, {"name":"Strategic Impact","score":6.0,"rationale":"..."}, {"name":"Market Scale","score":8.0,"rationale":"..."}, {"name":"Solution Differentiability","score":5.0,"rationale":"..."}, {"name":"Business Model Innovation","score":7.0,"rationale":"..."}], "total":6.6}"""
    summary: str = dspy.InputField()
    scorecard_json: str = dspy.OutputField()

# ---------------- Modules ----------------
class Deconstruct(dspy.Module):
    """Extract and classify business assumptions from idea description."""

    def __init__(self) -> None:
        super().__init__()
        self.p = dspy.Predict(DeconstructSig)

    def forward(self, idea: str, hunches: List[str]) -> List[AssumptionV1]:
        out = self.p(idea=idea, hunches=hunches)
        data = json.loads(out.assumptions_json)
        # post-process: bound / defaults
        items = []
        for obj in data[:8]:
            text = obj.get("text","").strip()
            if not text: continue
            level = int(obj.get("level", 2))
            level = 1 if level < 1 else 3 if level > 3 else level
            # Handle confidence - may be numeric or text like "high"/"medium"/"low"
            conf_val = obj.get("confidence", 0.6)
            if isinstance(conf_val, str):
                conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9, "very high": 1.0}
                conf = conf_map.get(conf_val.lower().strip(), 0.6)
            else:
                try:
                    conf = float(conf_val)
                except (ValueError, TypeError):
                    conf = 0.6
            conf = max(0.0, min(1.0, conf))
            items.append(AssumptionV1(
                assumption_id=f"assump:{_uid(text)}", text=text, level=level, confidence=conf,
                evidence=[e for e in obj.get("evidence", []) if isinstance(e, str)]
            ))
        return items

class Jobs(dspy.Module):
    """Generate JTBD statements with Four Forces analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.p = dspy.Predict(JobsSig)

    def forward(self, context: Dict[str, str], constraints: List[str]) -> List[JobV1]:
        out = self.p(context=json.dumps(context), constraints=json.dumps(constraints))
        try:
            arr = json.loads(out.jobs_json)
        except json.JSONDecodeError:
            # Try to repair malformed JSON
            arr = json.loads(repair_json(out.jobs_json))
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
    """Apply Doblin innovation framework for competitive moat analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.p = dspy.Predict(MoatSig)

    def forward(self, concept: str, triggers: str) -> List[InnovationLayerV1]:
        out = self.p(concept=concept, triggers=triggers)
        try:
            arr = json.loads(out.layers_json)
        except json.JSONDecodeError:
            # Try to repair malformed JSON
            arr = json.loads(repair_json(out.layers_json))
        layers = []
        for obj in arr[:6]:
            t = str(obj.get("type","")).strip()
            tr = str(obj.get("trigger","")).strip()
            ef = str(obj.get("effect","")).strip()
            if not t or not tr or not ef: continue
            layers.append(InnovationLayerV1(layer_id=f"layer:{_uid(t+tr+ef)}", type=t, trigger=tr, effect=ef))
        return layers

CRITERIA: List[str] = [
    "Underserved Opportunity",
    "Strategic Impact",
    "Market Scale",
    "Solution Differentiability",
    "Business Model Innovation"
]

JUDGE_COMPILED_PATH: Optional[str] = os.getenv("JTBD_JUDGE_COMPILED")
_compiled_judge = None
if JUDGE_COMPILED_PATH and os.path.exists(JUDGE_COMPILED_PATH):
    try:
        with open(JUDGE_COMPILED_PATH, "rb") as f:
            _compiled_judge = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        print(f"Warning: Failed to load compiled judge from {JUDGE_COMPILED_PATH}: {e}")
        _compiled_judge = None

class JudgeScore(dspy.Module):
    """Score business ideas on 5 standardized criteria with rationales."""

    def __init__(self) -> None:
        super().__init__()
        self.p = _compiled_judge or dspy.Predict(JudgeScoreSig)

    def forward(self, summary: str) -> ScorecardV1:
        out = self.p(summary=summary)
        try:
            data = json.loads(out.scorecard_json)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw output: {out.scorecard_json}")
            try:
                # Try to repair malformed JSON
                data = json.loads(repair_json(out.scorecard_json))
            except (json.JSONDecodeError, ValueError, TypeError) as repair_error:
                # Return default scores if JSON repair also fails
                print(f"JSON repair also failed: {repair_error}")
                data = {"criteria": [], "total": 5.0}
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
    """
    Execute dual-judge scoring with arbitration for improved reliability.

    Runs two independent judges and merges their scores using tie-breaking logic:
    - If scores differ by ≤1.5: average them
    - If scores differ by >1.5: take the lower (more conservative) score

    Args:
        summary: Business idea summary to evaluate

    Returns:
        ScorecardV1 with arbitrated scores and combined rationales

    Environment:
        JTBD_DOUBLE_JUDGE: Enable dual-judge (default: "1")
    """
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
