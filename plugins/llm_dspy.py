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
JUDGE_TEMPERATURE = float(os.getenv("JTBD_JUDGE_TEMPERATURE", "0.5"))  # Higher for score variance
SEED = int(os.getenv("JTBD_LLM_SEED", "42"))
USE_DOUBLE_JUDGE = os.getenv("JTBD_DOUBLE_JUDGE", "1") == "1"  # default ON

def _uid(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]

# Global LM instance - created once at module load
_lm_instance = None

def _create_lm():
    """
    Create DSPy LM instance based on environment config.

    Detects and initializes the appropriate LLM provider based on the
    JTBD_DSPY_MODEL environment variable. Falls back to generic LM
    if provider-specific initialization fails.

    Environment Variables:
        JTBD_DSPY_MODEL: Model identifier (default: "gpt-4o-mini")
        JTBD_LLM_TEMPERATURE: Sampling temperature (default: 0.2)
        JTBD_LLM_SEED: Random seed for reproducibility (default: 42)
    """
    model = os.getenv("JTBD_DSPY_MODEL", "gpt-4o-mini")

    # Check if it's a Claude model
    if "claude" in model.lower():
        try:
            return dspy.Anthropic(model=model, max_tokens=4000, temperature=TEMPERATURE)
        except (ImportError, ValueError) as e:
            # Fallback to generic LM if Anthropic library missing or invalid config
            print(f"Warning: Anthropic initialization failed ({e}), falling back to generic LM")
            return dspy.LM(model=model, max_tokens=4000, temperature=TEMPERATURE)
    else:
        # Try OpenAI first
        try:
            return dspy.OpenAI(model=model, max_tokens=4000, temperature=TEMPERATURE, seed=SEED)
        except (ImportError, ValueError) as e:
            # Fallback to a generic LM if OpenAI library missing or invalid config
            print(f"Warning: OpenAI initialization failed ({e}), falling back to generic LM")
            return dspy.LM(model=model, max_tokens=4000, temperature=TEMPERATURE)

def get_lm():
    """Get or create the global LM instance."""
    global _lm_instance
    if _lm_instance is None:
        _lm_instance = _create_lm()
    return _lm_instance

# Separate judge LM with higher temperature for score variance
_judge_lm_instance = None

def _create_judge_lm():
    """Create DSPy LM for judge scoring with higher temperature."""
    model = os.getenv("JTBD_DSPY_MODEL", "gpt-4o-mini")

    if "claude" in model.lower():
        try:
            return dspy.Anthropic(model=model, max_tokens=4000, temperature=JUDGE_TEMPERATURE)
        except (ImportError, ValueError) as e:
            print(f"Warning: Anthropic judge init failed ({e}), falling back to generic LM")
            return dspy.LM(model=model, max_tokens=4000, temperature=JUDGE_TEMPERATURE)
    else:
        try:
            # No seed for judge to allow variance
            return dspy.OpenAI(model=model, max_tokens=4000, temperature=JUDGE_TEMPERATURE)
        except (ImportError, ValueError) as e:
            print(f"Warning: OpenAI judge init failed ({e}), falling back to generic LM")
            return dspy.LM(model=model, max_tokens=4000, temperature=JUDGE_TEMPERATURE)

def get_judge_lm():
    """Get or create the judge LM instance with higher temperature."""
    global _judge_lm_instance
    if _judge_lm_instance is None:
        _judge_lm_instance = _create_judge_lm()
    return _judge_lm_instance

def get_judge_context():
    """Get a dspy.context() for judge scoring with higher temperature."""
    return dspy.context(lm=get_judge_lm())

def configure_lm() -> None:
    """
    Configure DSPy LM - uses dspy.context() for async compatibility.

    This function is now a no-op for backwards compatibility.
    The actual LM is set via dspy.context() in get_dspy_context().

    For async contexts (like MCP handlers), use get_dspy_context() instead
    to avoid the 'configure can only be called from the same async task' error.
    """
    pass

def get_dspy_context():
    """
    Get a dspy.context() manager with the configured LM.

    Use this in async contexts (like MCP handlers) to avoid the
    'configure can only be called from the same async task' error.

    Returns:
        A context manager that sets the DSPy LM for the current scope.

    Example:
        with get_dspy_context():
            result = my_dspy_module(input)
    """
    return dspy.context(lm=get_lm())

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
    """Score a business idea on 5 criteria using the rubrics below. Be critical and differentiate.

SCORING RUBRICS (use full 0-10 range):

1. Underserved Opportunity (Is there real unmet need?)
   - 1-3: Saturated market, many good solutions exist, no clear gap
   - 4-5: Some competitors, unclear if need is acute
   - 6-7: Clear pain point, few solutions, but need validation
   - 8-10: Obvious gap, strong push forces, no good alternatives

2. Strategic Impact (Does solving this create lasting value?)
   - 1-3: Nice-to-have, won't change buyer's business
   - 4-5: Helpful but not essential, easy to switch away
   - 6-7: Significant efficiency gain or risk reduction
   - 8-10: Mission-critical, clear $ firing event, regulatory/legal driver

3. Market Scale (How big is the addressable market?)
   - 1-3: Tiny niche (<$10M TAM), few potential buyers
   - 4-5: Small market ($10-100M), limited growth
   - 6-7: Solid market ($100M-1B), clear buyer segments
   - 8-10: Large market (>$1B), multiple verticals, platform potential

4. Solution Differentiability (Can this be defensible?)
   - 1-3: Commodity, easily replicated, no moat
   - 4-5: Some differentiation but competitors can copy quickly
   - 6-7: Clear differentiator (data, workflow, integration)
   - 8-10: Strong moat (network effects, proprietary data, regulatory lock-in)

5. Business Model Innovation (Is there pricing power?)
   - 1-3: Race to bottom, cost-plus pricing, low margins
   - 4-5: Standard SaaS pricing, nothing special
   - 6-7: Value-based pricing, clear ROI story
   - 8-10: Novel pricing (outcome-based, per-transaction), captures value created

Return JSON: {"criteria":[{"name":"<criterion>","score":<float>,"rationale":"<specific evidence>"},...], "total":<avg>}"""
    idea_summary: str = dspy.InputField(desc="Business idea title and description")
    assumptions: str = dspy.InputField(desc="Key assumptions with confidence levels and risk scores")
    jobs: str = dspy.InputField(desc="JTBD statements with push/pull/anxiety/inertia forces")
    moat: str = dspy.InputField(desc="Innovation layers and competitive moat analysis")
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
    """Score business ideas on 5 standardized criteria with rationales.

    Accepts rich context including assumptions, jobs, and moat analysis
    to enable differentiated scoring based on actual evidence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.p = _compiled_judge or dspy.Predict(JudgeScoreSig)

    def forward(
        self,
        idea_summary: str,
        assumptions: Optional[List[AssumptionV1]] = None,
        jobs: Optional[List[JobV1]] = None,
        moat: Optional[List[InnovationLayerV1]] = None
    ) -> ScorecardV1:
        """Score a business idea with full context.

        Args:
            idea_summary: Business idea title and description
            assumptions: List of extracted assumptions with confidence/risk
            jobs: List of JTBD statements with Four Forces
            moat: List of innovation layers from moat analysis
        """
        # Format assumptions for context
        assumptions_str = "No assumptions analyzed"
        if assumptions:
            assumption_lines = []
            for a in assumptions[:5]:  # Top 5
                risk_label = "HIGH" if a.confidence < 0.5 else "MED" if a.confidence < 0.8 else "LOW"
                assumption_lines.append(f"- [{risk_label} risk] {a.text} (conf: {a.confidence:.0%})")
            assumptions_str = "\n".join(assumption_lines)

        # Format jobs for context
        jobs_str = "No JTBD analysis"
        if jobs:
            job_lines = []
            for j in jobs[:3]:  # Top 3 jobs
                forces = j.forces or {}
                push = ", ".join(forces.get("push", [])[:2]) or "none"
                pull = ", ".join(forces.get("pull", [])[:2]) or "none"
                anxiety = ", ".join(forces.get("anxiety", [])[:2]) or "none"
                job_lines.append(f"- Job: {j.statement}\n  Push: {push} | Pull: {pull} | Anxiety: {anxiety}")
            jobs_str = "\n".join(job_lines)

        # Format moat for context
        moat_str = "No moat analysis"
        if moat:
            moat_lines = [f"- {m.type}: {m.trigger} → {m.effect}" for m in moat[:4]]
            moat_str = "\n".join(moat_lines)

        out = self.p(
            idea_summary=idea_summary,
            assumptions=assumptions_str,
            jobs=jobs_str,
            moat=moat_str
        )

        try:
            data = json.loads(out.scorecard_json)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw output: {out.scorecard_json}")
            try:
                data = json.loads(repair_json(out.scorecard_json))
            except (json.JSONDecodeError, ValueError, TypeError) as repair_error:
                print(f"JSON repair also failed: {repair_error}")
                data = {"criteria": [], "total": 5.0}

        crits = []
        for item in data.get("criteria", []):
            name = item.get("name")
            if name not in CRITERIA:
                continue
            score = float(item.get("score", 5.0))
            score = max(0.0, min(10.0, score))
            rationale = item.get("rationale", "")
            crits.append(Criterion(name=name, score=score, rationale=rationale))

        # Fill any missing criteria
        present = {c.name for c in crits}
        for name in CRITERIA:
            if name not in present:
                crits.append(Criterion(name=name, score=5.0, rationale="defaulted"))

        total = round(sum(c.score for c in crits) / len(crits), 2)
        return ScorecardV1(target_id="target:final", criteria=crits, total=total)

# --------------- Double-judge arbitration (optional) ---------------
def judge_with_arbitration(
    idea_summary: str,
    assumptions: Optional[List[AssumptionV1]] = None,
    jobs: Optional[List[JobV1]] = None,
    moat: Optional[List[InnovationLayerV1]] = None
) -> ScorecardV1:
    """
    Execute dual-judge scoring with arbitration for improved reliability.

    Uses higher temperature (JUDGE_TEMPERATURE=0.5) for score variance.
    Runs two independent judges with full context and merges their scores
    using tie-breaking logic:
    - If scores differ by ≤2.0: average them
    - If scores differ by >2.0: take the lower (more conservative) score

    Args:
        idea_summary: Business idea title and description
        assumptions: List of extracted assumptions with confidence/risk
        jobs: List of JTBD statements with Four Forces
        moat: List of innovation layers from moat analysis

    Returns:
        ScorecardV1 with arbitrated scores and combined rationales

    Environment:
        JTBD_DOUBLE_JUDGE: Enable dual-judge (default: "1")
        JTBD_JUDGE_TEMPERATURE: Judge temperature (default: 0.5)
    """
    # Use higher temperature judge context for score variance
    with get_judge_context():
        judge = JudgeScore()

        if not USE_DOUBLE_JUDGE:
            return judge(idea_summary=idea_summary, assumptions=assumptions, jobs=jobs, moat=moat)

        j1 = judge(idea_summary=idea_summary, assumptions=assumptions, jobs=jobs, moat=moat)
        j2 = judge(idea_summary=idea_summary, assumptions=assumptions, jobs=jobs, moat=moat)

    # Arbitration: average if close, else take conservative (lower) score
    merged = []
    for name in CRITERIA:
        c1 = next(c for c in j1.criteria if c.name == name)
        c2 = next(c for c in j2.criteria if c.name == name)
        diff = abs(c1.score - c2.score)
        # Increased threshold to 2.0 to allow more variance
        score = (c1.score + c2.score) / 2.0 if diff <= 2.0 else min(c1.score, c2.score)
        rationale = f"J1: {c1.rationale} | J2: {c2.rationale}"
        merged.append(Criterion(name=name, score=round(score, 1), rationale=rationale))

    total = round(sum(c.score for c in merged) / len(merged), 2)
    return ScorecardV1(target_id="target:final", criteria=merged, total=total)
