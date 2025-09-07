from contracts.scorecard_v1 import ScorecardV1, Criterion

WEIGHTS = {
    "Underserved Opportunity": 0.22,
    "Strategic Impact": 0.22,
    "Market Scale": 0.20,
    "Solution Differentiability": 0.18,
    "Business Model Innovation": 0.18
}

def initial_score(target_id: str) -> ScorecardV1:
    crits = [Criterion(name=k, score=5.0, rationale="baseline") for k in WEIGHTS]
    total = round(sum(c.score * WEIGHTS[c.name] for c in crits), 2)
    return ScorecardV1(target_id=target_id, criteria=crits, total=total)

def delta(old: ScorecardV1, new: ScorecardV1) -> float:
    return round(new.total - old.total, 2)
