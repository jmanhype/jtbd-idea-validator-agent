"""
Scoring utilities for JTBD idea validation.

This module provides functions for initializing baseline scores and calculating
score deltas for business idea evaluations.
"""
from typing import Dict
from contracts.scorecard_v1 import ScorecardV1, Criterion

# Weights for each scoring criterion (must sum to 1.0)
WEIGHTS: Dict[str, float] = {
    "Underserved Opportunity": 0.22,
    "Strategic Impact": 0.22,
    "Market Scale": 0.20,
    "Solution Differentiability": 0.18,
    "Business Model Innovation": 0.18
}


def initial_score(target_id: str) -> ScorecardV1:
    """
    Generate baseline scorecard with neutral scores for all criteria.

    Args:
        target_id: Unique identifier for the target being scored

    Returns:
        ScorecardV1 with all criteria set to baseline score of 5.0
    """
    crits = [Criterion(name=k, score=5.0, rationale="baseline") for k in WEIGHTS]
    total = round(sum(c.score * WEIGHTS[c.name] for c in crits), 2)
    return ScorecardV1(target_id=target_id, criteria=crits, total=total)


def delta(old: ScorecardV1, new: ScorecardV1) -> float:
    """
    Calculate score improvement between two scorecards.

    Args:
        old: Initial/baseline scorecard
        new: Updated scorecard after analysis

    Returns:
        Difference between new and old total scores (positive = improvement)
    """
    return round(new.total - old.total, 2)
