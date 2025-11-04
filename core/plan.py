"""
Validation planning for JTBD assumption testing.

This module creates actionable validation plans for testing business assumptions,
with focus on high-risk Level 3 assumptions that require empirical validation.
"""
from typing import List
from contracts.assumption_v1 import AssumptionV1
from contracts.validation_plan_v1 import ValidationPlanV1, Stage, Experiment


def akr(assumptions: List[AssumptionV1]) -> float:
    """
    Calculate Assumption-Kill-Rate (AKR) risk score.

    AKR measures the average uncertainty in Level 3 (critical) assumptions.
    Higher values indicate more high-risk assumptions with low confidence.

    Args:
        assumptions: List of extracted business assumptions

    Returns:
        AKR score between 0.0 (all confident) and 1.0 (all uncertain)
    """
    l3 = [a for a in assumptions if a.level == 3]
    if not l3:
        return 0.0
    return round(sum(1.0 - a.confidence for a in l3) / len(l3), 2)


def build_plan(target_id: str, assumptions: List[AssumptionV1]) -> ValidationPlanV1:
    """
    Build validation plan for testing critical assumptions.

    Creates experiments for Level 3 assumptions using Expected Value of
    Information (EVI) to prioritize which experiments to run first.

    Args:
        target_id: Identifier for the idea being validated
        assumptions: List of assumptions to potentially test

    Returns:
        ValidationPlanV1 with staged experiments for high-risk assumptions
    """
    exps: List[Experiment] = []
    for a in assumptions:
        if a.level != 3:
            continue
        pfail = 1.0 - a.confidence
        loss = 5000.0
        cost = 200.0
        evi = round(pfail * loss - cost, 2)
        exps.append(Experiment(
            exp_id=f"exp:{a.assumption_id}",
            hypothesis=a.text,
            design="fast falsifier (5–10 user shadow cases)",
            metric="pass/fail threshold",
            success="≥30% time reduction on 10 cases",
            evi=evi
        ))
    stage = Stage(stage="S1 - cheap falsifiers", budget=500.0, experiments=exps)
    return ValidationPlanV1(plan_id=f"plan:{target_id}", stages=[stage])
