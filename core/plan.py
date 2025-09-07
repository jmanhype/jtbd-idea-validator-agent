from typing import List
from contracts.assumption_v1 import AssumptionV1
from contracts.validation_plan_v1 import ValidationPlanV1, Stage, Experiment

def akr(assumptions: List[AssumptionV1]) -> float:
    l3 = [a for a in assumptions if a.level == 3]
    if not l3: return 0.0
    return round(sum(1.0 - a.confidence for a in l3) / len(l3), 2)

def build_plan(target_id: str, assumptions: List[AssumptionV1]) -> ValidationPlanV1:
    exps: List[Experiment] = []
    for a in assumptions:
        if a.level != 3: continue
        pfail = 1.0 - a.confidence
        loss = 5000.0
        cost = 200.0
        evi  = round(pfail * loss - cost, 2)
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
