from typing import List
from contracts.gamma_doc_v1 import GammaDocV1, Section, Asset
from contracts.scorecard_v1 import ScorecardV1
from contracts.assumption_v1 import AssumptionV1
from contracts.job_v1 import JobV1
from contracts.validation_plan_v1 import ValidationPlanV1

def _score_table(s: ScorecardV1) -> str:
    rows = "\n".join([f"| {c.name} | {c.score:.1f} | {c.rationale} |" for c in s.criteria])
    return f"""
| Criterion | Score | Rationale |
|---|---:|---|
{rows}

**Total:** {s.total:.2f}
"""

def render_markdown(doc_id: str,
                    s0: ScorecardV1,
                    s1: ScorecardV1,
                    delta: float,
                    akr: float,
                    assumptions: List[AssumptionV1],
                    jobs: List[JobV1],
                    plan: ValidationPlanV1,
                    asset_paths: dict) -> str:

    assump_md = "\n".join([f"- L{a.level} ({a.confidence:.2f}) — {a.text}" for a in assumptions])
    jobs_md = "\n".join([f"- {j.statement}" for j in jobs])
    exps = plan.stages[0].experiments if plan.stages else []
    plan_md = "\n".join([f"- **{e.exp_id}**: {e.hypothesis} — EVI ${e.evi:.0f}" for e in exps])

    return f"""# JTBD Idea Validator — Report

**ΔScore:** {delta:+.2f}   ·   **AKR:** {akr:.2f}

## Scores (Initial)
{_score_table(s0)}

## Scores (Final)
{_score_table(s1)}

![Radar]({asset_paths['radar']})
![Waterfall]({asset_paths['waterfall']})
![Forces]({asset_paths['forces']})

## Assumptions
{assump_md}

## JTBD (5 perspectives)
{jobs_md}

## Validation Plan (S1)
{plan_md}

*doc:* {doc_id}
"""

def assemble_gamma_doc(doc_id: str, md: str, assets: dict) -> GammaDocV1:
    return GammaDocV1(
        doc_id=doc_id,
        sections=[Section(title="Report", md=md)],
        assets=[Asset(id=k, type=k, path=v) for k,v in assets.items()]
    )
