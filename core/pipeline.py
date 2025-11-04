"""
Main JTBD validation pipeline orchestration.

This module coordinates the complete analysis workflow: assumption deconstruction,
JTBD analysis, moat evaluation, scoring, and report generation.
"""
from typing import Dict, List, Any
from contracts.idea_v1 import IdeaV1
from contracts.assumption_v1 import AssumptionV1
from contracts.job_v1 import JobV1
from contracts.scorecard_v1 import ScorecardV1
from contracts.validation_plan_v1 import ValidationPlanV1
from core.score import initial_score, delta
from core.plan import build_plan, akr
from core.export_gamma import render_markdown, assemble_gamma_doc
from core.export_spreadsheet import export_to_csv
from plugins import llm_dspy, charts_quickchart as charts


def run_pipeline(idea: IdeaV1, assets_dir: str, output_dir: str = "output") -> Dict[str, Any]:
    """
    Execute complete JTBD validation analysis pipeline.

    This function orchestrates the full analysis workflow:
    1. LLM configuration and context enrichment
    2. Assumption deconstruction with confidence scoring
    3. JTBD analysis with Four Forces
    4. Innovation moat analysis (Doblin framework)
    5. Dual-judge scoring with arbitration
    6. Validation plan generation
    7. Chart creation and report export

    Args:
        idea: Business idea to validate (IdeaV1 contract)
        assets_dir: Directory path for saving generated charts/images
        output_dir: Base directory for all output files (default: "output")

    Returns:
        Dictionary containing:
            - gamma_md: Markdown presentation content
            - gamma_doc: Structured Gamma document
            - metrics: Analysis metrics (delta, akr)
            - csv_files: Paths to exported CSV files
            - s0, s1: Initial and final scorecards
            - delta, akr: Score change and risk metrics
            - assumptions, jobs, plan: Analysis artifacts
    """
    llm_dspy.configure_lm()

    # Prepare enriched context from available business information
    enriched_context = dict(idea.context) if idea.context else {}
    
    # Add rich business context when available
    if idea.problem_statement:
        enriched_context["problem"] = idea.problem_statement
    if idea.solution_overview:
        enriched_context["solution"] = idea.solution_overview
    if idea.target_customer:
        enriched_context["customers"] = str(idea.target_customer)
    if idea.competitive_landscape:
        enriched_context["competitors"] = ", ".join(idea.competitive_landscape)
    if idea.revenue_streams:
        enriched_context["revenue_model"] = ", ".join(idea.revenue_streams)
    
    # Prepare enriched constraints/context for analysis
    all_constraints = list(idea.constraints) if idea.constraints else []
    if idea.risks_and_challenges:
        all_constraints.extend(idea.risks_and_challenges)
    
    # Enhanced deconstruction with business context
    deconstruct_input = idea.title
    if idea.problem_statement or idea.solution_overview:
        business_summary = f"{idea.title}. "
        if idea.problem_statement:
            business_summary += f"Problem: {idea.problem_statement} "
        if idea.solution_overview:
            business_summary += f"Solution: {idea.solution_overview}"
        deconstruct_input = business_summary

    # 1) Deconstruct & initial score
    assumptions: List[AssumptionV1] = llm_dspy.Deconstruct()(idea=deconstruct_input, hunches=idea.hunches)
    s0: ScorecardV1 = initial_score(target_id=idea.idea_id)

    # 2) JTBD with enriched context, Moat layers
    jobs: List[JobV1] = llm_dspy.Jobs()(context=enriched_context, constraints=all_constraints)
    
    # Enhanced moat analysis with competitive context (keep simple to avoid JSON parsing issues)
    moat_triggers = ""
    if idea.competitive_landscape:
        # Simplify to avoid JSON parsing errors
        moat_triggers = f"Competitors: {idea.competitive_landscape[0]}" if idea.competitive_landscape else ""
    layers = llm_dspy.Moat()(concept=idea.title, triggers=moat_triggers)

    # 3) Judge with enriched business context
    judge_summary = idea.title
    if idea.value_propositions:
        judge_summary += f" Value props: {', '.join(idea.value_propositions[:3])}"  # Limit to avoid token overflow
    s1: ScorecardV1 = llm_dspy.judge_with_arbitration(summary=judge_summary)

    # 4) ΔScore + AKR + Plan
    d = delta(s0, s1)
    risk = akr(assumptions)
    plan: ValidationPlanV1 = build_plan(idea.idea_id, assumptions)

    # 5) Charts
    radar_path = f"{assets_dir}/radar.png"
    waterfall_path = f"{assets_dir}/waterfall.png"
    forces_path = f"{assets_dir}/forces.png"

    charts.radar(
        initial_vals=[c.score for c in s0.criteria],
        final_vals=[c.score for c in s1.criteria],
        labels=[c.name for c in s0.criteria],
        out_path=radar_path,
    )
    charts.waterfall([("Start", 0.0), ("JTBD+Moat+Judge", d)], out_path=waterfall_path)

    # Use first job's forces as a proxy for chart
    forces = jobs[0].forces if jobs else {"push":[],"pull":[],"anxiety":[],"inertia":[]}
    charts.forces(forces["push"], forces["pull"], forces["anxiety"], forces["inertia"], out_path=forces_path)

    md = render_markdown(
        doc_id=f"gamma:{idea.idea_id}",
        s0=s0, s1=s1, delta=d, akr=risk,
        assumptions=assumptions, jobs=jobs, plan=plan,
        asset_paths={"radar": radar_path, "waterfall": waterfall_path, "forces": forces_path}
    )
    gamma_doc = assemble_gamma_doc(f"gamma:{idea.idea_id}", md, {"radar": radar_path, "waterfall": waterfall_path, "forces": forces_path})

    # Export to spreadsheets
    from pathlib import Path
    
    # Determine CSV directory based on output_dir structure
    csv_dir = output_dir
    if Path(output_dir).name.startswith(("test_idea_", "idea_")):
        # We're in a structured output dir, use csv subdirectory
        csv_subdir = Path(output_dir) / "csv"
        csv_subdir.mkdir(parents=True, exist_ok=True)
        csv_dir = str(csv_subdir)
    
    csv_files = export_to_csv(
        output_dir=csv_dir,
        s0=s0, s1=s1, delta=d, akr=risk,
        assumptions=assumptions, jobs=jobs, plan=plan
    )

    return {
        "gamma_md": md,
        "gamma_doc": gamma_doc.model_dump(),
        "metrics": {"delta": d, "akr": risk},
        "csv_files": csv_files,
        "s0": s0,
        "s1": s1,
        "delta": d,
        "akr": risk,
        "assumptions": assumptions,
        "jobs": jobs,
        "plan": plan
    }
