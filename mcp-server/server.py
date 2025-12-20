#!/usr/bin/env python3
"""
JTBD Idea Validator MCP Server

Exposes the JTBD analysis pipeline as MCP tools for use with Claude Code.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import structlog

# Import market reality functions
from plugins.market_reality import (
    search_competitors,
    detect_platform_risk,
    scan_regulatory_requirements,
)

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure structlog based on DEBUG environment variable
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

if DEBUG:
    # Pretty console output for development
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
else:
    # JSON output for production
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

logger = structlog.get_logger()

# Initialize MCP server
mcp = FastMCP("jtbd-validator")


# =============================================================================
# RESOURCES - Expose reports as browsable MCP resources
# =============================================================================

@mcp.resource("reports://recent")
def get_recent_reports_resource() -> str:
    """List of recent JTBD validation reports."""
    reports_dir = Path(__file__).parent.parent / "reports"

    if not reports_dir.exists():
        return "# Recent Reports\n\nNo reports found."

    lines = ["# Recent JTBD Validation Reports\n"]
    count = 0

    for date_dir in sorted(reports_dir.iterdir(), reverse=True)[:7]:
        if not date_dir.is_dir():
            continue
        for report_dir in sorted(date_dir.iterdir(), reverse=True):
            if not report_dir.is_dir():
                continue
            json_file = report_dir / "json" / "analysis_data.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    title = data.get("idea", {}).get("title", "Unknown")
                    delta = data.get("delta", 0)
                    akr = data.get("akr", 0)
                    lines.append(f"## {title}")
                    lines.append(f"- **Date**: {date_dir.name}")
                    lines.append(f"- **Delta Score**: {delta}")
                    lines.append(f"- **AKR**: {akr}")
                    lines.append(f"- **Path**: `{report_dir}`")
                    lines.append("")
                    count += 1
                except Exception:
                    pass
            if count >= 10:
                break
        if count >= 10:
            break

    if count == 0:
        lines.append("No reports found.")

    return "\n".join(lines)


@mcp.resource("reports://latest")
def get_latest_report_resource() -> str:
    """The most recent JTBD validation report."""
    reports_dir = Path(__file__).parent.parent / "reports"

    if not reports_dir.exists():
        return "# Latest Report\n\nNo reports found."

    for date_dir in sorted(reports_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for report_dir in sorted(date_dir.iterdir(), reverse=True):
            if not report_dir.is_dir():
                continue
            md_file = report_dir / "gamma" / "presentation.md"
            if md_file.exists():
                return md_file.read_text()

    return "# Latest Report\n\nNo reports found."


# =============================================================================
# PROMPTS - Pre-built prompt templates for common validation scenarios
# =============================================================================

@mcp.prompt()
def validate_startup(
    idea: str,
    target_market: str = "general consumers",
) -> str:
    """Validate a startup idea with JTBD methodology.

    Use this prompt to get a comprehensive analysis of a new startup concept.
    """
    return f"""Please validate this startup idea using the JTBD (Jobs to be Done) methodology.

**Startup Idea**: {idea}
**Target Market**: {target_market}

Use the `validate_idea` tool with these inputs:
- title: "{idea}"
- problem_statement: "Identify the core problem this startup solves for {target_market}"
- solution_overview: "{idea}"
- hunches: Generate 3-5 hypotheses about why this might succeed
- target_customer: "{target_market}"

After running the validation, summarize:
1. The key Jobs to Be Done this addresses
2. The Four Forces analysis (push, pull, anxiety, habit)
3. Top 3 riskiest assumptions to validate
4. Recommended next steps for validation"""


@mcp.prompt()
def validate_feature(
    feature: str,
    product: str,
    user_segment: str = "existing users",
) -> str:
    """Validate a new product feature idea.

    Use this prompt to analyze whether a feature is worth building.
    """
    return f"""Please validate this feature idea using JTBD methodology.

**Feature**: {feature}
**Product**: {product}
**User Segment**: {user_segment}

Use the `validate_idea` tool with these inputs:
- title: "{feature} for {product}"
- problem_statement: "What job does {feature} help {user_segment} accomplish?"
- solution_overview: "{feature}"
- hunches: Generate 3-4 hypotheses about user adoption
- target_customer: "{user_segment}"

Focus your analysis on:
1. Does this feature address a real Job to Be Done?
2. What's the switching cost for users?
3. How does this compare to existing solutions?
4. What's the minimum viable experiment to validate demand?"""


@mcp.prompt()
def compare_ideas(
    idea1: str,
    idea2: str,
) -> str:
    """Compare two business ideas head-to-head.

    Use this prompt to decide between two competing concepts.
    """
    return f"""Please compare these two business ideas using JTBD methodology.

**Idea 1**: {idea1}
**Idea 2**: {idea2}

For each idea, use the `validate_idea` tool, then create a comparison:

| Criteria | {idea1[:30]}... | {idea2[:30]}... |
|----------|-----------------|-----------------|
| Delta Score | ? | ? |
| AKR | ? | ? |
| Market Size | ? | ? |
| Validation Cost | ? | ? |
| Time to Validate | ? | ? |

Provide a recommendation on:
1. Which idea has stronger JTBD fit
2. Which has lower validation risk
3. Which should be pursued first and why"""


@mcp.prompt()
def review_report(report_path: str = "") -> str:
    """Review and summarize an existing validation report.

    Use this prompt to get insights from a previous analysis.
    """
    path_instruction = f'Use the `get_report` tool with path: "{report_path}"' if report_path else "First use `list_recent_reports` to find available reports, then use `get_report` to fetch one."

    return f"""Please review and summarize a JTBD validation report.

{path_instruction}

After retrieving the report, provide:
1. **Executive Summary** (2-3 sentences)
2. **Key Findings**:
   - Strongest JTBD identified
   - Most critical assumption
   - Biggest risk
3. **Recommended Actions**:
   - Immediate next step
   - Validation experiment to run
   - Decision criteria for go/no-go"""


@mcp.tool()
async def validate_idea(
    title: str,
    problem_statement: str,
    solution_overview: str,
    hunches: list[str],
    target_customer: Optional[str] = None,
    competitive_landscape: Optional[list[str]] = None,
    revenue_streams: Optional[list[str]] = None,
    value_propositions: Optional[list[str]] = None,
) -> dict:
    """
    Validate a business idea using JTBD (Jobs to be Done) methodology.

    Returns a comprehensive analysis including:
    - Score delta (improvement from baseline)
    - AKR (Assumption Kill Rate)
    - 5 JTBD perspectives with Four Forces
    - Key assumptions to validate
    - Validation plan

    Args:
        title: Business idea title (e.g., "AI-powered fitness coach")
        problem_statement: The problem being solved
        solution_overview: Brief description of the solution
        hunches: List of hypotheses about the market/customers (3-5 recommended)
        target_customer: Description of target customer segment
        competitive_landscape: List of competitors or alternatives
        revenue_streams: Planned revenue sources
        value_propositions: Key value props for customers
    """
    start_time = time.time()
    log = logger.bind(tool="validate_idea", title=title)
    log.info("Starting idea validation")

    try:
        from contracts.idea_v1 import IdeaV1
        from core.pipeline import run_pipeline
        import hashlib

        # Generate idea ID
        idea_id = f"urn:idea:{hashlib.sha1(title.encode()).hexdigest()[:12]}"

        # Build IdeaV1 object
        # Convert target_customer string to dict format if provided
        target_customer_dict = {"description": target_customer} if target_customer else None

        idea = IdeaV1(
            idea_id=idea_id,
            title=title,
            problem_statement=problem_statement,
            solution_overview=solution_overview,
            hunches=hunches,
            target_customer=target_customer_dict,
            competitive_landscape=competitive_landscape or [],
            revenue_streams=revenue_streams or [],
            value_propositions=value_propositions or [],
            constraints=[],
            risks_and_challenges=[],
            context={},
        )

        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_folder = datetime.now().strftime("%Y-%m-%d")
        base_dir = Path(__file__).parent.parent / f"reports/{date_folder}/mcp_{timestamp}"
        assets_dir = base_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        log.info("Running validation pipeline", idea_id=idea_id, report_dir=str(base_dir))

        # =============================================================================
        # Market Reality Analysis - Before pipeline
        # =============================================================================

        market_reality = {}

        # 1. Competitor Search (async, optional if EXA_API_KEY is set)
        exa_api_key = os.getenv("EXA_API_KEY")
        if exa_api_key:
            try:
                log.info("Searching for competitors via Exa API")
                competitor_data = await search_competitors(
                    idea_title=title,
                    industry=target_customer or "general"
                )
                market_reality["validated_competitors"] = competitor_data.get("validated_competitors", [])
                log.info("Competitor search completed", total_found=competitor_data.get("total_results", 0))
            except Exception as e:
                log.warning("Competitor search failed", error=str(e))
                market_reality["validated_competitors"] = []
                market_reality["competitor_search_error"] = str(e)
        else:
            log.info("Skipping competitor search - no EXA_API_KEY")
            market_reality["validated_competitors"] = []
            market_reality["competitor_search_note"] = "Competitor search skipped - no EXA_API_KEY"

        # 2. Platform Risk Detection (sync)
        try:
            log.info("Detecting platform risk")
            platform_risk = detect_platform_risk(
                idea_title=title,
                solution_overview=solution_overview
            )
            market_reality["platform_risk"] = {
                "level": platform_risk.get("risk_level", "UNKNOWN"),
                "warnings": platform_risk.get("warnings", []),
                "platform_dependencies": platform_risk.get("platform_dependencies", [])
            }
            log.info("Platform risk detected", level=platform_risk.get("risk_level"))
        except Exception as e:
            log.warning("Platform risk detection failed", error=str(e))
            market_reality["platform_risk"] = {
                "level": "UNKNOWN",
                "warnings": [f"Error detecting platform risk: {str(e)}"],
                "platform_dependencies": []
            }

        # 3. Regulatory Requirements Scan (sync)
        try:
            log.info("Scanning regulatory requirements")
            regulatory_scan = scan_regulatory_requirements(
                idea_title=title,
                industry=target_customer or "general",
                target_regions=None  # Could be enhanced to extract from idea context
            )
            market_reality["regulatory_requirements"] = regulatory_scan.get("regulations", [])
            market_reality["detected_industry"] = regulatory_scan.get("detected_industry", "General")
            market_reality["compliance_notes"] = regulatory_scan.get("compliance_notes", [])
            log.info("Regulatory scan completed",
                    detected_industry=regulatory_scan.get("detected_industry"),
                    regulations_found=len(regulatory_scan.get("regulations", [])))
        except Exception as e:
            log.warning("Regulatory scan failed", error=str(e))
            market_reality["regulatory_requirements"] = []
            market_reality["regulatory_scan_error"] = str(e)

        # Run the pipeline
        result = run_pipeline(idea, str(assets_dir), str(base_dir))

        duration = time.time() - start_time
        log.info(
            "Validation completed successfully",
            duration_seconds=round(duration, 2),
            delta_score=result["delta"],
            akr=result["akr"],
        )

        # Format response for MCP
        return {
        "success": True,
        "summary": {
            "title": title,
            "delta_score": result["delta"],
            "akr": result["akr"],
            "initial_score": result["s0"].total,
            "final_score": result["s1"].total,
        },
        "scores": {
            "initial": {c.name: c.score for c in result["s0"].criteria},
            "final": {c.name: {"score": c.score, "rationale": c.rationale} for c in result["s1"].criteria},
        },
        "assumptions": [
            {
                "text": a.text,
                "level": a.level,
                "confidence": a.confidence,
            }
            for a in result["assumptions"]
        ],
        "jobs": [
            {
                "statement": j.statement,
                "forces": j.forces,
            }
            for j in result["jobs"]
        ],
        "validation_plan": {
            "stages": [
                {
                    "stage": stage.stage,
                    "budget": stage.budget,
                    "experiments": [
                        {
                            "exp_id": exp.exp_id,
                            "hypothesis": exp.hypothesis,
                            "design": exp.design,
                            "metric": exp.metric,
                            "success": exp.success,
                            "evi": exp.evi,
                        }
                        for exp in stage.experiments
                    ]
                }
                for stage in result["plan"].stages
            ]
        },
        "market_reality": market_reality,
        "report_path": str(base_dir / "gamma" / "presentation.md"),
    }
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "Validation failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=round(duration, 2),
            exc_info=True,
        )
        raise


@mcp.tool()
async def quick_validate(
    idea: str,
    hunches: list[str],
) -> dict:
    """
    Quick validation of a business idea with minimal inputs.

    Args:
        idea: One-line description of the business idea
        hunches: 2-4 hypotheses about why this might work
    """
    start_time = time.time()
    log = logger.bind(tool="quick_validate", idea=idea)
    log.info("Starting quick validation")

    try:
        result = await validate_idea(
            title=idea,
            problem_statement=f"Solving a problem related to: {idea}",
            solution_overview=idea,
            hunches=hunches,
        )
        duration = time.time() - start_time
        log.info("Quick validation completed", duration_seconds=round(duration, 2))
        return result
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "Quick validation failed",
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=round(duration, 2),
            exc_info=True,
        )
        raise


@mcp.tool()
def list_recent_reports() -> list[dict]:
    """
    List recent JTBD validation reports.

    Returns a list of recent reports with their paths and summaries.
    """
    start_time = time.time()
    log = logger.bind(tool="list_recent_reports")
    log.info("Listing recent reports")

    try:
        reports_dir = Path(__file__).parent.parent / "reports"
        reports = []

        if not reports_dir.exists():
            log.info("Reports directory does not exist", reports_dir=str(reports_dir))
            return []

        # Walk through date folders
        for date_dir in sorted(reports_dir.iterdir(), reverse=True)[:7]:  # Last 7 days
            if not date_dir.is_dir():
                continue
            for report_dir in sorted(date_dir.iterdir(), reverse=True):
                if not report_dir.is_dir():
                    continue

                # Look for analysis data
                json_file = report_dir / "json" / "analysis_data.json"
                if json_file.exists():
                    try:
                        with open(json_file) as f:
                            data = json.load(f)
                        reports.append({
                            "path": str(report_dir),
                            "date": date_dir.name,
                            "title": data.get("idea", {}).get("title", "Unknown"),
                            "delta": data.get("delta", 0),
                            "akr": data.get("akr", 0),
                        })
                    except Exception as e:
                        log.warning(
                            "Failed to read report",
                            report_path=str(json_file),
                            error=str(e),
                        )

                if len(reports) >= 10:
                    break
            if len(reports) >= 10:
                break

        duration = time.time() - start_time
        log.info(
            "Listed reports",
            report_count=len(reports),
            duration_seconds=round(duration, 2),
        )
        return reports
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "Failed to list reports",
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=round(duration, 2),
            exc_info=True,
        )
        raise


@mcp.tool()
def get_report(report_path: str) -> dict:
    """
    Get the full details of a specific JTBD validation report.

    Args:
        report_path: Path to the report directory (from list_recent_reports)
    """
    start_time = time.time()
    log = logger.bind(tool="get_report", report_path=report_path)
    log.info("Getting report details")

    try:
        report_dir = Path(report_path)

        # Read JSON data
        json_file = report_dir / "json" / "analysis_data.json"
        if not json_file.exists():
            log.warning("Report not found", json_file=str(json_file))
            return {"error": f"Report not found at {report_path}"}

        with open(json_file) as f:
            data = json.load(f)

        # Read markdown report
        md_file = report_dir / "gamma" / "presentation.md"
        markdown = ""
        if md_file.exists():
            markdown = md_file.read_text()

        duration = time.time() - start_time
        log.info("Retrieved report", duration_seconds=round(duration, 2))

        return {
            "data": data,
            "markdown_report": markdown,
            "path": str(report_dir),
        }
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "Failed to get report",
            error=str(e),
            error_type=type(e).__name__,
            duration_seconds=round(duration, 2),
            exc_info=True,
        )
        raise


# Health and readiness endpoints
def add_health_endpoints(app):
    """Add health and ready endpoints to the SSE app."""
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health_check(request):
        """Liveness check endpoint for container orchestration."""
        logger.info("Health check requested")
        return JSONResponse({
            "status": "healthy",
            "version": "0.1.0"
        })

    async def readiness_check(request):
        """Readiness check endpoint - verifies external dependencies."""
        log = logger.bind(endpoint="ready")
        log.info("Readiness check requested")

        status = "ready"
        checks = {}

        # Check if Groq API key is configured
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            checks["groq_api"] = "configured"
        else:
            checks["groq_api"] = "not_configured"
            status = "not_ready"

        log.info("Readiness check completed", status=status, checks=checks)

        return JSONResponse({
            "status": status,
            "checks": checks,
            "version": "0.1.0"
        })

    app.routes.append(Route("/health", health_check))
    app.routes.append(Route("/ready", readiness_check))


if __name__ == "__main__":
    import sys
    if "--http" in sys.argv:
        # Run as HTTP server on port 8001 for Claude Desktop
        import uvicorn
        from middleware.auth import BearerTokenMiddleware

        host = "127.0.0.1"
        port = 8001

        logger.info(
            "Starting JTBD Validator MCP Server",
            mode="http",
            host=host,
            port=port,
            debug=DEBUG,
        )

        app = mcp.sse_app()

        # Add health endpoints first (before middleware wrapping)
        add_health_endpoints(app)

        # Wrap with bearer token authentication middleware
        # Using pure ASGI middleware to avoid breaking SSE streaming
        app = BearerTokenMiddleware(app)

        uvicorn.run(app, host=host, port=port)
    else:
        # Run as stdio for Claude Code
        logger.info(
            "Starting JTBD Validator MCP Server",
            mode="stdio",
            debug=DEBUG,
        )
        mcp.run()
