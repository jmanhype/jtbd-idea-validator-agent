"""
Example integration of Market Reality Plugin with JTBD Agent

This demonstrates how to use the market_reality module in the context
of the JTBD idea validation workflow.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plugins.market_reality import (
    search_competitors,
    detect_platform_risk,
    scan_regulatory_requirements,
    CompetitorSearchResult
)


async def validate_market_reality(
    idea_title: str,
    solution_overview: str,
    industry: str,
    target_regions: list[str] = None
) -> Dict[str, Any]:
    """
    Complete market reality validation combining all checks.

    Args:
        idea_title: The business idea title
        solution_overview: Description of how the solution works
        industry: Industry category (e.g., "SaaS", "healthcare")
        target_regions: Geographic regions to target (optional)

    Returns:
        dict: Complete market reality assessment with scores and warnings
    """
    results = {
        "idea_title": idea_title,
        "industry": industry,
        "validation_passed": True,
        "warnings": [],
        "scores": {},
    }

    # 1. Platform Risk Assessment
    print("Checking platform dependencies...")
    platform_risk = detect_platform_risk(idea_title, solution_overview)

    results["platform_risk"] = platform_risk

    # Score platform risk (0-100, lower is better)
    risk_scores = {"LOW": 100, "MEDIUM": 50, "HIGH": 20}
    results["scores"]["platform_independence"] = risk_scores.get(
        platform_risk["risk_level"], 50
    )

    if platform_risk["risk_level"] == "HIGH":
        results["warnings"].extend(platform_risk["warnings"])

    # 2. Competitor Validation
    print("Searching for competitors...")
    try:
        competitors = await search_competitors(idea_title, industry)
        results["competitors"] = competitors

        # Score based on competition level
        num_competitors = competitors["total_results"]
        if num_competitors == 0:
            results["scores"]["market_validation"] = 30
            results["warnings"].append(
                "No competitors found - market demand uncertain"
            )
        elif num_competitors < 5:
            results["scores"]["market_validation"] = 70
        elif num_competitors < 15:
            results["scores"]["market_validation"] = 85
        else:
            results["scores"]["market_validation"] = 60
            results["warnings"].append(
                f"High competition: {num_competitors} competitors found"
            )

    except ValueError as e:
        print(f"Competitor search skipped: {e}")
        results["competitors"] = {"error": str(e)}
        results["scores"]["market_validation"] = 50  # neutral score

    # 3. Regulatory Requirements
    print("Scanning regulatory requirements...")
    regulatory = scan_regulatory_requirements(
        idea_title, industry, target_regions or []
    )

    results["regulatory"] = regulatory

    # Score based on regulatory complexity
    num_regulations = len(regulatory["regulations"])
    if num_regulations == 0:
        results["scores"]["regulatory_complexity"] = 100
    elif num_regulations < 3:
        results["scores"]["regulatory_complexity"] = 70
    else:
        results["scores"]["regulatory_complexity"] = 40
        results["warnings"].append(
            f"High regulatory burden: {num_regulations} regulations detected"
        )

    # 4. Calculate Overall Market Reality Score
    scores = results["scores"]
    overall_score = sum(scores.values()) / len(scores) if scores else 50

    results["overall_market_reality_score"] = round(overall_score, 1)

    # 5. Final validation decision
    if overall_score < 40:
        results["validation_passed"] = False
        results["warnings"].insert(
            0, "Market reality score below threshold - high risk factors detected"
        )

    return results


async def main():
    """Example usage of market reality validation."""

    print("=" * 70)
    print("MARKET REALITY VALIDATION - EXAMPLE 1")
    print("=" * 70)

    # Example 1: High-risk idea (platform dependency, healthcare)
    result1 = await validate_market_reality(
        idea_title="AI Medical Diagnosis Assistant powered by Claude",
        solution_overview="A medical diagnosis tool built on Anthropic's Claude API that analyzes patient symptoms",
        industry="healthcare",
        target_regions=["US", "EU"]
    )

    print(f"\nIdea: {result1['idea_title']}")
    print(f"Overall Score: {result1['overall_market_reality_score']}/100")
    print(f"Validation Passed: {result1['validation_passed']}")
    print(f"\nScores Breakdown:")
    for metric, score in result1['scores'].items():
        print(f"  - {metric}: {score}/100")

    if result1['warnings']:
        print(f"\nWarnings ({len(result1['warnings'])}):")
        for warning in result1['warnings']:
            print(f"  - {warning}")

    print("\n" + "=" * 70)
    print("MARKET REALITY VALIDATION - EXAMPLE 2")
    print("=" * 70)

    # Example 2: Lower-risk idea (no platform dependency)
    result2 = await validate_market_reality(
        idea_title="Local-first task management app",
        solution_overview="An offline-first task manager with local encryption and optional cloud sync",
        industry="productivity software",
        target_regions=[]
    )

    print(f"\nIdea: {result2['idea_title']}")
    print(f"Overall Score: {result2['overall_market_reality_score']}/100")
    print(f"Validation Passed: {result2['validation_passed']}")
    print(f"\nScores Breakdown:")
    for metric, score in result2['scores'].items():
        print(f"  - {metric}: {score}/100")

    if result2['warnings']:
        print(f"\nWarnings ({len(result2['warnings'])}):")
        for warning in result2['warnings']:
            print(f"  - {warning}")
    else:
        print("\nNo warnings - clear path forward!")


if __name__ == "__main__":
    asyncio.run(main())
