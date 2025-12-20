"""
Market Reality Plugin - Platform Risk Detection & Competitor Validation

Analyzes business ideas for platform dependencies, associated risks, and validates
competitors using web search via Exa API.
"""

import os
from typing import Dict, List, Tuple, Optional

import httpx
from pydantic import BaseModel, Field


# ============================================================================
# Data Models
# ============================================================================


class CompetitorSearchResult(BaseModel):
    """Response model for competitor search validation."""

    validated_competitors: List[Dict[str, str]] = Field(
        description="List of competitor companies/products found with title, URL, and snippet"
    )
    search_queries_used: List[str] = Field(
        description="Search queries executed to find competitors"
    )
    sources: List[str] = Field(
        description="URLs of competitor websites found"
    )
    total_results: int = Field(
        description="Total number of competitors found"
    )


# ============================================================================
# Competitor Search via Exa API
# ============================================================================


async def search_competitors(idea_title: str, industry: str) -> Dict:
    """
    Search for real competitors using Exa API web search.

    Executes multiple search queries to validate if competitors exist in the market.
    Requires EXA_API_KEY environment variable to be set.

    Args:
        idea_title: The title/category of the idea to search for
        industry: The industry or domain (e.g., "SaaS", "AI tools", "healthcare")

    Returns:
        dict: CompetitorSearchResult as dictionary with validated competitors,
              search queries used, and source URLs

    Raises:
        ValueError: If EXA_API_KEY is not set in environment
        httpx.HTTPError: If API request fails
    """
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError(
            "EXA_API_KEY environment variable is required for competitor search"
        )

    # Extract idea category from title (simple heuristic)
    idea_category = idea_title.lower().split("for")[0].strip()

    # Build search queries
    search_queries = [
        f"{idea_category} existing solutions 2025",
        f"{industry} AI tools competitors",
        f"{idea_category} alternatives comparison",
        f"best {idea_category} platforms {industry}",
    ]

    all_competitors = []
    all_sources = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for query in search_queries:
            try:
                response = await client.post(
                    "https://api.exa.ai/search",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                    },
                    json={
                        "query": query,
                        "numResults": 5,
                        "useAutoprompt": True,
                        "type": "auto",
                        "contents": {
                            "text": {"maxCharacters": 300}
                        }
                    }
                )
                response.raise_for_status()
                data = response.json()

                # Extract results
                for result in data.get("results", []):
                    competitor = {
                        "title": result.get("title", "Unknown"),
                        "url": result.get("url", ""),
                        "snippet": result.get("text", "")[:200],
                    }
                    all_competitors.append(competitor)
                    all_sources.append(result.get("url", ""))

            except httpx.HTTPError as e:
                # Log error but continue with other queries
                print(f"Error searching with query '{query}': {e}")
                continue

    # Deduplicate sources
    unique_sources = list(set(all_sources))

    result = CompetitorSearchResult(
        validated_competitors=all_competitors,
        search_queries_used=search_queries,
        sources=unique_sources,
        total_results=len(all_competitors)
    )

    return result.model_dump()


# ============================================================================
# Platform Risk Detection
# ============================================================================


def detect_platform_risk(idea_title: str, solution_overview: str) -> dict:
    """
    Detect platform dependencies and assess associated risks.

    Analyzes the idea for dependencies on major tech platforms and classifies
    the risk level based on how tightly coupled the solution is to those platforms.

    Args:
        idea_title: The title/name of the business idea
        solution_overview: Description of the solution and how it works

    Returns:
        dict: {
            "platform_dependencies": List[str] - Detected platform names,
            "risk_level": str - "HIGH", "MEDIUM", or "LOW",
            "warnings": List[str] - Specific warnings about each dependency
        }
    """
    # Platform detection configuration
    PLATFORMS = {
        "anthropic": ["anthropic", "claude"],
        "openai": ["openai", "gpt", "chatgpt"],
        "google": ["google", "gemini", "bard"],
        "microsoft": ["microsoft", "azure", "copilot"],
    }

    # Combine text for analysis
    combined_text = f"{idea_title} {solution_overview}".lower()

    # Detect platform dependencies
    detected_platforms = []
    for platform_name, keywords in PLATFORMS.items():
        if any(keyword in combined_text for keyword in keywords):
            detected_platforms.append(platform_name)

    # Classify risk level based on usage patterns
    risk_level, warnings = _classify_risk_level(
        combined_text,
        detected_platforms
    )

    return {
        "platform_dependencies": detected_platforms,
        "risk_level": risk_level,
        "warnings": warnings
    }


def _classify_risk_level(text: str, platforms: List[str]) -> Tuple[str, List[str]]:
    """
    Classify the risk level based on platform usage patterns.

    Args:
        text: Combined text to analyze
        platforms: List of detected platform names

    Returns:
        Tuple of (risk_level, warnings)
    """
    if not platforms:
        return "LOW", []

    warnings = []

    # HIGH risk indicators - building ON the platform
    high_risk_patterns = [
        "built on",
        "powered by",
        "using",
        "wrapper for",
        "interface for",
        "frontend for",
        "dashboard for",
        "built with",
        "leveraging",
        "based on",
        "runs on",
    ]

    # MEDIUM risk indicators - integrating WITH the platform
    medium_risk_patterns = [
        "integrates with",
        "connects to",
        "api",
        "integration",
        "compatible with",
    ]

    # Check for HIGH risk patterns
    has_high_risk = any(pattern in text for pattern in high_risk_patterns)
    has_medium_risk = any(pattern in text for pattern in medium_risk_patterns)

    # Generate platform-specific warnings
    platform_warnings = {
        "anthropic": [
            "Building on Anthropic platform - they may ship competing features",
            "Dependent on Claude API - pricing and terms can change",
            "Platform risk: Anthropic controls your core functionality"
        ],
        "openai": [
            "Building on OpenAI platform - they may ship competing features",
            "Dependent on OpenAI API - pricing and terms can change",
            "Platform risk: OpenAI controls your core functionality"
        ],
        "google": [
            "Building on Google platform - they may ship competing features",
            "Dependent on Gemini API - pricing and terms can change",
            "Platform risk: Google has history of deprecating products"
        ],
        "microsoft": [
            "Building on Microsoft platform - they may ship competing features",
            "Dependent on Azure/Copilot - pricing and terms can change",
            "Platform risk: Microsoft integrates features into core products"
        ],
    }

    # Determine risk level and collect warnings
    if has_high_risk:
        risk_level = "HIGH"
        for platform in platforms:
            if platform in platform_warnings:
                # Add all warnings for HIGH risk
                warnings.extend(platform_warnings[platform])
    elif has_medium_risk:
        risk_level = "MEDIUM"
        for platform in platforms:
            if platform in platform_warnings:
                # Add API-specific warning for MEDIUM risk
                warnings.append(platform_warnings[platform][1])
    else:
        # Detected platform mention but no clear dependency pattern
        risk_level = "MEDIUM"
        for platform in platforms:
            warnings.append(
                f"Mentions {platform.title()} - clarify dependency level"
            )

    return risk_level, warnings


def scan_regulatory_requirements(
    idea_title: str,
    industry: str,
    target_regions: Optional[list[str]] = None
) -> dict:
    """
    Scan for applicable regulatory requirements based on industry and target regions.

    Uses keyword matching to identify relevant regulations without external API calls.

    Args:
        idea_title: Title or description of the business idea
        industry: Industry category or description
        target_regions: Optional list of target geographic regions (e.g., ["US", "EU", "California"])

    Returns:
        dict: {
            "detected_industry": str,
            "regulations": list[dict],
            "compliance_notes": list[str]
        }
    """
    if target_regions is None:
        target_regions = []

    # Normalize inputs for keyword matching
    combined_text = f"{idea_title} {industry}".lower()
    regions_text = " ".join(target_regions).lower() if target_regions else ""

    regulations = []
    compliance_notes = []
    detected_industry = "General"

    # Healthcare & Medical
    if any(keyword in combined_text for keyword in ["healthcare", "medical", "patient", "health", "clinic", "hospital"]):
        detected_industry = "Healthcare & Medical"
        regulations.append({
            "name": "HIPAA",
            "full_name": "Health Insurance Portability and Accountability Act",
            "jurisdiction": "United States",
            "description": "Protects sensitive patient health information from disclosure",
            "key_requirements": [
                "Implement physical, network, and process security measures",
                "Ensure confidentiality, integrity, and availability of protected health information",
                "Employee training and access controls"
            ]
        })
        regulations.append({
            "name": "HITECH",
            "full_name": "Health Information Technology for Economic and Clinical Health Act",
            "jurisdiction": "United States",
            "description": "Strengthens HIPAA privacy and security protections",
            "key_requirements": [
                "Breach notification requirements",
                "Enhanced penalties for HIPAA violations",
                "Business associate agreement requirements"
            ]
        })
        compliance_notes.append("Healthcare applications require strict data protection and privacy controls")
        compliance_notes.append("Consider hiring a HIPAA compliance officer or consultant")

        # Check for medical device indicators
        if any(keyword in combined_text for keyword in ["device", "diagnostic", "monitor", "sensor", "wearable"]):
            regulations.append({
                "name": "FDA",
                "full_name": "Food and Drug Administration Regulations",
                "jurisdiction": "United States",
                "description": "Regulates medical devices for safety and effectiveness",
                "key_requirements": [
                    "Device classification (Class I, II, or III)",
                    "510(k) premarket notification or PMA approval",
                    "Quality System Regulation (QSR) compliance"
                ]
            })
            compliance_notes.append("Medical devices require FDA clearance or approval before market entry")

    # Finance & Banking
    if any(keyword in combined_text for keyword in ["finance", "banking", "payment", "transaction", "fintech", "money", "loan", "credit"]):
        detected_industry = "Financial Services"
        regulations.append({
            "name": "SOX",
            "full_name": "Sarbanes-Oxley Act",
            "jurisdiction": "United States",
            "description": "Mandates financial record accuracy and internal controls for public companies",
            "key_requirements": [
                "Accurate financial reporting and disclosures",
                "Internal control assessment and documentation",
                "Executive certification of financial statements"
            ]
        })
        regulations.append({
            "name": "PCI-DSS",
            "full_name": "Payment Card Industry Data Security Standard",
            "jurisdiction": "Global",
            "description": "Security standards for organizations that handle credit card information",
            "key_requirements": [
                "Maintain secure network and systems",
                "Protect cardholder data with encryption",
                "Regular security testing and monitoring"
            ]
        })
        regulations.append({
            "name": "GLBA",
            "full_name": "Gramm-Leach-Bliley Act",
            "jurisdiction": "United States",
            "description": "Requires financial institutions to protect consumer information",
            "key_requirements": [
                "Privacy notices to customers",
                "Information security program",
                "Pretexting protections"
            ]
        })
        compliance_notes.append("Financial services require robust security controls and regular audits")
        compliance_notes.append("Consider using payment processors that handle PCI-DSS compliance")

    # Insurance
    if any(keyword in combined_text for keyword in ["insurance", "insurer", "policy", "underwriting", "claims"]):
        detected_industry = "Insurance"
        regulations.append({
            "name": "State Insurance Regulations",
            "full_name": "State-Level Insurance Regulations",
            "jurisdiction": "United States (State-Level)",
            "description": "Each state regulates insurance products and providers",
            "key_requirements": [
                "State-specific licensing requirements",
                "Rate filing and approval processes",
                "Solvency and reserve requirements"
            ]
        })
        regulations.append({
            "name": "NAIC",
            "full_name": "National Association of Insurance Commissioners Standards",
            "jurisdiction": "United States",
            "description": "Model regulations and standards adopted by states",
            "key_requirements": [
                "Market conduct standards",
                "Consumer protection requirements",
                "Data security model law compliance"
            ]
        })
        compliance_notes.append("Insurance is heavily regulated at the state level - requires licenses in each operating state")
        compliance_notes.append("Work with insurance compliance attorneys familiar with target states")

    # Education
    if any(keyword in combined_text for keyword in ["education", "student", "school", "university", "learning", "educational"]):
        detected_industry = "Education"
        regulations.append({
            "name": "FERPA",
            "full_name": "Family Educational Rights and Privacy Act",
            "jurisdiction": "United States",
            "description": "Protects privacy of student education records",
            "key_requirements": [
                "Parental consent for disclosure of records",
                "Student rights to access and request amendments",
                "Institutional policies for record protection"
            ]
        })
        compliance_notes.append("Educational applications handling student data require FERPA compliance")
        compliance_notes.append("Implement strict access controls and audit logging for student records")

    # Children-Focused
    if any(keyword in combined_text for keyword in ["children", "kids", "child", "minor", "teen", "youth", "under 13"]):
        if detected_industry == "General":
            detected_industry = "Children & Youth Services"
        regulations.append({
            "name": "COPPA",
            "full_name": "Children's Online Privacy Protection Act",
            "jurisdiction": "United States",
            "description": "Protects privacy of children under 13 online",
            "key_requirements": [
                "Parental consent before collecting personal information",
                "Clear privacy policy disclosure",
                "Data security and retention limitations"
            ]
        })
        compliance_notes.append("Services for children under 13 require verifiable parental consent")
        compliance_notes.append("Consider age-gating and parental control features")

    # Geographic/Regional Regulations

    # GDPR - EU
    if any(keyword in regions_text for keyword in ["eu", "europe", "european", "gdpr"]) or \
       any(keyword in combined_text for keyword in ["gdpr", "european union"]):
        regulations.append({
            "name": "GDPR",
            "full_name": "General Data Protection Regulation",
            "jurisdiction": "European Union",
            "description": "Comprehensive data protection and privacy regulation",
            "key_requirements": [
                "Lawful basis for data processing",
                "Data subject rights (access, erasure, portability)",
                "Privacy by design and default",
                "Data breach notification within 72 hours",
                "Data Protection Impact Assessments for high-risk processing"
            ]
        })
        compliance_notes.append("GDPR applies to any business processing EU resident data, regardless of location")
        compliance_notes.append("Implement consent management, data subject access requests, and right to erasure")

    # CCPA - California
    if any(keyword in regions_text for keyword in ["california", "ca", "ccpa"]) or \
       any(keyword in combined_text for keyword in ["ccpa", "california"]):
        regulations.append({
            "name": "CCPA",
            "full_name": "California Consumer Privacy Act",
            "jurisdiction": "California, United States",
            "description": "California consumer data privacy rights",
            "key_requirements": [
                "Consumer right to know what data is collected",
                "Consumer right to delete personal information",
                "Consumer right to opt-out of data sales",
                "Privacy policy disclosures"
            ]
        })
        compliance_notes.append("CCPA applies to businesses meeting revenue or data processing thresholds in California")
        compliance_notes.append("Implement 'Do Not Sell My Personal Information' mechanisms")

    # Government/Federal
    if any(keyword in combined_text for keyword in ["government", "federal", "gov", "public sector", "agency"]):
        if detected_industry == "General":
            detected_industry = "Government & Public Sector"
        regulations.append({
            "name": "FedRAMP",
            "full_name": "Federal Risk and Authorization Management Program",
            "jurisdiction": "United States Federal Government",
            "description": "Standardized approach to security assessment for cloud services",
            "key_requirements": [
                "Security controls based on NIST standards",
                "Third-party assessment organization review",
                "Continuous monitoring requirements"
            ]
        })
        regulations.append({
            "name": "FISMA",
            "full_name": "Federal Information Security Management Act",
            "jurisdiction": "United States Federal Government",
            "description": "Framework for securing government information systems",
            "key_requirements": [
                "Risk-based approach to security",
                "Annual security reviews",
                "Incident response procedures"
            ]
        })
        compliance_notes.append("Federal government contracts require FedRAMP authorization for cloud services")
        compliance_notes.append("Plan for 6-12 month authorization timeline and significant compliance investment")

    # Add general notes if regulations were detected
    if regulations:
        compliance_notes.insert(0, f"Detected {len(regulations)} potentially applicable regulation(s)")
        compliance_notes.append("Consult with legal counsel specializing in your industry for detailed compliance guidance")
        compliance_notes.append("Factor compliance costs and timelines into business planning and funding requirements")
    else:
        compliance_notes.append("No specific regulatory frameworks detected based on provided information")
        compliance_notes.append("General business regulations (tax, employment, contracts) still apply")
        compliance_notes.append("Consult legal counsel to identify any industry-specific requirements")

    return {
        "detected_industry": detected_industry,
        "regulations": regulations,
        "compliance_notes": compliance_notes
    }


# ============================================================================
# Example Usage & Testing
# ============================================================================


if __name__ == "__main__":
    import asyncio

    print("=" * 70)
    print("PLATFORM RISK DETECTION TESTS")
    print("=" * 70)

    # Test case 1: HIGH risk - building on platform
    result1 = detect_platform_risk(
        "Claude Code Assistant",
        "A coding assistant built on Anthropic's Claude API that helps developers write better code"
    )
    print("\nTest 1 - HIGH risk:")
    print(f"  Platforms: {result1['platform_dependencies']}")
    print(f"  Risk: {result1['risk_level']}")
    print(f"  Warnings: {result1['warnings']}")

    # Test case 2: MEDIUM risk - API integration
    result2 = detect_platform_risk(
        "Multi-Model Chat",
        "A chat application that integrates with OpenAI and Anthropic APIs for comparison"
    )
    print("\nTest 2 - MEDIUM risk:")
    print(f"  Platforms: {result2['platform_dependencies']}")
    print(f"  Risk: {result2['risk_level']}")
    print(f"  Warnings: {result2['warnings']}")

    # Test case 3: LOW risk - no dependencies
    result3 = detect_platform_risk(
        "Local Task Manager",
        "A simple task management app that runs entirely offline"
    )
    print("\nTest 3 - LOW risk:")
    print(f"  Platforms: {result3['platform_dependencies']}")
    print(f"  Risk: {result3['risk_level']}")
    print(f"  Warnings: {result3['warnings']}")

    # Test case 4: Competitor search (async)
    print("\n" + "=" * 70)
    print("COMPETITOR SEARCH TEST")
    print("=" * 70)

    async def test_competitor_search():
        if not os.getenv("EXA_API_KEY"):
            print("\nSkipping competitor search test - EXA_API_KEY not set")
            print("Set EXA_API_KEY environment variable to enable this test")
            return

        try:
            result = await search_competitors(
                idea_title="AI-powered code review tool",
                industry="developer tools"
            )
            print(f"\nFound {result['total_results']} competitors:")
            for i, comp in enumerate(result['validated_competitors'][:3], 1):
                print(f"\n{i}. {comp['title']}")
                print(f"   URL: {comp['url']}")
                print(f"   Snippet: {comp['snippet'][:100]}...")
            print(f"\nSearch queries used: {result['search_queries_used']}")
        except Exception as e:
            print(f"\nCompetitor search error: {e}")

    asyncio.run(test_competitor_search())
