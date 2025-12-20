# Market Reality Plugin

Comprehensive market validation module for the JTBD Idea Validator Agent.

## Features

1. **Competitor Search via Exa API** - Validates real market competitors
2. **Platform Risk Detection** - Identifies dependencies on major tech platforms
3. **Regulatory Scanning** - Detects applicable compliance requirements

## Installation

The module requires `httpx` and `pydantic`, which are already in the project dependencies.

## Configuration

### Environment Variables

```bash
export EXA_API_KEY="your-exa-api-key-here"
```

Get your Exa API key from: https://exa.ai

## Usage

### 1. Competitor Search (Async)

Search for real competitors using Exa's web search API.

```python
import asyncio
from plugins.market_reality import search_competitors

async def main():
    result = await search_competitors(
        idea_title="AI-powered code review tool for teams",
        industry="developer tools"
    )

    print(f"Found {result['total_results']} competitors")

    for competitor in result['validated_competitors']:
        print(f"\n{competitor['title']}")
        print(f"URL: {competitor['url']}")
        print(f"Snippet: {competitor['snippet']}")

asyncio.run(main())
```

**Returns:**
```python
{
    "validated_competitors": [
        {
            "title": "CompetitorName",
            "url": "https://example.com",
            "snippet": "Brief description..."
        },
        # ... more competitors
    ],
    "search_queries_used": [
        "AI-powered code review existing solutions 2025",
        "developer tools AI tools competitors",
        # ... more queries
    ],
    "sources": ["https://...", "https://..."],
    "total_results": 15
}
```

### 2. Platform Risk Detection (Sync)

Detect dependencies on major tech platforms and assess risk level.

```python
from plugins.market_reality import detect_platform_risk

result = detect_platform_risk(
    idea_title="Claude Dashboard Pro",
    solution_overview="A dashboard built on Anthropic's Claude API for managing AI workflows"
)

print(f"Risk Level: {result['risk_level']}")  # HIGH, MEDIUM, or LOW
print(f"Platforms: {result['platform_dependencies']}")
print(f"Warnings: {result['warnings']}")
```

**Returns:**
```python
{
    "platform_dependencies": ["anthropic"],
    "risk_level": "HIGH",
    "warnings": [
        "Building on Anthropic platform - they may ship competing features",
        "Dependent on Claude API - pricing and terms can change",
        "Platform risk: Anthropic controls your core functionality"
    ]
}
```

### 3. Regulatory Scanning (Sync)

Identify applicable regulatory requirements based on industry and regions.

```python
from plugins.market_reality import scan_regulatory_requirements

result = scan_regulatory_requirements(
    idea_title="Health tracking app for patients",
    industry="healthcare",
    target_regions=["US", "EU"]
)

print(f"Industry: {result['detected_industry']}")
for reg in result['regulations']:
    print(f"\n{reg['name']} - {reg['full_name']}")
    print(f"Jurisdiction: {reg['jurisdiction']}")
```

## Data Models

### CompetitorSearchResult

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class CompetitorSearchResult(BaseModel):
    validated_competitors: List[Dict[str, str]]  # title, url, snippet
    search_queries_used: List[str]               # queries executed
    sources: List[str]                           # unique URLs
    total_results: int                           # total count
```

## Architecture

### Async Design
- `search_competitors()` is async-compatible for integration with async workflows
- Uses `httpx.AsyncClient` for non-blocking HTTP requests
- Concurrent query execution for faster results

### Error Handling
- Graceful degradation: continues with other queries if one fails
- Validates `EXA_API_KEY` environment variable before execution
- Returns empty results with error logging on API failures

### Performance
- Timeout: 30 seconds per request
- Default: 5 results per query, 4 queries total (max 20 competitors)
- Text snippets limited to 200 characters for efficiency

## Testing

Run the test suite:

```bash
python plugins/market_reality.py
```

With Exa API key set:
```bash
export EXA_API_KEY="your-key"
python plugins/market_reality.py
```

## Integration Examples

### Using in DSPy Module

```python
import dspy
from plugins.market_reality import search_competitors

class MarketValidator(dspy.Module):
    async def forward(self, idea_title: str, industry: str):
        # Validate competitors exist
        competitors = await search_competitors(idea_title, industry)

        # Use in your validation logic
        if competitors['total_results'] > 10:
            return "High competition - validate unique value proposition"
        elif competitors['total_results'] == 0:
            return "No competitors found - validate market demand"
        else:
            return "Moderate competition - validate differentiation"
```

### Using in FastAPI Endpoint

```python
from fastapi import FastAPI, HTTPException
from plugins.market_reality import search_competitors

app = FastAPI()

@app.post("/validate/competitors")
async def validate_competitors(idea: str, industry: str):
    try:
        result = await search_competitors(idea, industry)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## API Reference

### `search_competitors(idea_title: str, industry: str) -> dict`
- **Type**: Async function
- **Parameters**:
  - `idea_title`: The idea/product name or category
  - `industry`: Industry domain (e.g., "SaaS", "AI tools")
- **Returns**: Dictionary matching `CompetitorSearchResult` schema
- **Raises**:
  - `ValueError` if `EXA_API_KEY` not set
  - `httpx.HTTPError` on API failures

### `detect_platform_risk(idea_title: str, solution_overview: str) -> dict`
- **Type**: Sync function
- **Parameters**:
  - `idea_title`: Business idea name
  - `solution_overview`: Description of how solution works
- **Returns**:
  ```python
  {
      "platform_dependencies": List[str],
      "risk_level": str,  # "HIGH", "MEDIUM", "LOW"
      "warnings": List[str]
  }
  ```

### `scan_regulatory_requirements(idea_title: str, industry: str, target_regions: Optional[List[str]]) -> dict`
- **Type**: Sync function
- **Parameters**:
  - `idea_title`: Business idea name
  - `industry`: Industry category
  - `target_regions`: Geographic regions (optional)
- **Returns**:
  ```python
  {
      "detected_industry": str,
      "regulations": List[dict],
      "compliance_notes": List[str]
  }
  ```

## Limitations

- Competitor search requires Exa API key (paid service)
- Platform risk detection uses keyword matching (not ML-based)
- Regulatory scanning is guidance only, not legal advice
- Rate limits apply based on Exa API tier

## Future Enhancements

- [ ] Cache competitor search results for common queries
- [ ] Add confidence scores to risk detection
- [ ] Support more platforms (AWS, Meta, etc.)
- [ ] Integrate with legal compliance databases
- [ ] Add market size estimation from search results
