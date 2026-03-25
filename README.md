# jtbd-idea-validator-agent

A DSPy-based agent that validates business ideas using Jobs-to-be-Done (JTBD) methodology. Takes a structured idea description, runs it through 4 analysis stages, and produces scored reports with charts and exportable data.

## What it does

Input: a JSON file describing a business idea (problem, solution, target customer, hunches).
Output: scored analysis with assumption deconstruction, JTBD job statements, competitive moat assessment, and actionable validation plans.

The 4 pipeline stages:

| Stage | DSPy Module | Output |
|-------|-------------|--------|
| Deconstruct | `DeconstructSig` | Assumptions with confidence levels (1-3) |
| Jobs | `JobsSig` | 5 JTBD statements, each with Four Forces (push/pull/anxiety/inertia) |
| Moat | `MoatSig` | Innovation layer analysis (Doblin framework) |
| Judge | `JudgeScoreSig` | Scores across 5 criteria, 0-10 scale |

Scoring criteria: Underserved Opportunity, Strategic Impact, Market Scale, Solution Differentiability, Business Model Innovation.

## Dual-judge arbitration

By default, 2 independent judge passes run and scores are merged with tie-breaking logic. Disable with `JTBD_DOUBLE_JUDGE=0`.

## Quick start

```bash
pip install -e .

export OPENAI_API_KEY=...  # or ANTHROPIC_API_KEY

python run_direct.py examples/rehab_exercise_tracking_rich.json
```

Output goes to a timestamped directory:

```
reports/<timestamp>/
  gamma/presentation.md    # Gamma-ready markdown
  gamma/presentation.html  # Preview
  csv/                     # Structured exports
  json/analysis_data.json  # Raw data
  assets/                  # Radar, waterfall, Four Forces charts
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `JTBD_DSPY_MODEL` | `gpt-4o-mini` | LLM model |
| `JTBD_LLM_TEMPERATURE` | `0.2` | Response randomness |
| `JTBD_DOUBLE_JUDGE` | `1` | Dual-judge arbitration |
| `JTBD_JUDGE_COMPILED` | (none) | Path to GEPA-compiled judge |
| `JTBD_LLM_SEED` | `42` | Reproducibility seed |

## Input format

```json
{
  "idea_id": "urn:idea:example:001",
  "title": "Your idea",
  "hunches": ["Key assumption 1", "Key assumption 2"],
  "problem_statement": "The problem",
  "solution_overview": "The solution",
  "target_customer": {
    "primary": "Main segment",
    "secondary": "Secondary users",
    "demographics": "Details"
  },
  "value_propositions": ["Benefit 1"],
  "competitive_landscape": ["Competitor 1"],
  "revenue_streams": ["Revenue model 1"]
}
```

See `examples/` for complete samples.

## Alternative interfaces

**FastAPI service:**
```bash
uvicorn service.dspy_sidecar:app --port 8088 --reload
# Endpoints: /deconstruct, /jobs, /moat, /judge
# Web console at http://localhost:8088/
```

**Electron app:** Desktop UI in `electron-app/` (React + TypeScript).

**MCP server:** `mcp-server/server.py` for tool integration.

## Judge optimization

The judge module can be compiled using DSPy's GEPA optimizer for improved scoring consistency:

```bash
# Train
python tools/optimize_judge.py --train data/judge_train.jsonl --out artifacts/judge_compiled.dspy --budget medium

# Use
export JTBD_JUDGE_COMPILED=artifacts/judge_compiled.dspy
```

## Project layout

```
contracts/       # Pydantic v1 models (frozen; create v2 for changes)
core/            # Pipeline, scoring, validation planning, export
plugins/         # DSPy LLM interface, chart generation, market data
service/         # FastAPI sidecar, OpenTelemetry, Modaic agent loader
orchestration/   # Prefect flows
mcp-server/      # MCP tool server
electron-app/    # Desktop UI
examples/        # Sample idea JSONs
data/            # Judge training data
```

## Dependencies

Core: DSPy, Pydantic, FastAPI, Modaic, OpenTelemetry, httpx.
Optional: Prefect (workflow orchestration), sse-starlette (streaming).

## Limitations

- Scoring quality depends on the backing LLM. GPT-4o-mini is adequate for structure but less reliable on nuanced market analysis.
- Dual-judge arbitration doubles LLM cost per run.
- GEPA-compiled judges require training data that may not generalize across idea domains.
- Chart generation uses QuickChart (external service).
- The Modaic hub integration requires a separate account and token.
- No automated tests are included in the repository.

## License

Not specified.
