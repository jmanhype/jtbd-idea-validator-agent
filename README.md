
# JTBD Idea Validator

A **Jobs to be Done (JTBD)** analysis agent powered by DSPy that validates business ideas through comprehensive framework-based evaluation.

## What it does

This tool performs systematic business idea validation using JTBD methodology:

- **Assumption Deconstruction**: Extract and classify core business assumptions (1-3 levels)
- **JTBD Analysis**: Generate 5 distinct job statements with Four Forces (push/pull/anxiety/inertia)
- **Moat Analysis**: Assess competitive advantages using innovation layers
- **Scoring & Judgment**: Evaluate ideas across 5 criteria with detailed rationales
- **Validation Planning**: Create actionable plans for assumption testing

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .

# Configure LLM (required)
export OPENAI_API_KEY=...            # for OpenAI models
export ANTHROPIC_API_KEY=...         # for Claude models

# Run analysis on example
python run_direct.py examples/rehab_exercise_tracking_rich.json

# Or specify custom output location
python run_direct.py examples/insurance_photo_ai.json --output custom_reports/
```

## Output Files

The tool generates organized reports in timestamped directories:

- **Gamma Presentations**: `gamma/presentation.md` (Gamma-ready) + `gamma/presentation.html` (preview)
- **CSV Exports**: `csv/` - Structured data for spreadsheet analysis
- **JSON Data**: `json/analysis_data.json` - Raw analysis data
- **Charts**: `assets/` - Radar charts, waterfall charts, and Four Forces diagrams

## Configuration

**Model Selection**: Edit `plugins/llm_dspy.py` → `configure_lm()` or set `JTBD_DSPY_MODEL`:

```bash
export JTBD_DSPY_MODEL="gpt-4o-mini"              # OpenAI
export JTBD_DSPY_MODEL="claude-3-5-sonnet-20240620"  # Anthropic
```

**Other Options**:

- `JTBD_LLM_TEMPERATURE=0.2` - Response randomness (0.0-1.0)
- `JTBD_DOUBLE_JUDGE=1` - Enable dual-judge arbitration (default: enabled)

## Input Format

Ideas are defined in JSON files with the following structure:

```json
{
  "idea_id": "urn:idea:example:001",
  "title": "Your business idea title",
  "hunches": [
    "Key assumption about the problem",
    "Belief about customer behavior",
    "Market hypothesis"
  ],
  "problem_statement": "Clear description of the problem",
  "solution_overview": "How your idea solves the problem",
  "target_customer": {
    "primary": "Main customer segment",
    "secondary": "Secondary users",
    "demographics": "Age, profession, context"
  },
  "value_propositions": ["Key benefit 1", "Key benefit 2"],
  "competitive_landscape": ["Competitor 1", "Competitor 2"],
  "revenue_streams": ["Revenue model 1", "Revenue model 2"]
}
```

See `examples/` directory for complete examples.

## Alternative Execution Methods

### Direct Python Script

```bash
python run_direct.py your_idea.json
```

### FastAPI Service (Optional)

Run as a service with HTTP endpoints:

```bash
uvicorn service.dspy_sidecar:app --port 8088 --reload
```

Exposes endpoints: `/deconstruct`, `/jobs`, `/moat`, `/judge`

### Prefect Flow (Advanced)

For complex orchestration scenarios using the Prefect workflow engine.

## Advanced Features

### Judge Optimization

Train a custom judge model for improved scoring:

```bash
# 1. Add training data to data/judge_train.jsonl
# Format: {"summary": "...", "scorecard": {"criteria":[...], "total": 6.7}}

# 2. Train the judge
python tools/optimize_judge.py --train data/judge_train.jsonl --out artifacts/judge_compiled.dspy

# 3. Use the compiled judge
export JTBD_JUDGE_COMPILED=artifacts/judge_compiled.dspy
python run_direct.py your_idea.json
```

### Environment Variables

- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` - API keys for LLM providers
- `JTBD_DSPY_MODEL` - Model name (default: "gpt-4o-mini")
- `JTBD_LLM_TEMPERATURE` - Temperature setting (default: 0.2)
- `JTBD_LLM_SEED` - Random seed for reproducibility (default: 42)
- `JTBD_DOUBLE_JUDGE` - Enable dual-judge arbitration (default: 1)
- `JTBD_JUDGE_COMPILED` - Path to compiled judge model

## Project Structure

```
├── contracts/          # Pydantic models (v1 frozen contracts)
├── core/              # Main business logic
│   ├── pipeline.py    # Main analysis pipeline
│   ├── score.py       # Scoring algorithms
│   ├── plan.py        # Validation planning
│   └── export_*.py    # Output formatters
├── plugins/           # External integrations
│   ├── llm_dspy.py    # DSPy LLM interface
│   └── charts_quickchart.py  # Chart generation
├── service/           # FastAPI service
├── orchestration/     # Prefect flows
├── examples/          # Sample business ideas
├── tools/            # Optimization utilities
└── run_direct.py     # Main CLI entry point
```

## Dependencies

- **DSPy**: Language model orchestration framework
- **Pydantic**: Data validation and serialization
- **FastAPI/Uvicorn**: Optional HTTP service
- **Prefect**: Optional workflow orchestration
- **Requests**: HTTP client for external services

## Contract Stability

Data contracts in `contracts/*_v1.py` are frozen. For changes, create new `v2` versions rather than modifying existing contracts to ensure backward compatibility.
