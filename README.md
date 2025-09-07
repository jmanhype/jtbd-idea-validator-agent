
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

## Technical Architecture

This implementation uses **DSPy** (Declarative Self-improving Language Programs) for structured LLM interactions through **Signatures** and **Modules**.

### DSPy Signatures

Signatures define input/output schemas for LLM tasks:

```python
class DeconstructSig(dspy.Signature):
    """Extract assumptions and classify levels.
    Return JSON list of objects: [{text, level(1..3), confidence, evidence:[]}]"""
    idea: str = dspy.InputField()
    hunches: List[str] = dspy.InputField()
    assumptions_json: str = dspy.OutputField()

class JobsSig(dspy.Signature):
    """Generate 5 distinct JTBD statements with Four Forces each."""
    context: str = dspy.InputField()
    constraints: str = dspy.InputField()
    jobs_json: str = dspy.OutputField()
```

### DSPy Modules

Modules implement business logic with automatic prompt optimization:

- **`Deconstruct`**: Extracts assumptions with confidence scoring
- **`Jobs`**: Generates JTBD statements with Four Forces analysis
- **`Moat`**: Applies Doblin innovation framework + strategic triggers
- **`JudgeScore`**: Evaluates ideas across 5 standardized criteria:
  - Underserved Opportunity
  - Strategic Impact  
  - Market Scale
  - Solution Differentiability
  - Business Model Innovation

### Dual-Judge Arbitration

The system uses two independent judges with tie-breaking for scoring reliability:

```python
USE_DOUBLE_JUDGE = os.getenv("JTBD_DOUBLE_JUDGE", "1") == "1"  # default ON

def judge_with_arbitration(summary: str):
    if USE_DOUBLE_JUDGE:
        score1 = JudgeScore()(summary=summary)
        score2 = JudgeScore()(summary=summary) 
        return merge_scores(score1, score2)  # tie-breaker logic
    return JudgeScore()(summary=summary)
```

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

### Judge Optimization with DSPy

The system supports **compiled judge models** using DSPy's GEPA optimizer (reflective prompt evolution):

```bash
# 1. Add training data to data/judge_train.jsonl
# Format: {"summary": "...", "scorecard": {"criteria":[...], "total": 6.7}}

# 2. Train the judge using GEPA (evolutionary optimizer)
python tools/optimize_judge.py --train data/judge_train.jsonl --out artifacts/judge_compiled.dspy

# 3. Use the compiled judge (automatically loaded at runtime)
export JTBD_JUDGE_COMPILED=artifacts/judge_compiled.dspy
python run_direct.py your_idea.json
```

**GEPA** (Generate, Evolve, Prune, Aggregate) is an evolutionary optimizer that:
- Captures full execution traces of DSPy modules
- Uses reflection to evolve text components (prompts/instructions)
- Allows textual feedback at predictor or system level
- Outperforms reinforcement learning in prompt optimization

From the source implementation in `tools/optimize_judge.py`:

```python
from dspy.teleprompt import GEPA

class NonDecreasingEval(dspy.Eval):
    """Returns 1 if predicted total >= gold total, else 0."""
    def __call__(self, pred, gold, trace=None):
        p = json.loads(pred); g = json.loads(gold)
        return 1.0 if p.get("total",0) >= g.get("total",0) else 0.0

tele = GEPA(metric=NonDecreasingEval(), max_bootstrapped_demos=4)
compiled = tele.compile(prog, trainset=train)
```

The compiled judge replaces the default `dspy.Predict` with an optimized program:

```python
_compiled_judge = None
if JUDGE_COMPILED_PATH and os.path.exists(JUDGE_COMPILED_PATH):
    with open(JUDGE_COMPILED_PATH, "rb") as f:
        _compiled_judge = pickle.load(f)

class JudgeScore(dspy.Module):
    def __init__(self):
        self.p = _compiled_judge or dspy.Predict(JudgeScoreSig)  # fallback
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
