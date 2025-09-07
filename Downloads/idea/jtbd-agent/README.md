# JTBD Idea Validator — DSPy Edition (Real Deal)

This is a **DSPy-powered** implementation (no stubs). It uses DSPy `Signatures`/`Modules` for
- Assumption deconstruction & level classification
- JTBD generation (5 distinct jobs with Four Forces)
- Moat layering (Doblin + timing/ops/customer/value)
- Judge/score (5 criteria with rationales)

The pipeline is orchestrated with Prefect; output is a **Gamma-ready Markdown** plus chart images.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# Configure an LLM in DSPy via env vars. Examples:
export OPENAI_API_KEY=...            # if using OpenAI client in DSPy
export ANTHROPIC_API_KEY=...         # if using Anthropic
# Run
python -m jtbd_agent_dspy.cli run fixtures/idea.json --out out/gamma.md --assets out/assets
```

> DSPy model selection: edit `plugins/llm_dspy.py` → `configure_lm()`.
Set your preferred model (e.g., `"gpt-4o-mini"` or `"claude-3-5-sonnet-20240620"`) and temperature/seed.

## Optional: HTTP sidecar for stage endpoints

You can run a FastAPI server that exposes `/deconstruct`, `/jobs`, `/moat`, `/judge`:
```bash
uvicorn jtbd_agent_dspy.service.dspy_sidecar:app --port 8088 --reload
```

Then point a different orchestrator to these endpoints if desired.

## What gets produced
- `out/gamma.md` — paste/import into Gamma
- `out/assets/{radar.png,waterfall.png,forces.png}`

## Determinism Guardrails
- Temperature defaults to 0.2, fixed seeds where supported.
- Two-judge pattern is available in `plugins/llm_dspy.py` (toggle `USE_DOUBLE_JUDGE`).

## Contracts Are Frozen (v1)
See `contracts/*_v1.py`. Keep formats stable; evolve via `v2` rather than editing `v1`.


---

## Two-judge arbitration (default ON)

Set `JTBD_DOUBLE_JUDGE=0` to disable. By default it's ON and merges two independent judgments
with a simple tie-breaker.

## GEPA optimizer for Judge

Train a compiled Judge and use it at runtime:

```bash
# 1) (optional) add more training rows to data/judge_train.jsonl (JSONL format described below)
python -m jtbd_agent_dspy.tools.optimize_judge --train data/judge_train.jsonl --out artifacts/judge_compiled.dspy

# 2) point runtime to the compiled program
export JTBD_JUDGE_COMPILED=artifacts/judge_compiled.dspy

# 3) run the report as usual
python -m jtbd_agent_dspy.cli run jtbd_agent_dspy/fixtures/idea.json --out out/gamma.md --assets out/assets
```

**Training JSONL format** (one object per line):
```json
{"summary": "...", "scorecard": {"criteria":[{"name":"Underserved Opportunity","score":7.0,"rationale":"..."}, ...], "total": 6.7}}
```
