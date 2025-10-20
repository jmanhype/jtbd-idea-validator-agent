# Modaic JTBD Agent Playbook

**Repository**: https://github.com/straughterguthrie/jtbd-idea-validator-agent

Showcase guide for the Modaic-hosted JTBD DSPy agent: learn the architecture, run it locally, and plug in observability, retrievers, and streaming APIs. Matches the style of your other plugin gists with narrative framing and copy/paste workflows.

---

## 🎯 Purpose

- Document how to fetch the Modaic-hosted agent (`straughterguthrie/jtbd-agent`) and run every tool (`deconstruct`, `jobs`, `moat`, `judge`).
- Provide a local runner that reads fixtures, reloads remote revisions, and prints structured JSON for quick eyeballing.
- Show the minimal environment required (Modaic token, agent slug, OpenAI key) and how to swap models when Anthropic credits run out.

---

## 🏗️ Architecture Snapshot

- **Hosted Brain**: Modaic PrecompiledAgent (`straughterguthrie/jtbd-agent`) with retriever hooks + DSPy modules.
- **Local Harness**: `tools/modaic_jtbd_agent_demo.py` → prompts each tool via JSON envelope.
- **Retriever**: Notes-based fallback seeded from `.env` (`RETRIEVER_NOTES`).
- **Observability**: Request-ID propagation + OTLP exporter (optional) from the FastAPI sidecar.

```
 Modaic Hub ─┐
             ├── JTBDDSPyAgent.from_precompiled()  (pull latest revision)
 Local Runner┘        │
                      ├── deconstruct → assumptions[]
                      ├── jobs        → jobs[]
                      ├── moat        → layers[]
                      └── judge       → scorecard{}
```

---

## ⚙️ Prerequisites

- Python 3.10+
- `pip install -e .` (installs `jtbd_agent_dspy` with Modaic + OpenTelemetry extras)
- **Environment variables** (recommend storing in `.env`):
  ```bash
  MODAIC_TOKEN=...                     # personal access token
  MODAIC_AGENT_ID=straughterguthrie/jtbd-agent
  RETRIEVER_KIND=notes
  RETRIEVER_NOTES=$'JTBD primer\nMoat checklist\nValidation plan quickstart'
  JTBD_DSPY_MODEL=gpt-4o-mini          # swap if Anthropic credits are low
  OPENAI_API_KEY=sk-...                # or any provider supported by DSPy
  ```
- Reload env vars before exploring: `set -a && source .env && set +a`

---

## 🚀 Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional: ensure agent reloads from Modaic hub
python - <<'PY'
from service.agent_loader import reload_agent
print("Reload source:", reload_agent())
PY

# Run the agent locally
python tools/modaic_jtbd_agent_demo.py
```

This prints four JSON sections (assumptions, jobs, moat layers, judge scorecard) sourced from `fixtures/idea.json`. Edit `DEMO_IDEA_PATH` or pass a custom payload to test different ideas.

---

## 🧪 Demo Script (copy into gist)

> File name suggestion: `modaic_jtbd_agent_demo.py`

```python
#!/usr/bin/env python
"""Modaic JTBD agent quickstart script."""

from __future__ import annotations

import json
import os
from pathlib import Path

from service.modaic_agent import JTBDDSPyAgent


def load_fixture() -> dict:
    idea_path = Path(os.getenv("DEMO_IDEA_PATH", Path(__file__).resolve().parent / "fixtures" / "idea.json"))
    try:
        return json.loads(idea_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "title": "AI inbox triage for clinicians",
            "hunches": ["too many portal messages", "after-hours burden"],
            "context": {"audience": "safety-net hospital"},
            "constraints": [],
        }


def main() -> None:
    agent_id = os.getenv("MODAIC_AGENT_ID", "straughterguthrie/jtbd-agent")
    agent = JTBDDSPyAgent.from_precompiled(agent_id)

    sample = load_fixture()
    payloads = {
        "deconstruct": {"idea": sample.get("title", "AI assistant"), "hunches": sample.get("hunches", [])},
        "jobs": {
            "context": {
                "idea": sample.get("title", ""),
                "audience": sample.get("context", {}).get("audience", "mixed"),
                "hunches": sample.get("hunches", []),
            },
            "constraints": sample.get("constraints", []),
        },
        "moat": {
            "concept": f"{sample.get('title', '')} moat strategy",
            "triggers": "retention, brand, integrations",
        },
        "judge": {
            "summary": (
                f"Assess {sample.get('title', '')} for {sample.get('context', {}).get('audience', 'mixed')}, "
                "covering adoption risk, operational fit, and differentiation."
            )
        },
    }

    for tool, args in payloads.items():
        envelope = json.dumps({"tool": tool, "args": args})
        result = agent(envelope)
        print(f"\n== {tool.upper()} ==\n{result}")


if __name__ == "__main__":
    main()
```

---

## 📦 Optional: Publish Script via GitHub CLI

```bash
gh auth status  # ensure logged in
gh gist create modaic_jtbd_agent_demo.py \
  -d "Modaic JTBD agent quickstart harness" \
  -p
```

Attach the markdown narrative above as the gist description or an additional `README.md` file to match our typical multi-section style.

---

## ✅ Recap

| Step | Action | Output |
|------|--------|--------|
| 1 | Load env / reload agent | pulls latest Modaic revision |
| 2 | Run demo script | JSON for `deconstruct`, `jobs`, `moat`, `judge` |
| 3 | Inspect responses | Quick sanity check before shipping |
| 4 | Publish gist (optional) | Shareable runbook for team |

Happy exploring! Ping `@straughterguthrie` if you want to plug in a vector retriever or expand the SSE streaming demo.
