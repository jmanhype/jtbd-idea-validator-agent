#!/usr/bin/env python
"""Quick runner against the Modaic-hosted JTBD agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from service.agent_loader import call_agent_envelope, reload_agent


FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "idea.json"


def main() -> int:
    agent_id = os.getenv("MODAIC_AGENT_ID")
    if not agent_id:
        print("MODAIC_AGENT_ID must be set (e.g. 'straughterguthrie/jtbd-agent').", file=sys.stderr)
        return 1

    if not os.getenv("MODAIC_TOKEN"):
        print("Warning: MODAIC_TOKEN not set; assuming public repo access.", file=sys.stderr)

    # Force reload so we fetch the latest remote revision.
    reload_agent()

    if not FIXTURE_PATH.exists():
        print(f"Fixture not found: {FIXTURE_PATH}", file=sys.stderr)
        return 1

    sample = json.loads(FIXTURE_PATH.read_text())
    idea_title = sample.get("title", "AI assistant")
    hunches = sample.get("hunches", [])

    # Build prompts for each tool.
    deconstruct_payload = {"idea": idea_title, "hunches": hunches}
    jobs_payload = {
        "context": {
            "idea": idea_title,
            "audience": sample.get("context", {}).get("audience", "mixed"),
            "hunches": hunches,
        },
        "constraints": sample.get("constraints", []),
    }
    moat_payload = {
        "concept": f"{idea_title} moat strategy",
        "triggers": "Retention, brand, integrations",
    }
    judge_payload = {
        "summary": (
            f"Summarize the strengths and risks of {idea_title} for clinicians, "
            "highlighting operational fit, measurable outcomes, and differentiation."
        )
    }

    results = {
        "agent_id": agent_id,
        "deconstruct": call_agent_envelope("deconstruct", deconstruct_payload),
        "jobs": call_agent_envelope("jobs", jobs_payload),
        "moat": call_agent_envelope("moat", moat_payload),
        "judge": call_agent_envelope("judge", judge_payload),
    }

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
