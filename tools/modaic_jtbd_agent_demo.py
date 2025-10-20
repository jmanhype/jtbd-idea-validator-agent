#!/usr/bin/env python
"""Minimal demo script for the Modaic-hosted JTBD DSPy agent.

Run with:

    MODAIC_AGENT_ID=straughterguthrie/jtbd-agent \
    JTBD_DSPY_MODEL=${JTBD_DSPY_MODEL:-gpt-4o-mini} \
    python modaic_jtbd_agent_demo.py

Requires a compatible model API key (e.g., `OPENAI_API_KEY`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from service.modaic_agent import JTBDDSPyAgent


@dataclass
class DemoPayloads:
    title: str
    hunches: list[str]
    audience: str

    @classmethod
    def from_fixture(cls) -> "DemoPayloads":
        fixture_path = os.getenv(
            "DEMO_IDEA_PATH",
            os.path.join(os.path.dirname(__file__), "..", "fixtures", "idea.json"),
        )
        try:
            with open(fixture_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {
                "title": "AI inbox triage for clinicians",
                "hunches": ["too many portal messages", "after-hours burden"],
                "context": {"audience": "safety-net hospital"},
                "constraints": [],
            }
        return cls(
            title=data.get("title", "AI assistant"),
            hunches=data.get("hunches", []),
            audience=data.get("context", {}).get("audience", "mixed"),
        )

    def to_requests(self) -> Dict[str, Dict[str, Any]]:
        return {
            "deconstruct": {"idea": self.title, "hunches": self.hunches},
            "jobs": {
                "context": {
                    "idea": self.title,
                    "audience": self.audience,
                    "hunches": self.hunches,
                },
                "constraints": [],
            },
            "moat": {
                "concept": f"{self.title} moat strategy",
                "triggers": "retention, brand, integrations",
            },
            "judge": {
                "summary": (
                    f"Assess {self.title} for {self.audience}, focusing on adoption risk, "
                    "operational fit, and differentiation."
                )
            },
        }


def main() -> None:
    agent_id = os.getenv("MODAIC_AGENT_ID", "straughterguthrie/jtbd-agent")
    print(f"Loading Modaic agent: {agent_id}")

    agent = JTBDDSPyAgent.from_precompiled(agent_id)

    payloads = DemoPayloads.from_fixture().to_requests()

    print("\n== Deconstruct ==")
    print(agent(json.dumps({"tool": "deconstruct", "args": payloads["deconstruct"]})))

    print("\n== Jobs ==")
    print(agent(json.dumps({"tool": "jobs", "args": payloads["jobs"]})))

    print("\n== Moat ==")
    print(agent(json.dumps({"tool": "moat", "args": payloads["moat"]})))

    print("\n== Judge ==")
    print(agent(json.dumps({"tool": "judge", "args": payloads["judge"]})))


if __name__ == "__main__":
    main()
