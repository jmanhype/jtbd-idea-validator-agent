#!/usr/bin/env python
"""Push the JTBD DSPy agent to Modaic Hub using environment variables."""

from __future__ import annotations

import os
import sys

from service.modaic_agent import JTBDDSPyAgent, JTBDConfig
from service.retrievers import NotesRetriever, NullRetriever


def build_retriever():
    kind = os.getenv("RETRIEVER_KIND", "notes").lower()
    if kind == "notes":
        raw = os.getenv("RETRIEVER_NOTES", "")
        notes = [line for line in raw.splitlines() if line.strip()]
        return NotesRetriever(notes=notes or ["JTBD primer"])
    return NullRetriever()


def main() -> int:
    agent_id = os.getenv("MODAIC_AGENT_ID")
    token = os.getenv("MODAIC_TOKEN")

    if not agent_id:
        print("MODAIC_AGENT_ID is not set", file=sys.stderr)
        return 1
    if not token:
        print("MODAIC_TOKEN is not set", file=sys.stderr)
        return 1

    agent = JTBDDSPyAgent(JTBDConfig(), retriever=build_retriever())
    agent.push_to_hub(agent_id, with_code=True)
    print(f"Agent pushed to Modaic Hub: {agent_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

