"""Agent factory utilities and tracing-aware invocation helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from modaic import PrecompiledAgent, Retriever
from opentelemetry import trace

from service.modaic_agent import JTBDDSPyAgent, JTBDConfig
from service.retrievers import NotesRetriever, NullRetriever


_AGENT: Optional[PrecompiledAgent] = None
_TRACER = trace.get_tracer("jtbd.agent")


def _make_retriever() -> Retriever:
    kind = os.getenv("RETRIEVER_KIND", "null").lower()
    if kind == "notes":
        notes_env = os.getenv("RETRIEVER_NOTES", "")
        notes = [line for line in notes_env.splitlines() if line.strip()]
        top_k = int(os.getenv("RETRIEVER_TOP_K", "3"))
        return NotesRetriever(notes=notes, top_k=top_k)
    return NullRetriever()


def _new_local_agent() -> PrecompiledAgent:
    return JTBDDSPyAgent(JTBDConfig(), retriever=_make_retriever())


def _load_from_hub() -> PrecompiledAgent:
    repo = os.getenv("MODAIC_AGENT_ID", "").strip()
    if not repo:
        raise RuntimeError("MODAIC_AGENT_ID is required to load an agent from the Modaic hub")

    revision = os.getenv("MODAIC_AGENT_REV") or None
    retriever = _make_retriever()

    if revision:
        return JTBDDSPyAgent.from_precompiled(repo, revision=revision, retriever=retriever)
    return JTBDDSPyAgent.from_precompiled(repo, retriever=retriever)


def get_agent() -> PrecompiledAgent:
    global _AGENT
    if _AGENT is None:
        try:
            if os.getenv("MODAIC_AGENT_ID"):
                _AGENT = _load_from_hub()
            else:
                _AGENT = _new_local_agent()
        except Exception:
            _AGENT = _new_local_agent()
    return _AGENT


def reload_agent() -> str:
    global _AGENT
    _AGENT = None
    return "hub" if os.getenv("MODAIC_AGENT_ID") else "local"


def call_agent_envelope(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"tool": tool, "args": args}
    with _TRACER.start_as_current_span("agent.invoke") as span:
        span.set_attribute("agent.tool", tool)
        span.set_attribute("agent.arg_keys", ",".join(sorted(args.keys())))
        agent = get_agent()
        raw = agent(json.dumps(payload))

    telemetry: Optional[Dict[str, Any]] = None
    if hasattr(agent, "flush_usage_metrics"):
        telemetry = agent.flush_usage_metrics()  # type: ignore[assignment]
        if telemetry:
            span.set_attribute("agent.tool_calls", telemetry.get("total_calls", 0))
            span.set_attribute("agent.tool_errors", telemetry.get("error_count", 0))

    try:
        result: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        result = {"error": "Agent returned non-JSON response", "raw": raw}

    if telemetry and telemetry.get("total_calls"):
        result.setdefault("_telemetry", {})["tool_usage"] = telemetry

    return result

