"""Modaic-compatible JTBD DSPy agent with retriever integration."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import dspy
from modaic import PrecompiledAgent, PrecompiledConfig, Retriever

from plugins.llm_dspy import (
    Deconstruct,
    Jobs,
    Moat,
    configure_lm,
    judge_with_arbitration,
)
from .retrievers import NullRetriever


configure_lm()


class JTBDConfig(PrecompiledConfig):
    default_mode: str = "deconstruct"
    allow_freeform_route: bool = True
    return_json: bool = True


class JTBDDSPyAgent(PrecompiledAgent):
    """Agent exposing DSPy modules via Modaic's PrecompiledAgent interface."""

    config: JTBDConfig

    def __init__(self, config: Optional[JTBDConfig] = None, retriever: Optional[Retriever] = None, **kwargs):
        config = config or JTBDConfig()
        self.config = config
        self.retriever = retriever or NullRetriever()

        self._deconstruct = Deconstruct()
        self._jobs = Jobs()
        self._moat = Moat()

        super().__init__(config=config, retriever=self.retriever, **kwargs)

        # ReAct agent that can call the retriever alongside core tools.
        self.react = dspy.ReAct(
            signature="question->answer",
            tools=[
                self.retriever.retrieve,
                self.deconstruct,
                self.jobs,
                self.moat,
                self.judge,
            ],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, query: str, **kwargs) -> str:  # type: ignore[override]
        return self.forward(query, **kwargs)

    def forward(self, query: str, **kwargs) -> str:  # type: ignore[override]
        # Allow JSON envelopes to force tool dispatch.
        try:
            payload = json.loads(query)
        except Exception:
            payload = None

        if isinstance(payload, dict) and "tool" in payload and "args" in payload:
            return self._dispatch(str(payload["tool"]), payload.get("args") or {})

        if not self.config.allow_freeform_route:
            return self._dispatch(self.config.default_mode, {"query": query})

        lowered = query.lower()
        if any(token in lowered for token in ("context", "note", "retriev")):
            context = self.retriever.retrieve(query)
            return self._as_json({"context": context})
        if any(token in lowered for token in ("assumption", "deconstruct")):
            return self.deconstruct(idea=query, hunches=[])
        if "jtbd" in lowered or "job" in lowered:
            return self.jobs(context={"prompt": query}, constraints=[])
        if any(token in lowered for token in ("moat", "defens")):
            return self.moat(concept=query, triggers="")
        if any(token in lowered for token in ("judge", "score", "evaluate")):
            return self.judge(summary=query)

        return self._dispatch(self.config.default_mode, {"query": query})

    # ------------------------------------------------------------------
    # Tool wrappers
    # ------------------------------------------------------------------
    def deconstruct(self, idea: str, hunches: Optional[List[str]] = None) -> str:
        items = self._deconstruct(idea=idea, hunches=hunches or [])
        return self._as_json({"assumptions": [item.model_dump() for item in items]})

    def jobs(self, context: Optional[Dict[str, Any]] = None, constraints: Optional[List[str]] = None) -> str:
        jobs = self._jobs(context=context or {}, constraints=constraints or [])
        return self._as_json({"jobs": [job.model_dump() for job in jobs]})

    def moat(self, concept: str, triggers: Optional[str] = "") -> str:
        layers = self._moat(concept=concept, triggers=triggers or "")
        return self._as_json({"layers": [layer.model_dump() for layer in layers]})

    def judge(self, summary: str) -> str:
        scorecard = judge_with_arbitration(summary=summary)
        return self._as_json({"scorecard": scorecard.model_dump()})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dispatch(self, tool: str, args: Dict[str, Any]) -> str:
        slug = tool.lower()
        if slug in {"retrieve", "retriever", "context"}:
            context = self.retriever.retrieve(args.get("query", ""))
            return self._as_json({"context": context})
        if slug == "deconstruct":
            return self.deconstruct(
                idea=args.get("idea", ""),
                hunches=args.get("hunches") or [],
            )
        if slug == "jobs":
            return self.jobs(
                context=args.get("context") or {},
                constraints=args.get("constraints") or [],
            )
        if slug == "moat":
            return self.moat(
                concept=args.get("concept", ""),
                triggers=args.get("triggers", ""),
            )
        if slug == "judge":
            return self.judge(summary=args.get("summary", ""))
        return self._as_json({"error": f"unknown tool '{tool}'"})

    def _as_json(self, payload: Dict[str, Any]) -> str:
        if self.config.return_json:
            return json.dumps(payload)
        return str(payload)
