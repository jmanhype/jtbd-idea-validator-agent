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
from service.tooling import ToolRegistry, ToolSpec, ToolUsageTracker
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

        self._usage_tracker = ToolUsageTracker()
        self._registry = ToolRegistry(namespace="jtbd", tracker=self._usage_tracker)

        self._register_tools()

        super().__init__(config=config, retriever=self.retriever, **kwargs)

        # ReAct agent that can call the retriever alongside core tools.
        self.react = dspy.ReAct(
            signature="question->answer",
            tools=self._registry.tools_for_react(),
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
            return self._registry.invoke("retrieve", query=query)
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
        return self._registry.invoke("deconstruct", idea=idea, hunches=hunches or [])

    def jobs(self, context: Optional[Dict[str, Any]] = None, constraints: Optional[List[str]] = None) -> str:
        return self._registry.invoke("jobs", context=context or {}, constraints=constraints or [])

    def moat(self, concept: str, triggers: Optional[str] = "") -> str:
        return self._registry.invoke("moat", concept=concept, triggers=triggers or "")

    def judge(self, summary: str) -> str:
        return self._registry.invoke("judge", summary=summary)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _dispatch(self, tool: str, args: Dict[str, Any]) -> str:
        slug = tool.lower()
        if slug in {"retrieve", "retriever", "context"}:
            return self._registry.invoke("retrieve", query=args.get("query", ""))
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

    # ------------------------------------------------------------------
    # Tool registration & telemetry
    # ------------------------------------------------------------------
    def _register_tools(self) -> None:
        self._registry.register(
            ToolSpec(
                name="retrieve",
                description="Retrieve contextual notes that match the provided query terms.",
                args_schema={"query": "Natural language description of the context you need."},
                returns="JSON string with a 'context' key containing newline-delimited snippets.",
                example="{\"query\": \"pain points for physical therapists\"}",
            ),
            self._retrieve_tool,
        )

        self._registry.register(
            ToolSpec(
                name="deconstruct",
                description="Break an idea into explicit assumptions with confidence levels and evidence notes.",
                args_schema={
                    "idea": "Short paragraph describing the idea or problem/solution pair.",
                    "hunches": "List of bullet hypotheses supplied by the operator.",
                },
                returns="JSON string with an 'assumptions' list of structured objects.",
                example="{\"idea\": \"AI inbox triage for clinicians\", \"hunches\": [\"after-hours burden\"]}",
            ),
            self._deconstruct_tool,
        )

        self._registry.register(
            ToolSpec(
                name="jobs",
                description="Generate Jobs To Be Done statements plus Four Forces analysis for the audience.",
                args_schema={
                    "context": "Dictionary describing the market, customer, or workflow context.",
                    "constraints": "List of explicit limitations, risks, or requirements.",
                },
                returns="JSON string with a 'jobs' list containing statements and forces.",
                example="{\"context\": {\"industry\": \"Healthcare\"}, \"constraints\": [\"HIPAA\"]}",
            ),
            self._jobs_tool,
        )

        self._registry.register(
            ToolSpec(
                name="moat",
                description="Recommend innovation layers and moat triggers to strengthen differentiation.",
                args_schema={
                    "concept": "Short concept name or description.",
                    "triggers": "Optional hints such as competitor names or risk factors.",
                },
                returns="JSON string with a 'layers' list describing Doblin innovation layers.",
                example="{\"concept\": \"Rehab exercise tracking app\", \"triggers\": \"Competitors: Physitrack\"}",
            ),
            self._moat_tool,
        )

        self._registry.register(
            ToolSpec(
                name="judge",
                description="Score the idea across standardized criteria with rationale for each dimension.",
                args_schema={"summary": "1-3 sentences summarizing the idea, audience, and value."},
                returns="JSON string with a 'scorecard' object containing criteria and totals.",
                example="{\"summary\": \"Summarize the AI rehab coach for clinics\"}",
            ),
            self._judge_tool,
        )

    def _retrieve_tool(self, query: str = "") -> str:
        context = self.retriever.retrieve(query)
        return self._as_json({"context": context})

    def _deconstruct_tool(self, idea: str = "", hunches: Optional[List[str]] = None) -> str:
        items = self._deconstruct(idea=idea, hunches=hunches or [])
        return self._as_json({"assumptions": [item.model_dump() for item in items]})

    def _jobs_tool(self, context: Optional[Dict[str, Any]] = None, constraints: Optional[List[str]] = None) -> str:
        jobs = self._jobs(context=context or {}, constraints=constraints or [])
        return self._as_json({"jobs": [job.model_dump() for job in jobs]})

    def _moat_tool(self, concept: str = "", triggers: Optional[str] = "") -> str:
        layers = self._moat(concept=concept, triggers=triggers or "")
        return self._as_json({"layers": [layer.model_dump() for layer in layers]})

    def _judge_tool(self, summary: str = "") -> str:
        scorecard = judge_with_arbitration(summary=summary)
        return self._as_json({"scorecard": scorecard.model_dump()})

    def flush_usage_metrics(self) -> Dict[str, Any]:
        """Expose usage telemetry so callers can aggregate analytics."""

        return self._registry.flush_usage()

    def describe_tools(self) -> List[Dict[str, Any]]:
        """Return structured tool metadata for documentation or evaluation."""

        return self._registry.describe()
