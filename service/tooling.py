"""Utility classes for defining, documenting, and instrumenting JTBD agent tools."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


def _preview_payload(payload: Any, limit: int = 160) -> Optional[str]:
    """Return a short, single-line preview of a payload for telemetry."""

    if payload is None:
        return None

    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload)
        except TypeError:
            text = repr(payload)

    text = " ".join(text.split())
    if len(text) > limit:
        return f"{text[:limit]}…"
    return text


@dataclass
class ToolCallRecord:
    """Structured telemetry for a single tool invocation."""

    name: str
    args: Dict[str, Any]
    latency_s: float
    result_preview: Optional[str] = None
    error: Optional[str] = None


class ToolUsageTracker:
    """Collects per-call telemetry so callers can build aggregate summaries."""

    def __init__(self) -> None:
        self._calls: List[ToolCallRecord] = []

    def record(
        self,
        *,
        name: str,
        args: Dict[str, Any],
        latency_s: float,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        preview = _preview_payload(result)
        self._calls.append(
            ToolCallRecord(
                name=name,
                args=dict(args),
                latency_s=round(latency_s, 4),
                result_preview=preview,
                error=error,
            )
        )

    def summary(self) -> Dict[str, Any]:
        """Return aggregated telemetry without mutating internal state."""

        total_calls = len(self._calls)
        error_count = sum(1 for call in self._calls if call.error)

        by_tool: Dict[str, Dict[str, Any]] = {}
        for call in self._calls:
            stats = by_tool.setdefault(
                call.name,
                {"count": 0, "errors": 0, "_latencies": []},
            )
            stats["count"] += 1
            stats["errors"] += 1 if call.error else 0
            stats["_latencies"].append(call.latency_s)

        for stats in by_tool.values():
            latencies = stats.pop("_latencies")
            if latencies:
                stats["avg_latency_s"] = round(sum(latencies) / len(latencies), 4)
            else:
                stats["avg_latency_s"] = 0.0

        recent_calls = [
            {
                "tool": call.name,
                "latency_s": call.latency_s,
                "error": call.error,
                "result_preview": call.result_preview,
            }
            for call in self._calls[-5:]
        ]

        return {
            "total_calls": total_calls,
            "error_count": error_count,
            "by_tool": by_tool,
            "recent_calls": recent_calls,
        }

    def flush(self) -> Dict[str, Any]:
        """Return the aggregated telemetry and reset the tracker."""

        summary = self.summary()
        self._calls.clear()
        return summary


@dataclass
class ToolSpec:
    """Human- and machine-readable description of an agent tool."""

    name: str
    description: str
    args_schema: Dict[str, str]
    returns: str
    namespace: str = "jtbd"
    example: Optional[str] = None

    def format_docstring(self) -> str:
        """Produce a docstring that ReAct and other planners can consume."""

        lines = [self.description.strip(), "", "Parameters:"]
        for arg, desc in self.args_schema.items():
            lines.append(f"  {arg}: {desc}")
        lines.append("")
        lines.append(f"Returns: {self.returns}")
        if self.example:
            lines.append("")
            lines.append(f"Example: {self.example}")
        return "\n".join(lines)


class ToolRegistry:
    """Registers deterministic tool functions with rich metadata and telemetry."""

    def __init__(self, namespace: str = "jtbd", tracker: Optional[ToolUsageTracker] = None) -> None:
        self.namespace = namespace
        self.tracker = tracker or ToolUsageTracker()
        self._specs: Dict[str, ToolSpec] = {}
        self._functions: Dict[str, Callable[..., Any]] = {}
        self._order: List[str] = []

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Register a tool and return the instrumented callable."""

        name = spec.name
        if name in self._specs:
            raise ValueError(f"Tool '{name}' already registered")

        instrumented = self._instrument(name, fn, spec)
        instrumented.__doc__ = spec.format_docstring()
        instrumented.__name__ = f"{self.namespace}_{name}"

        self._specs[name] = spec
        self._functions[name] = instrumented
        self._order.append(name)
        return instrumented

    def _instrument(self, name: str, fn: Callable[..., Any], spec: ToolSpec) -> Callable[..., Any]:
        def instrumented(**kwargs: Any) -> Any:
            start = time.perf_counter()
            error: Optional[str] = None
            result: Any = None
            try:
                result = fn(**kwargs)
                return result
            except Exception as exc:  # pragma: no cover - surfaced via tracker
                error = repr(exc)
                raise
            finally:
                duration = time.perf_counter() - start
                self.tracker.record(
                    name=name,
                    args=kwargs,
                    latency_s=duration,
                    result=result if error is None else None,
                    error=error,
                )

        instrumented.__signature__ = getattr(fn, "__signature__", None)  # type: ignore[attr-defined]
        instrumented._tool_spec = spec  # type: ignore[attr-defined]
        return instrumented

    def tools_for_react(self) -> List[Callable[..., Any]]:
        """Return instrumented callables preserving registration order."""

        return [self._functions[name] for name in self._order]

    def describe(self) -> List[Dict[str, Any]]:
        """Return metadata describing each registered tool."""

        return [
            {
                "name": spec.name,
                "namespace": spec.namespace,
                "description": spec.description,
                "args": spec.args_schema,
                "returns": spec.returns,
                "example": spec.example,
            }
            for spec in (self._specs[name] for name in self._order)
        ]

    def invoke(self, name: str, **kwargs: Any) -> Any:
        """Invoke an instrumented function by name."""

        if name not in self._functions:
            raise KeyError(f"Unknown tool '{name}'")
        return self._functions[name](**kwargs)

    def flush_usage(self) -> Dict[str, Any]:
        """Return telemetry for all tools and reset the tracker."""

        return self.tracker.flush()

    def recent_calls(self) -> Iterable[ToolCallRecord]:
        """Expose the raw call records for downstream observers."""

        return tuple(self.tracker._calls)

