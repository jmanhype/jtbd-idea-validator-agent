"""Programmatic evaluation harness for the JTBD DSPy agent tools."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from service.agent_loader import call_agent_envelope, reload_agent


Validator = Callable[[Dict[str, Any]], Tuple[bool, str]]


def _ensure_path(path: Sequence[str], *, minimum: int = 1) -> Validator:
    """Validate that a nested field exists and has a minimum length or truthiness."""

    key_path = ".".join(path)

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, str]:
        target: Any = payload
        try:
            for key in path:
                target = target[key]
        except Exception:
            return False, f"Missing '{key_path}' in response"

        if isinstance(target, (list, tuple, set)):
            ok = len(target) >= minimum
        elif isinstance(target, dict):
            ok = len(target) >= minimum
        elif isinstance(target, str):
            ok = bool(target.strip())
        else:
            ok = target is not None

        if not ok:
            return False, f"Field '{key_path}' did not meet minimum requirements"
        return True, ""

    return _validator


def _ensure_score_range(minimum: float, maximum: float) -> Validator:
    def _validator(payload: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            total = float(payload["scorecard"]["total"])
        except Exception:
            return False, "Missing scorecard.total"
        if not (minimum <= total <= maximum):
            return False, f"Score {total} out of expected range {minimum}-{maximum}"
        return True, ""

    return _validator


@dataclass
class EvaluationStep:
    tool: str
    args: Dict[str, Any]
    validate: Validator


@dataclass
class EvaluationTask:
    name: str
    description: str
    steps: List[EvaluationStep]

    def run(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        aggregate_usage: Dict[str, Any] = {
            "total_calls": 0,
            "error_count": 0,
            "by_tool": {},
        }
        passed = True

        for step in self.steps:
            response = call_agent_envelope(step.tool, step.args)
            telemetry = response.get("_telemetry", {}).get("tool_usage", {})
            _merge_usage(aggregate_usage, telemetry)

            ok, message = step.validate(response)
            passed = passed and ok
            results.append(
                {
                    "tool": step.tool,
                    "args": step.args,
                    "passed": ok,
                    "message": message,
                    "telemetry": telemetry,
                }
            )

        _finalize_usage(aggregate_usage)
        return {
            "name": self.name,
            "description": self.description,
            "passed": passed,
            "steps": results,
            "usage": aggregate_usage,
        }


def _merge_usage(accumulator: Dict[str, Any], usage: Dict[str, Any]) -> None:
    if not usage:
        return

    accumulator["total_calls"] += usage.get("total_calls", 0)
    accumulator["error_count"] += usage.get("error_count", 0)

    for tool, stats in usage.get("by_tool", {}).items():
        bucket = accumulator["by_tool"].setdefault(
            tool, {"count": 0, "errors": 0, "_latency_sum": 0.0}
        )
        count = stats.get("count", 0)
        bucket["count"] += count
        bucket["errors"] += stats.get("errors", 0)
        bucket["_latency_sum"] += stats.get("avg_latency_s", 0.0) * count


def _finalize_usage(accumulator: Dict[str, Any]) -> None:
    for stats in accumulator["by_tool"].values():
        count = max(stats["count"], 1)
        stats["avg_latency_s"] = round(stats.pop("_latency_sum") / count, 4)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _default_summary(idea: Dict[str, Any]) -> str:
    summary = idea.get("title", "Untitled idea")
    problem = idea.get("problem_statement")
    if problem:
        summary += f" Problem: {problem[:160]}"
    value_props = idea.get("value_propositions") or []
    if value_props:
        summary += f" Value props: {', '.join(value_props[:2])}"
    return summary


def _build_tasks(idea: Dict[str, Any]) -> List[EvaluationTask]:
    hunches = idea.get("hunches") or []
    constraints: List[str] = []
    constraints.extend(idea.get("constraints") or [])
    constraints.extend(idea.get("risks_and_challenges") or [])
    context = idea.get("context") or {}
    concept = idea.get("title", "")
    triggers = ""
    competitors = idea.get("competitive_landscape") or []
    if competitors:
        triggers = f"Competitors: {competitors[0]}"

    summary = _default_summary(idea)

    return [
        EvaluationTask(
            name="full_jtbd_review",
            description="Run deconstruction, jobs, and judging to simulate an analyst workflow.",
            steps=[
                EvaluationStep(
                    tool="deconstruct",
                    args={"idea": summary, "hunches": hunches},
                    validate=_ensure_path(["assumptions"], minimum=3),
                ),
                EvaluationStep(
                    tool="jobs",
                    args={"context": context, "constraints": constraints[:5]},
                    validate=_ensure_path(["jobs"], minimum=3),
                ),
                EvaluationStep(
                    tool="judge",
                    args={"summary": summary},
                    validate=_ensure_score_range(0.0, 10.0),
                ),
            ],
        ),
        EvaluationTask(
            name="moat_depth_check",
            description="Ensure moat recommendations align with the market context after jobs analysis.",
            steps=[
                EvaluationStep(
                    tool="jobs",
                    args={"context": context, "constraints": constraints[:3]},
                    validate=_ensure_path(["jobs"], minimum=2),
                ),
                EvaluationStep(
                    tool="moat",
                    args={"concept": concept, "triggers": triggers},
                    validate=_ensure_path(["layers"], minimum=2),
                ),
            ],
        ),
        EvaluationTask(
            name="assumption_regression",
            description="Verify repeated deconstruction remains stable and returns structured evidence.",
            steps=[
                EvaluationStep(
                    tool="deconstruct",
                    args={"idea": concept or summary, "hunches": hunches[:3]},
                    validate=_ensure_path(["assumptions"], minimum=2),
                ),
                EvaluationStep(
                    tool="deconstruct",
                    args={"idea": concept or summary, "hunches": hunches[:3]},
                    validate=_ensure_path(["assumptions"], minimum=2),
                ),
            ],
        ),
    ]


def _print_summary(results: List[Dict[str, Any]]) -> None:
    print("\n=== JTBD Agent Evaluation Summary ===")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']} - {result['description']}")
        for step in result["steps"]:
            step_status = "PASS" if step["passed"] else "FAIL"
            telemetry = step.get("telemetry") or {}
            latency = None
            for stats in telemetry.get("by_tool", {}).values():
                latency = stats.get("avg_latency_s")
            latency_str = f" ({latency:.3f}s)" if latency is not None else ""
            message = f" - {step['tool']}{latency_str}: {step_status}"
            if step["message"]:
                message += f" -> {step['message']}"
            print(message)
        usage = result["usage"]
        print(
            f"    Tool calls: {usage['total_calls']} | Errors: {usage['error_count']} | Tools: "
            + ", ".join(
                f"{tool}={stats['count']}" for tool, stats in usage["by_tool"].items()
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate JTBD agent tools with realistic workflows")
    parser.add_argument(
        "--idea",
        type=Path,
        default=Path("examples/rehab_exercise_tracking_rich.json"),
        help="Path to an idea payload JSON file used for evaluation tasks.",
    )
    parser.add_argument(
        "--reload-agent",
        action="store_true",
        help="Force reload of the agent before running evaluations.",
    )
    args = parser.parse_args()

    if args.reload_agent:
        mode = reload_agent()
        print(f"Reloaded agent ({mode}) before evaluation")

    idea = _load_json(args.idea)
    tasks = _build_tasks(idea)

    results = [task.run() for task in tasks]
    _print_summary(results)

    all_passed = all(result["passed"] for result in results)
    if not all_passed:
        print("\nAt least one evaluation failed.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
