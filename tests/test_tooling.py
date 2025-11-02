from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.tooling import ToolRegistry, ToolSpec, ToolUsageTracker


def test_tool_registry_records_usage() -> None:
    tracker = ToolUsageTracker()
    registry = ToolRegistry(namespace="demo", tracker=tracker)

    def sample_tool(text: str = "") -> str:
        return json.dumps({"echo": text})

    tool = registry.register(
        ToolSpec(
            name="echo",
            description="Echo back the provided text in JSON format.",
            args_schema={"text": "Input string."},
            returns="JSON string containing an 'echo' field.",
            example='{ "text": "hello" }',
        ),
        sample_tool,
    )

    assert "Echo back" in (tool.__doc__ or "")

    out = tool(text="alpha")
    assert json.loads(out) == {"echo": "alpha"}

    usage = registry.flush_usage()
    assert usage["total_calls"] == 1
    assert usage["error_count"] == 0
    assert usage["by_tool"]["echo"]["count"] == 1
    assert "avg_latency_s" in usage["by_tool"]["echo"]

    # The tracker should be reset after flush
    assert registry.flush_usage()["total_calls"] == 0
