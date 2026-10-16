"""Tests for the public data contract."""

from __future__ import annotations

import pytest

from maestro_sdk import (
    ContextBucket,
    ContextReport,
    CostReport,
    QueryOptions,
    QueryResult,
    QueryTiming,
    QueryUsage,
    StreamDelta,
    ToolCall,
    UsageReport,
)


def test_query_options_defaults_are_safe() -> None:
    opts = QueryOptions()
    assert opts.cwd is None
    assert opts.system is None
    assert opts.startup_timeout_seconds > 0
    assert opts.stream_idle_timeout_seconds > 0
    # Output-quiescence completion is opt-in: None leaves the read loop on its
    # EOF / idle-marker / complete_on done-signals only (proposal 081).
    assert opts.idle_quiescence_seconds is None
    # Spawn-pinned session id is opt-in: None leaves the spawn command bare and
    # the reported id sourced only from an on-screen marker (proposal 081).
    assert opts.session_id is None


def test_query_usage_requires_all_counts() -> None:
    # No defaults: constructing a QueryUsage is an assertion that you measured
    # something. The "unavailable" path stays at QueryResult.usage = None so
    # zeroed defaults can never silently fake usage in dashboards.
    with pytest.raises(TypeError):
        QueryUsage()  # type: ignore[call-arg]


def test_query_usage_round_trip() -> None:
    usage = QueryUsage(
        input_tokens=1,
        output_tokens=2,
        cache_read_tokens=3,
        cache_creation_tokens=4,
    )
    assert usage.input_tokens == 1
    assert usage.output_tokens == 2
    assert usage.cache_read_tokens == 3
    assert usage.cache_creation_tokens == 4


def test_query_timing_total_sums_components() -> None:
    timing = QueryTiming(send_ms=120.0, stream_ms=400.0, heartbeat_ms=800.0)
    assert timing.total_ms == pytest.approx(1320.0)


def test_tool_call_has_independent_default_args() -> None:
    a = ToolCall(name="Read")
    b = ToolCall(name="Write")
    # Frozen dataclass + default_factory: mutating one must not bleed into the other.
    a.args["k"] = 1
    assert "k" not in b.args


def test_query_result_minimal_construction() -> None:
    result = QueryResult(response="hi")
    assert result.response == "hi"
    assert result.tool_calls == []
    assert result.usage is None
    assert result.session_id is None
    assert result.raw_screen == ""
    assert result.timing is None


def test_query_result_full_construction() -> None:
    timing = QueryTiming(send_ms=10.0, stream_ms=20.0, heartbeat_ms=30.0)
    result = QueryResult(
        response="ok",
        tool_calls=[ToolCall(name="Bash", args={"cmd": "ls"})],
        usage=QueryUsage(
            input_tokens=5,
            output_tokens=2,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        ),
        session_id="abc-123",
        raw_screen="> ok",
        timing=timing,
    )
    assert result.tool_calls[0].name == "Bash"
    assert result.usage is not None
    assert result.usage.input_tokens == 5
    assert result.session_id == "abc-123"
    assert result.timing is timing


def test_cost_report_defaults_are_estimated_and_unmeasured() -> None:
    # No measured field is faked: an unmeasured report carries None counts, and
    # a /cost figure is always flagged a local estimate, never authoritative.
    report = CostReport()
    assert report.total_cost_usd is None
    assert report.total_tokens is None
    assert report.estimated is True


def test_usage_report_defaults_are_unmeasured() -> None:
    report = UsageReport()
    assert report.plan_percent_used is None
    assert report.plan_label is None


def test_context_report_defaults_to_empty_buckets() -> None:
    report = ContextReport()
    assert report.used_tokens is None
    assert report.total_tokens is None
    assert report.buckets == ()


def test_context_bucket_holds_name_and_tokens() -> None:
    bucket = ContextBucket(name="System prompt", tokens=2_500)
    assert bucket.name == "System prompt"
    assert bucket.tokens == 2_500


def test_stream_delta_kinds() -> None:
    a = StreamDelta(text="thinking...", kind="thinking", screen_snapshot="")
    b = StreamDelta(text="answer", kind="answer", screen_snapshot="> answer")
    assert a.kind == "thinking"
    assert b.kind == "answer"
    assert b.screen_snapshot == "> answer"
