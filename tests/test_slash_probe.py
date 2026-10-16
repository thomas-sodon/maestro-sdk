"""Tests for the slash-command metrics extractor.

The extractor is a pure function of a screen snapshot and a marker pack, so every
test is a table test: feed a captured `/cost`, `/usage`, or `/context` redraw and
assert the parsed report — or assert `None` on a garbled, empty, or
version-skewed snapshot (the honest-degenerate discipline, no fabricated zeros).
No I/O, no sleep, no subprocess.

`SlashProbeMarkers` is the layout policy; tests pass explicit markers tuned to
their fixtures so they pin the *mechanism*, not any one CLI version's glyphs. A
couple of glyphy lines appear verbatim to prove `.search` skips leading TUI
furniture after SGR stripping.
"""

from __future__ import annotations

import pytest

from maestro_sdk import (
    DEFAULT_PROBE_MARKERS,
    ContextBucket,
    ContextReport,
    CostReport,
    SlashProbeMarkers,
    UsageReport,
    parse_context,
    parse_cost,
    parse_usage,
)

# ---------------------------------------------------------------------------
# Captured fixtures + the markers tuned to them.

_COST_SNAPSHOT = "\n".join(
    [
        "Total cost:            $0.4823",
        "Total duration (API):  2m 14.3s",
        "Total code changes:    128 lines added, 9 lines removed",
        "Token usage:           1.4M tokens",
    ]
)
_COST_MARKERS = SlashProbeMarkers(
    cost_total=r"Total cost:\s*\$?(?P<usd>[\d.,]+)",
    cost_tokens=r"Token usage:\s*(?P<tokens>[\d.,]+[kKmM]?)",
)

_USAGE_SNAPSHOT = "\n".join(
    [
        "Current week (all models)",
        "████░░░░  45% used · resets Mon",
    ]
)
_USAGE_MARKERS = SlashProbeMarkers(usage_plan=r"(?P<percent>[\d.]+)% used")
_USAGE_MARKERS_WITH_LABEL = SlashProbeMarkers(
    usage_plan=r"(?P<percent>[\d.]+)% used · resets (?P<label>\w+)"
)

_CONTEXT_SNAPSHOT = "\n".join(
    [
        "Context Usage",
        "claude-opus-4 · 47.2k/200k tokens (24%)",
        "",
        "⛁ System prompt: 2.5k tokens (1.3%)",
        "⛁ MCP tools: 3.2k tokens (1.6%)",
        "⛀ Free space: 140k tokens (70%)",
    ]
)
_CONTEXT_MARKERS = SlashProbeMarkers(
    context_window=r"(?P<used>[\d.,]+[kKmM]?)/(?P<total>[\d.,]+[kKmM]?) tokens",
    context_bucket=r"(?P<name>[A-Za-z][\w ]*?):\s*(?P<tokens>[\d.,]+[kKmM]?) tokens",
)


# ---------------------------------------------------------------------------
# SlashProbeMarkers — default and compilation contract.


def test_default_probe_markers_match_nothing() -> None:
    markers = DEFAULT_PROBE_MARKERS
    assert markers.cost_total is None
    assert markers.cost_tokens is None
    assert markers.usage_plan is None
    assert markers.context_window is None
    assert markers.context_bucket is None


def test_compiled_helpers_are_none_without_patterns() -> None:
    markers = SlashProbeMarkers()
    assert markers.compiled_cost_total() is None
    assert markers.compiled_cost_tokens() is None
    assert markers.compiled_usage_plan() is None
    assert markers.compiled_context_window() is None
    assert markers.compiled_context_bucket() is None


def test_compiled_helpers_compile_valid_patterns() -> None:
    markers = SlashProbeMarkers(
        cost_total=r"(?P<usd>\d+)",
        cost_tokens=r"(?P<tokens>\d+)",
        usage_plan=r"(?P<percent>\d+)",
        context_window=r"(?P<used>\d+)/(?P<total>\d+)",
        context_bucket=r"(?P<name>\w+):(?P<tokens>\d+)",
    )
    assert markers.compiled_cost_total() is not None
    assert markers.compiled_cost_tokens() is not None
    assert markers.compiled_usage_plan() is not None
    assert markers.compiled_context_window() is not None
    assert markers.compiled_context_bucket() is not None


@pytest.mark.parametrize(
    ("markers", "method"),
    [
        (SlashProbeMarkers(cost_total=r"\$\d+"), "compiled_cost_total"),
        (SlashProbeMarkers(cost_tokens=r"\d+"), "compiled_cost_tokens"),
        (SlashProbeMarkers(usage_plan=r"\d+%"), "compiled_usage_plan"),
        (SlashProbeMarkers(context_window=r"(?P<used>\d+)"), "compiled_context_window"),
        (SlashProbeMarkers(context_bucket=r"(?P<name>\w+)"), "compiled_context_bucket"),
    ],
)
def test_compiled_helpers_reject_patterns_missing_a_required_group(
    markers: SlashProbeMarkers, method: str
) -> None:
    # A pattern that omits a group the extractor reads unconditionally must fail
    # loudly at construction, not crash mid-probe.
    with pytest.raises(ValueError, match="must define a"):
        getattr(markers, method)()


# ---------------------------------------------------------------------------
# /cost


def test_parse_cost_reads_dollar_and_tokens() -> None:
    report = parse_cost(_COST_SNAPSHOT, _COST_MARKERS)
    assert report == CostReport(total_cost_usd=0.4823, total_tokens=1_400_000)
    assert report is not None
    assert report.estimated is True


def test_parse_cost_with_only_a_dollar_line() -> None:
    report = parse_cost(_COST_SNAPSHOT, SlashProbeMarkers(cost_total=_COST_MARKERS.cost_total))
    assert report == CostReport(total_cost_usd=0.4823, total_tokens=None)


def test_parse_cost_with_only_a_tokens_line() -> None:
    report = parse_cost(_COST_SNAPSHOT, SlashProbeMarkers(cost_tokens=_COST_MARKERS.cost_tokens))
    assert report == CostReport(total_cost_usd=None, total_tokens=1_400_000)


def test_parse_cost_returns_none_on_no_match() -> None:
    assert parse_cost("nothing here\nor here", _COST_MARKERS) is None


def test_parse_cost_returns_none_with_default_markers() -> None:
    assert parse_cost(_COST_SNAPSHOT) is None


def test_parse_cost_skips_uncoercible_captures() -> None:
    # The lines match the patterns, but the captured groups are not numbers, so
    # both coercions return None and the whole report degrades to None rather
    # than a fabricated zero.
    markers = SlashProbeMarkers(
        cost_total=r"cost (?P<usd>\S+)",
        cost_tokens=r"tokens (?P<tokens>\S+)",
    )
    assert parse_cost("cost ABC\ntokens XYZ", markers) is None


def test_parse_cost_skips_bad_line_and_reads_a_later_good_one() -> None:
    # A matching line whose capture fails coercion is skipped, not committed: the
    # scan continues and the first coercible line wins. This pins parse_cost's
    # scan-past-bad-coercion discipline (the sibling of
    # test_parse_usage_skips_bad_line_and_reads_a_later_good_one), so a regression
    # anchoring on the first *regex* match — like parse_context's window — would
    # be caught instead of silently returning None on a garbled-first-line snapshot.
    markers = SlashProbeMarkers(cost_total=r"cost (?P<usd>\S+)")
    snapshot = "cost ABC\ncost 1.23"
    assert parse_cost(snapshot, markers) == CostReport(total_cost_usd=1.23, total_tokens=None)


def test_parse_cost_degrades_on_comma_only_captures() -> None:
    # A digit-less capture of only thousands separators ("," ) matches the number
    # grammar but strips to "" — the coercions must degrade to None (honest
    # "no reading"), not crash on float("").
    markers = SlashProbeMarkers(
        cost_total=r"cost (?P<usd>[\d,]+)",
        cost_tokens=r"tokens (?P<tokens>[\d,]+)",
    )
    assert parse_cost("cost ,\ntokens ,", markers) is None


# ---------------------------------------------------------------------------
# /usage


def test_parse_usage_reads_plan_percent() -> None:
    report = parse_usage(_USAGE_SNAPSHOT, _USAGE_MARKERS)
    assert report == UsageReport(plan_percent_used=45.0, plan_label=None)


def test_parse_usage_reads_optional_label() -> None:
    report = parse_usage(_USAGE_SNAPSHOT, _USAGE_MARKERS_WITH_LABEL)
    assert report == UsageReport(plan_percent_used=45.0, plan_label="Mon")


def test_parse_usage_returns_none_with_default_markers() -> None:
    assert parse_usage(_USAGE_SNAPSHOT) is None


def test_parse_usage_returns_none_on_no_match() -> None:
    assert parse_usage("Current week\nno headroom here", _USAGE_MARKERS) is None


def test_parse_usage_skips_uncoercible_percent_then_returns_none() -> None:
    # The first matching line has a non-numeric percent capture (skipped); no
    # other line matches, so the parse honestly degrades to None.
    markers = SlashProbeMarkers(usage_plan=r"(?P<percent>\S+) used")
    assert parse_usage("?? used", markers) is None


def test_parse_usage_skips_bad_line_and_reads_a_later_good_one() -> None:
    markers = SlashProbeMarkers(usage_plan=r"(?P<percent>\S+) used")
    snapshot = "?? used\n30 used"
    assert parse_usage(snapshot, markers) == UsageReport(plan_percent_used=30.0, plan_label=None)


def test_parse_usage_degrades_on_comma_only_percent() -> None:
    # A digit-less percent capture ("," ) strips to "" — the coercion must
    # degrade to None rather than crash on float("").
    markers = SlashProbeMarkers(usage_plan=r"(?P<percent>[\d,]+) used")
    assert parse_usage(", used", markers) is None


# ---------------------------------------------------------------------------
# /context


def test_parse_context_reads_window_and_buckets() -> None:
    report = parse_context(_CONTEXT_SNAPSHOT, _CONTEXT_MARKERS)
    assert report == ContextReport(
        used_tokens=47_200,
        total_tokens=200_000,
        buckets=(
            ContextBucket(name="System prompt", tokens=2_500),
            ContextBucket(name="MCP tools", tokens=3_200),
            ContextBucket(name="Free space", tokens=140_000),
        ),
    )


def test_parse_context_window_only() -> None:
    report = parse_context(
        _CONTEXT_SNAPSHOT, SlashProbeMarkers(context_window=_CONTEXT_MARKERS.context_window)
    )
    assert report == ContextReport(used_tokens=47_200, total_tokens=200_000, buckets=())


def test_parse_context_buckets_only() -> None:
    report = parse_context(
        _CONTEXT_SNAPSHOT, SlashProbeMarkers(context_bucket=_CONTEXT_MARKERS.context_bucket)
    )
    assert report is not None
    assert report.used_tokens is None
    assert report.total_tokens is None
    assert report.buckets == (
        ContextBucket(name="System prompt", tokens=2_500),
        ContextBucket(name="MCP tools", tokens=3_200),
        ContextBucket(name="Free space", tokens=140_000),
    )


def test_parse_context_returns_none_with_default_markers() -> None:
    assert parse_context(_CONTEXT_SNAPSHOT) is None


def test_parse_context_returns_none_on_no_match() -> None:
    assert parse_context("Context Usage\nnothing parseable", _CONTEXT_MARKERS) is None


def test_parse_context_skips_bucket_with_uncoercible_tokens() -> None:
    # The bucket line matches but its token capture ("1.2.3") fails coercion, so
    # the bucket is skipped, not zeroed; only the well-formed bucket survives.
    markers = SlashProbeMarkers(context_bucket=r"(?P<name>[A-Za-z ]+): (?P<tokens>[\d.kKmM]+)")
    snapshot = "Weird bucket: 1.2.3\nGood bucket: 5k"
    report = parse_context(snapshot, markers)
    assert report == ContextReport(
        used_tokens=None, total_tokens=None, buckets=(ContextBucket(name="Good bucket", tokens=5_000),)
    )


def test_parse_context_window_is_read_once() -> None:
    # Two window-shaped lines: only the first is taken; the second is ignored
    # because the window slots are already filled.
    markers = SlashProbeMarkers(context_window=r"(?P<used>\d+)/(?P<total>\d+) tokens")
    snapshot = "10/100 tokens\n20/200 tokens"
    report = parse_context(snapshot, markers)
    assert report == ContextReport(used_tokens=10, total_tokens=100, buckets=())


def test_parse_context_window_anchored_to_first_match_when_coercion_fails() -> None:
    # The first window-matching line is the window summary even when both its
    # captures fail to coerce: the scan must not advance to a later, unrelated
    # line and report its numbers as the window. "first match" is the first
    # *regex* match, not the first successful coercion. With both window fields
    # unread and no buckets, the parse honestly degrades to None rather than
    # fabricating the later line's 10/100.
    markers = SlashProbeMarkers(context_window=r"win (?P<used>\S+)/(?P<total>\S+)")
    snapshot = "win ABC/XYZ\nwin 10/100"
    report = parse_context(snapshot, markers)
    assert report is None


def test_parse_context_window_keeps_partial_first_match() -> None:
    # A partially-coercible first window line keeps the readable field and does
    # not advance to a later line for the missing one — the honest-degenerate
    # contract (one field None reflects an unread metric, not a fabricated value).
    markers = SlashProbeMarkers(context_window=r"win (?P<used>\S+)/(?P<total>\S+)")
    snapshot = "win 47200/XYZ\nwin 10/100"
    report = parse_context(snapshot, markers)
    assert report == ContextReport(used_tokens=47_200, total_tokens=None, buckets=())


# ---------------------------------------------------------------------------
# Numeric coercion edge cases (exercised through the public parsers).


@pytest.mark.parametrize(
    ("captured", "expected"),
    [
        ("1,234", 1_234),
        ("2.5k", 2_500),
        ("200K", 200_000),
        ("1.2M", 1_200_000),
        ("7", 7),
    ],
)
def test_token_magnitude_suffixes(captured: str, expected: int) -> None:
    markers = SlashProbeMarkers(cost_tokens=r"t (?P<tokens>\S+)")
    report = parse_cost(f"t {captured}", markers)
    assert report == CostReport(total_cost_usd=None, total_tokens=expected)


@pytest.mark.parametrize(
    ("captured", "expected"),
    [
        # Decimal suffixes whose product lands fractionally below the true
        # integer under IEEE 754 (``32.3 * 1_000 == 32299.999999999996``). The
        # earlier ``int()`` truncation reported these one too low; ``round``
        # recovers the displayed figure.
        ("32.3k", 32_300),
        ("64.1k", 64_100),
        ("4.1M", 4_100_000),
    ],
)
def test_token_magnitude_suffixes_round_inexact_floats(
    captured: str, expected: int
) -> None:
    markers = SlashProbeMarkers(cost_tokens=r"t (?P<tokens>\S+)")
    report = parse_cost(f"t {captured}", markers)
    assert report == CostReport(total_cost_usd=None, total_tokens=expected)


def test_usd_with_thousands_separator() -> None:
    markers = SlashProbeMarkers(cost_total=r"cost \$(?P<usd>[\d,.]+)")
    report = parse_cost("cost $1,234.50", markers)
    assert report == CostReport(total_cost_usd=1234.5, total_tokens=None)
