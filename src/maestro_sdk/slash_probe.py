"""Slash-command metrics extractor — a screen snapshot into a metrics report.

The ``claude`` TUI answers three slash commands with live session metrics that
:class:`~maestro_sdk.Session.run_command` can probe off a live runner:

- ``/cost`` — session token usage and a *locally estimated* dollar amount;
- ``/usage`` — token consumption and, on a subscription login, plan rate-limit
  headroom;
- ``/context`` — a token-by-token breakdown of what fills the current context
  window (system prompt, tools, MCP registrations, memory files, conversation
  history, free space, …).

These metrics live in a repainting TUI region, not the answer region, so the
output :class:`~maestro_sdk.Parser` extracts is the wrong layer: this module is a
*separate* extractor that parses a rendered :meth:`~maestro_sdk.Screen.display`
snapshot of a slash-command redraw into one of the report dataclasses
(:class:`~maestro_sdk.CostReport`, :class:`~maestro_sdk.UsageReport`,
:class:`~maestro_sdk.ContextReport`).

**Policy is configuration, not a guess.** Exactly as :class:`~maestro_sdk.Parser`
delegates its layout policy to :class:`~maestro_sdk.ParserMarkers`, this module
ships the *mechanism* (line scan, named-group extraction, numeric coercion) and
leaves the version-specific regex pack to :class:`SlashProbeMarkers`, which the
caller supplies. The default :data:`DEFAULT_PROBE_MARKERS` matches nothing, so
every parse degrades to ``None`` until a caller opts into a layout — the same
honest-degenerate discipline the parser uses (``None`` on a non-matching
snapshot, never a fabricated zero), because the TUI layout is a per-CLI-version
contract, not a stable one.

This module is pure: no I/O, no sleep, no subprocess. It is a function of the
snapshot text and the markers alone, which makes it exhaustively table-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from maestro_sdk.types import ContextBucket, ContextReport, CostReport, UsageReport

# A leading number with optional thousands separators and decimals, plus an
# optional k/m magnitude suffix: "1,234", "2.5k", "200K", "1.2M". Shared by the
# token and dollar coercions so every numeric read is parsed one way.
_NUMBER = r"[\d,]+(?:\.\d+)?"
_TOKEN_RE = re.compile(rf"^({_NUMBER})\s*([kKmM]?)$")
_USD_RE = re.compile(rf"^\$?({_NUMBER})$")
_PERCENT_RE = re.compile(rf"^({_NUMBER})\s*%?$")
_MAGNITUDE = {"": 1, "k": 1_000, "m": 1_000_000}


def _parse_token_count(text: str) -> int | None:
    """Coerce a token figure (``"2.5k"``, ``"200K"``, ``"1,234"``) to an int.

    Returns ``None`` on anything the shared number grammar does not match, so a
    caller that captured a non-numeric group degrades to "no reading" rather
    than crashing or fabricating a count.
    """
    match = _TOKEN_RE.match(text.strip())
    if match is None:
        return None
    # ``_NUMBER`` permits a capture of only thousands separators (``","``), which
    # strips to ``""`` and would crash ``float("")``. Treat a digit-less capture
    # as uncoercible — ``None``, per the docstring — not an exception.
    cleaned = match.group(1).replace(",", "")
    if not cleaned:
        return None
    number = float(cleaned)
    # ``round``, not ``int``: several common decimal suffixes land fractionally
    # below the true integer under IEEE 754 (e.g. ``32.3 * 1_000`` is
    # ``32299.999999999996``), so truncation toward zero would silently report a
    # count one too low. Rounding to nearest recovers the displayed figure.
    return round(number * _MAGNITUDE[match.group(2).lower()])


def _parse_usd(text: str) -> float | None:
    """Coerce a dollar figure (``"$0.12"``, ``"1,234.50"``) to a float, or ``None``."""
    match = _USD_RE.match(text.strip())
    if match is None:
        return None
    cleaned = match.group(1).replace(",", "")
    if not cleaned:
        return None
    return float(cleaned)


def _parse_percent(text: str) -> float | None:
    """Coerce a percentage figure (``"45%"``, ``"45"``) to a float, or ``None``."""
    match = _PERCENT_RE.match(text.strip())
    if match is None:
        return None
    cleaned = match.group(1).replace(",", "")
    if not cleaned:
        return None
    return float(cleaned)


def _compile(pattern: str | None, *required: str) -> re.Pattern[str] | None:
    """Compile ``pattern`` once, asserting every ``required`` named group exists.

    Returns ``None`` when ``pattern`` is unset (the field's degenerate
    matches-nothing state). A pattern that omits a group the extractor reads
    unconditionally is a misconfiguration that must fail loudly at construction
    rather than crash mid-probe — the same contract as
    :meth:`ParserMarkers.compiled_tool_call`.
    """
    if pattern is None:
        return None
    compiled = re.compile(pattern)
    missing = [name for name in required if name not in compiled.groupindex]
    if missing:
        raise ValueError(f"pattern must define a (?P<{missing[0]}>...) group")
    return compiled


@dataclass(frozen=True)
class SlashProbeMarkers:
    """Layout policy: the per-CLI-version regexes that read a metrics redraw.

    Mirrors :class:`~maestro_sdk.ParserMarkers`. Every field defaults to
    "matches nothing," leaving every parse in its degenerate ``None`` state until
    a caller opts into a layout. Each pattern is matched with :meth:`re.Pattern.search`
    against each *stripped* snapshot line, so leading TUI glyphs (``⛁``, ``▐``)
    need not be spelled out unless the pattern chooses to anchor on them; the
    named groups decide what is extracted.

    - ``cost_total`` — a ``/cost`` line exposing a ``usd`` group (the estimated
      session dollar amount).
    - ``cost_tokens`` — a ``/cost`` line exposing a ``tokens`` group (the session
      token total).
    - ``usage_plan`` — a ``/usage`` line exposing a ``percent`` group (plan
      rate-limit headroom used) and an optional ``label`` group (the window
      name, e.g. the reset cadence).
    - ``context_window`` — a ``/context`` line exposing ``used`` and ``total``
      groups (the window's used and total token counts).
    - ``context_bucket`` — a ``/context`` line exposing ``name`` and ``tokens``
      groups; matched against every line, so each bucket row is captured.
    """

    cost_total: str | None = None
    cost_tokens: str | None = None
    usage_plan: str | None = None
    context_window: str | None = None
    context_bucket: str | None = None

    def compiled_cost_total(self) -> re.Pattern[str] | None:
        return _compile(self.cost_total, "usd")

    def compiled_cost_tokens(self) -> re.Pattern[str] | None:
        return _compile(self.cost_tokens, "tokens")

    def compiled_usage_plan(self) -> re.Pattern[str] | None:
        return _compile(self.usage_plan, "percent")

    def compiled_context_window(self) -> re.Pattern[str] | None:
        return _compile(self.context_window, "used", "total")

    def compiled_context_bucket(self) -> re.Pattern[str] | None:
        return _compile(self.context_bucket, "name", "tokens")


DEFAULT_PROBE_MARKERS = SlashProbeMarkers()
"""Empty markers: every parse returns ``None`` (no layout configured)."""


def parse_cost(
    snapshot: str, markers: SlashProbeMarkers = DEFAULT_PROBE_MARKERS
) -> CostReport | None:
    """Parse a ``/cost`` redraw into a :class:`~maestro_sdk.CostReport`, or ``None``.

    Scans the snapshot for the first line matching each configured pattern and
    coerces its capture. Returns ``None`` when no field could be read (no markers,
    a non-matching layout, or captures that failed numeric coercion), so an
    all-``None`` report is never produced.
    """
    total_re = markers.compiled_cost_total()
    tokens_re = markers.compiled_cost_tokens()
    usd: float | None = None
    tokens: int | None = None
    for line in snapshot.split("\n"):
        stripped = line.strip()
        if usd is None and total_re is not None:
            match = total_re.search(stripped)
            if match is not None:
                usd = _parse_usd(match.group("usd"))
        if tokens is None and tokens_re is not None:
            match = tokens_re.search(stripped)
            if match is not None:
                tokens = _parse_token_count(match.group("tokens"))
    if usd is None and tokens is None:
        return None
    return CostReport(total_cost_usd=usd, total_tokens=tokens)


def parse_usage(
    snapshot: str, markers: SlashProbeMarkers = DEFAULT_PROBE_MARKERS
) -> UsageReport | None:
    """Parse a ``/usage`` redraw into a :class:`~maestro_sdk.UsageReport`, or ``None``.

    Returns ``None`` when the plan-headroom line is absent or its percentage
    capture failed coercion (a non-subscription login or a layout skew), rather
    than a fabricated headroom.
    """
    plan_re = markers.compiled_usage_plan()
    if plan_re is None:
        return None
    for line in snapshot.split("\n"):
        match = plan_re.search(line.strip())
        if match is None:
            continue
        percent = _parse_percent(match.group("percent"))
        if percent is None:
            continue
        label = match.groupdict().get("label")
        return UsageReport(plan_percent_used=percent, plan_label=label)
    return None


def parse_context(
    snapshot: str, markers: SlashProbeMarkers = DEFAULT_PROBE_MARKERS
) -> ContextReport | None:
    """Parse a ``/context`` redraw into a :class:`~maestro_sdk.ContextReport`, or ``None``.

    Reads the window's used/total counts (first match) and every bucket row.
    Bucket rows whose token capture fails coercion are skipped, not zeroed.
    Returns ``None`` when neither the window line nor any bucket could be read.
    """
    window_re = markers.compiled_context_window()
    bucket_re = markers.compiled_context_bucket()
    used: int | None = None
    total: int | None = None
    window_found = False
    buckets: list[ContextBucket] = []
    for line in snapshot.split("\n"):
        stripped = line.strip()
        # "first match" means the first *regex* match, not the first successful
        # coercion: a dedicated flag consumes the window slot on the first line
        # the pattern hits. Keying off ``used``/``total`` instead would re-fire
        # the scan when both captures fail to coerce, grabbing a later, unrelated
        # line as the window summary — and asymmetrically not re-fire when only
        # one coerced. The flag keeps a partial reading (one field None) while
        # never advancing past the first window line.
        if not window_found and window_re is not None:
            match = window_re.search(stripped)
            if match is not None:
                window_found = True
                used = _parse_token_count(match.group("used"))
                total = _parse_token_count(match.group("total"))
        if bucket_re is not None:
            match = bucket_re.search(stripped)
            if match is not None:
                tokens = _parse_token_count(match.group("tokens"))
                if tokens is not None:
                    buckets.append(ContextBucket(name=match.group("name").strip(), tokens=tokens))
    if used is None and total is None and not buckets:
        return None
    return ContextReport(used_tokens=used, total_tokens=total, buckets=tuple(buckets))
