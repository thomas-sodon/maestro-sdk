"""Public data contract.

These dataclasses are the contract between the user of :mod:`maestro_sdk` and
the SDK. The fields are shaped to mirror :class:`maestro.sdk.QueryResult` so a
future Maestro adapter is a thin re-pack rather than a translation layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class QueryOptions:
    """Per-call options passed into :meth:`AsyncClient.query`.

    ``cwd`` is honoured today (the default runner factory spawns the CLI in it).
    ``system`` is **not yet threaded** by :class:`AsyncClient`: turning it into a
    CLI invocation is command-building, which is deferred to the Maestro selection
    seam along with session continuity, so setting it currently has no effect.
    Both timeouts bound the read stream (see :class:`AsyncClient`).

    ``session_id`` pins a brand-new session to a caller-chosen id at spawn
    (proposal 081): when set, a one-shot :meth:`AsyncClient.query` spawns
    ``claude --session-id <id>`` so the id is known *a priori* — and that pinned id
    is reported on :attr:`QueryResult.session_id` even with no on-screen
    :attr:`~maestro_sdk.ClientMarkers.session_id` marker (the pinned id is the
    *floor*; a matching marker still overrides it for that turn). ``None`` (the
    default) leaves the spawn command and the result id exactly as before. For a
    persistent :class:`~maestro_sdk.Session`, pin via
    ``AsyncClient.create_session(session_id=…)`` instead, which also reuses the id
    for the ``--resume`` respawn.

    ``complete_on`` is an optional in-stream early-exit predicate evaluated against
    the accumulated answer text after each output snapshot. When it returns truthy,
    the read loop stops at that snapshot and the turn returns the text gathered so
    far without waiting for end-of-stream or the idle marker — the streaming
    capability mirroring ``maestro.sdk``'s ``complete_on``, single-argument here
    because a repainting TUI exposes only text (no message object).

    ``idle_quiescence_seconds`` opts a turn into **output-quiescence completion**
    (proposal 081): when set (and no :attr:`~maestro_sdk.ClientMarkers.idle` marker
    competes for the role), a turn is complete once the rendered screen has stopped
    changing for this many seconds while the child is still alive — a layout-free
    done-signal for a persistent CLI that never EOFs and paints no matchable idle
    prefix. Quiescence is measured on the *rendered snapshot*
    (:meth:`maestro_sdk.Screen.display`), not raw bytes, so a cursor blink or an
    identical repaint counts as quiet while a spinner/token-tick repaint keeps the
    turn live. ``None`` (the default) leaves the read loop exactly as before: EOF,
    a configured idle marker, or ``complete_on`` are the only turn-complete signals.
    Tune it above the model's inter-repaint cadence and below
    ``stream_idle_timeout_seconds``, which remains the hard outer bound the window
    sits inside.
    """

    cwd: str | None = None
    system: str | None = None
    session_id: str | None = None
    startup_timeout_seconds: float = 30.0
    stream_idle_timeout_seconds: float = 10.0
    idle_quiescence_seconds: float | None = None
    complete_on: Callable[[str], bool] | None = None


@dataclass(frozen=True)
class QueryUsage:
    """Best-effort token / cost accounting.

    The CLI does not always expose this; when it is unavailable the field on
    :class:`QueryResult` is ``None`` rather than zeroed — silently faking usage
    would mislead callers building dashboards on top of it. All four counts
    are required: there is no honest default for "did not measure," so
    constructing a :class:`QueryUsage` is itself an assertion that you
    measured something.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


@dataclass(frozen=True)
class QueryTiming:
    """Observed timing for one query, so callers can verify that human
    pacing actually happened."""

    send_ms: float
    stream_ms: float
    heartbeat_ms: float

    @property
    def total_ms(self) -> float:
        return self.send_ms + self.stream_ms + self.heartbeat_ms


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation observed in the CLI's output stream."""

    name: str
    args: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    """The result of one query.

    ``session_id`` is the CLI-reported session id captured from the output when a
    :class:`~maestro_sdk.ClientMarkers.session_id` marker is configured, or the
    spawn-pinned id when one was pinned via :attr:`QueryOptions.session_id` /
    ``AsyncClient.create_session(session_id=...)`` (proposal 081 — the pinned id is
    the *floor*, so an on-screen marker still overrides it for that turn), or
    ``None`` when neither is configured and none matched. A caller persists it to
    resume the conversation later (``AsyncClient.create_session(resume=...)``).

    ``idle_marker_missed`` is the layout-drift sentinel (proposal 075): ``True``
    when a non-empty :attr:`~maestro_sdk.ClientMarkers.idle` was configured but
    the turn still ended on EOF without the marker ever matching — the signature
    of a CLI upgrade silently changing the TUI layout. The turn itself is valid
    (EOF is the honest fallback done-signal); the flag makes the degradation
    observable instead of silent. Always ``False`` for an unconfigured client,
    where EOF is the *expected* done-signal rather than a miss.
    """

    response: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: QueryUsage | None = None
    session_id: str | None = None
    raw_screen: str = ""
    timing: QueryTiming | None = None
    idle_marker_missed: bool = False


@dataclass(frozen=True)
class StreamDelta:
    """One slice of streamed output."""

    text: str
    kind: Literal["thinking", "answer"]
    screen_snapshot: str


@dataclass(frozen=True)
class CostReport:
    """Parsed ``/cost`` slash-command snapshot — a *local estimate*.

    The TUI computes this figure from token counts times published rates, not from
    authoritative billing, so :attr:`estimated` is ``True`` by construction: a
    consumer must treat it as a cross-check, never as the spend of record. Every
    measured field is ``| None`` because a given CLI layout may surface some
    lines and not others; the extractor returns the whole report as ``None``
    when *nothing* matched, so an all-``None`` report never stands for "measured
    zero." See :mod:`maestro_sdk.slash_probe`.
    """

    total_cost_usd: float | None = None
    total_tokens: int | None = None
    estimated: bool = True


@dataclass(frozen=True)
class UsageReport:
    """Parsed ``/usage`` slash-command snapshot — subscription plan headroom.

    Reports the rate-limit headroom a subscription login exposes (and neither
    the API response nor the transcript sidecar does). Both fields are ``| None``
    so a non-subscription login or a layout miss degrades to "no reading" rather
    than a fabricated percentage. See :mod:`maestro_sdk.slash_probe`.
    """

    plan_percent_used: float | None = None
    plan_label: str | None = None


@dataclass(frozen=True)
class ContextBucket:
    """One row of a ``/context`` breakdown: a named slice and its token count."""

    name: str
    tokens: int


@dataclass(frozen=True)
class ContextReport:
    """Parsed ``/context`` slash-command snapshot — context-window composition.

    Captures how the *current* context window is filled: the window's used and
    total token counts plus a per-bucket breakdown (system prompt, tools, MCP
    registrations, memory files, conversation history, free space, …). This is
    the only runtime surface that measures realized context composition for the
    CLI path. ``used_tokens`` / ``total_tokens`` are ``| None`` on a layout
    miss; ``buckets`` defaults to the empty tuple — the honest "no buckets
    parsed" — and the extractor returns the whole report as ``None`` when
    nothing matched. See :mod:`maestro_sdk.slash_probe`.
    """

    used_tokens: int | None = None
    total_tokens: int | None = None
    buckets: tuple[ContextBucket, ...] = ()
