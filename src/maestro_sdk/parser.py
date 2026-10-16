"""Output parser — a stream of screen snapshots into deltas and a result.

The :class:`Screen` emulator (previous slice) turns raw pty bytes into a stable
picture of what the terminal would show. This module is the next layer up: it
consumes a *stream* of those pictures (one :meth:`Screen.display` string per
feed) and answers two questions the rest of the SDK cares about:

1. **What is new since the last snapshot?** The CLI repaints; most of a snapshot
   is identical to the one before it. :meth:`Parser.feed` returns only the text
   that was *added*, packaged as :class:`StreamDelta` objects, so a caller can
   surface streaming output without re-emitting the whole screen each tick.
2. **What did the run produce?** Across all feeds the parser accumulates the
   answer text into :attr:`Parser.response` and collects every tool invocation
   it recognised into :attr:`Parser.tool_calls`.

**Policy is configuration, not a guess.** A repainting TUI carries no
machine-readable "this is reasoning / this is the answer / this is a tool call"
signal once SGR colour is stripped — only the *layout* survives, and that layout
is specific to a CLI version. So this module ships the *mechanism* (a region
state machine, incremental diffing, tool extraction) and leaves the *markers*
that drive it to :class:`ParserMarkers`, which the caller supplies. This mirrors
how :class:`Cadence` owns every timing constant: re-tune by passing different
markers, never by editing this file.

The default :class:`ParserMarkers` is empty: with no markers, every line is
treated as answer text and no tool calls are extracted. That is the honest
degenerate behaviour — "give me everything as the answer" — rather than a
fragile guess at one CLI version's glyphs. Configure markers to enable
thinking/answer discrimination and tool-call extraction; an example tuned for
Claude Code's layout lives in the package README.

This module is pure: no I/O, no sleep, no subprocess. It is a function of the
snapshot stream alone, which makes it exhaustively table-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from maestro_sdk.types import StreamDelta, ToolCall

_Region = Literal["thinking", "answer"]
_REGION_ORDER: tuple[_Region, ...] = ("thinking", "answer")


@dataclass(frozen=True)
class ParserMarkers:
    """Layout policy: how to classify the lines of a screen snapshot.

    All matching is against each line *stripped* of leading and trailing
    whitespace, so indentation in the TUI does not defeat a marker. Every field
    defaults to "matches nothing," which leaves the parser in its degenerate
    everything-is-answer mode until a caller opts into richer behaviour.

    - ``thinking_open`` / ``thinking_close``: a line whose stripped form
      *starts with* any of these strings opens (resp. closes) the thinking
      region. The marker line itself is consumed, not emitted. Between an open
      and the next close, lines are classified ``"thinking"``; everywhere else
      they are ``"answer"``.
    - ``tool_call``: a regular expression matched against the stripped line. A
      match records a :class:`ToolCall` and the line is not emitted as text. The
      pattern must expose a ``name`` group (enforced by
      :meth:`compiled_tool_call`); an optional ``args`` group is stored verbatim
      under the ``"raw"`` key (the CLI surfaces no structured arguments, so
      faking a parse would mislead callers).
    - ``chrome``: a line whose stripped form starts with any of these is dropped
      entirely — neither emitted nor counted. Use it for input-box borders,
      shortcut hints, spinners, and other non-content furniture. Empty by
      default so nothing is silently discarded.
    """

    thinking_open: tuple[str, ...] = ()
    thinking_close: tuple[str, ...] = ()
    tool_call: str | None = None
    chrome: tuple[str, ...] = ()

    def compiled_tool_call(self) -> re.Pattern[str] | None:
        """Compile :attr:`tool_call` once, or ``None`` if unset.

        Raises :class:`ValueError` if the pattern omits the required ``name``
        group: extraction in :meth:`Parser._record_tool_call` reads ``name``
        unconditionally, so a missing group is a misconfiguration that must fail
        loudly at construction rather than crash mid-stream during a query.
        """
        if self.tool_call is None:
            return None
        pattern = re.compile(self.tool_call)
        if "name" not in pattern.groupindex:
            raise ValueError("tool_call pattern must define a (?P<name>...) group")
        return pattern


DEFAULT_MARKERS = ParserMarkers()
"""Empty markers: everything is answer text, no tool calls extracted."""


def _common_prefix_len(old: str, new: str) -> int:
    """Length of the longest shared leading run of ``old`` and ``new``."""
    limit = min(len(old), len(new))
    i = 0
    while i < limit and old[i] == new[i]:
        i += 1
    return i


class Parser:
    """Folds a stream of screen snapshots into deltas, a response, and tool calls.

    Feed each new snapshot (a :meth:`Screen.display` string) to :meth:`feed`;
    it returns the :class:`StreamDelta` objects for text revealed since the
    previous feed. Read the run's product from :attr:`response` and
    :attr:`tool_calls` once the stream ends.

    Region classification is recomputed from scratch for every snapshot — a
    repaint redraws the whole screen, so there is no region state to carry
    between feeds, only the per-region *text* needed to diff against the next
    snapshot.

    **Incremental diffing** uses the longest common prefix of a region's old and
    new text: the suffix beyond it is the delta. This is exact for append-only
    streaming (the common case) and a no-op for a repaint to identical content.
    It does *not* reconstruct text that has scrolled off the top of the screen;
    a query whose output exceeds the pty height will lose the scrolled portion
    from :attr:`response`. Size the pty tall enough (``Cadence.winsize``) for the
    expected output, or treat very long answers as out of scope for this
    backend — the same observability caveat the proposal records.
    """

    def __init__(self, markers: ParserMarkers = DEFAULT_MARKERS) -> None:
        self._markers = markers
        self._tool_re = markers.compiled_tool_call()
        # Last snapshot's text per region, for prefix-diffing the next snapshot.
        self._prev: dict[_Region, str] = {"thinking": "", "answer": ""}
        # Accumulated emitted text per region; answer is the response.
        self._acc: dict[_Region, str] = {"thinking": "", "answer": ""}
        self._tool_calls: list[ToolCall] = []
        # Tool-call lines already recorded, so a repaint does not double-count.
        self._seen_tool_lines: set[str] = set()

    @property
    def response(self) -> str:
        """The accumulated answer text — exactly the concatenation of every
        ``"answer"`` delta this parser has emitted."""
        return self._acc["answer"]

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool invocations recognised across the stream, in first-seen order."""
        return list(self._tool_calls)

    def feed(self, snapshot: str) -> list[StreamDelta]:
        """Advance by one screen snapshot; return deltas revealed since the last.

        Thinking deltas precede answer deltas within a single feed, matching the
        order a turn produces them (reason, then answer). A region that did not
        grow yields no delta.
        """
        regions = self._classify(snapshot)
        deltas: list[StreamDelta] = []
        for region in _REGION_ORDER:
            new_text = regions[region]
            shared = _common_prefix_len(self._prev[region], new_text)
            added = new_text[shared:]
            # Only update _prev when the snapshot genuinely advances or
            # diverges. If the new text is a strict prefix of the previous
            # text (a transient shrink during repaint), keep the longer
            # snapshot — otherwise a later regrowth would re-emit the tail
            # we already streamed and duplicate it in the response.
            if added or not self._prev[region].startswith(new_text):
                self._prev[region] = new_text
            if added:
                self._acc[region] += added
                deltas.append(StreamDelta(text=added, kind=region, screen_snapshot=snapshot))
        return deltas

    def _classify(self, snapshot: str) -> dict[_Region, str]:
        """Split one snapshot into per-region text, recording any tool calls.

        Walks the snapshot top to bottom through a two-state region machine.
        Chrome and tool-call lines are consumed without contributing to either
        region's text; thinking open/close markers flip the region and are
        themselves consumed.
        """
        region: _Region = "answer"
        lines: dict[_Region, list[str]] = {"thinking": [], "answer": []}
        for raw_line in snapshot.split("\n"):
            stripped = raw_line.strip()
            if self._starts_with_any(stripped, self._markers.chrome):
                continue
            if self._record_tool_call(stripped):
                continue
            if region == "answer" and self._starts_with_any(stripped, self._markers.thinking_open):
                region = "thinking"
                continue
            if region == "thinking" and self._starts_with_any(stripped, self._markers.thinking_close):
                region = "answer"
                continue
            lines[region].append(raw_line)
        return {
            "thinking": "\n".join(lines["thinking"]).rstrip(),
            "answer": "\n".join(lines["answer"]).rstrip(),
        }

    def _record_tool_call(self, stripped: str) -> bool:
        """Record a tool call if ``stripped`` matches the pattern; report whether it did.

        Deduped on the exact line text so a repainted tool line is recorded once.
        """
        if self._tool_re is None:
            return False
        match = self._tool_re.match(stripped)
        if match is None:
            return False
        if stripped not in self._seen_tool_lines:
            self._seen_tool_lines.add(stripped)
            groups = match.groupdict()
            raw_args = groups.get("args")
            args: dict[str, object] = {"raw": raw_args} if raw_args else {}
            self._tool_calls.append(ToolCall(name=groups["name"], args=args))
        return True

    @staticmethod
    def _starts_with_any(stripped: str, markers: tuple[str, ...]) -> bool:
        return any(stripped.startswith(marker) for marker in markers)
