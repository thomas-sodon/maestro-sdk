"""Public client — composes the runner, screen, parser, and approvals into a query.

This is the top of the stack: the slices below it (pty runner, screen emulator,
output parser, approval handler) each do one thing; :class:`AsyncClient` wires
them into the single operation the rest of the world wants — *type a prompt at
human cadence, watch the CLI answer, hand back a result.*

A query runs as a strict sequence, each step timed so a caller can prove the
pacing happened (:class:`QueryTiming`):

1. **Spawn.** Build a runner for the command and ``start()`` it.
2. **Send.** Type the prompt one keystroke at a time through the runner (so
   :class:`Cadence` paces it) and press Enter.
3. **Stream.** Drain the runner chunk by chunk into the :class:`Screen`, render
   a snapshot, and feed it to the :class:`Parser`. On every snapshot, check for
   the whole-screen conditions the client must act on — a rate-limit or
   expired-session notice (raise), a tool-approval modal (resolve via the
   :class:`ApprovalHandler`), an idle marker (turn complete).
4. **Heartbeat.** Wait :meth:`Cadence.heartbeat_for` the answer length before
   returning, mirroring an operator reading the reply before typing again.

**Two timeouts guard the stream** (the safety nets, never the normal path).
The first read after sending is bounded by ``startup_timeout_seconds`` — no
output at all within it means the CLI never came up, a ``"timeout"``-category
:class:`CLITimeoutError`. Every later read is bounded by
``stream_idle_timeout_seconds`` — a silent stream that has not completed the
turn is a wedged subprocess, a ``"stream_stall"`` :class:`CLITimeoutError`.
Either way the runner is always torn down in a ``finally`` so a timed-out,
cancelled, or rejected query leaks neither child nor descriptor.

**Done-detection is EOF by default; an idle marker enables interactive turns.**
With no ``idle`` marker configured, end-of-stream (the CLI process exiting) is
the sole turn-complete signal — correct for a one-shot invocation. A persistent
interactive CLI never EOFs between turns, so configure :attr:`ClientMarkers.idle`
(the input prompt redrawing) to mark turn completion; without it such a CLI
would stall. This is the same *mechanism + configuration* split the parser and
approval handler use: the client ships the machinery and leaves the
CLI-version-specific glyphs to :class:`ClientMarkers`.

**Observability is honest, not faked.** The CLI surfaces no token counts through
a repainting TUI, so :attr:`QueryResult.usage` is ``None`` (never zero-filled —
a zero would read as "measured zero," which is a lie). :attr:`QueryResult.timing`
*is* populated, from monotonic clocks, because the client measures it directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from maestro_sdk._logging import get_logger
from maestro_sdk.approvals import (
    DEFAULT_APPROVAL_MARKERS,
    ApprovalHandler,
    ApprovalMarkers,
    ApprovalPolicy,
    ApprovalRequest,
)
from maestro_sdk.cadence import Cadence, InstantCadence
from maestro_sdk.errors import CLIExitedError, CLITimeoutError, RateLimitError, SessionExpiredError
from maestro_sdk.parser import DEFAULT_MARKERS, Parser, ParserMarkers
from maestro_sdk.pty_runner import PtyRunner
from maestro_sdk.screen import Screen
from maestro_sdk.session import Session
from maestro_sdk.types import QueryOptions, QueryResult, QueryTiming

_log = get_logger("client")

# The keystroke that submits the typed prompt (and any approval choice). Carriage
# return rather than newline because that is what a terminal delivers on Enter.
_SUBMIT_KEY = "\r"

# The CLI flag that reconnects a spawn to a prior server-side transcript. A
# session built with ``resume`` (or one that has captured a CLI-reported id)
# prepends ``--resume <id>`` to the spawn command so the child continues the
# conversation rather than starting fresh.
_RESUME_FLAG = "--resume"

# The CLI flag that pins a brand-new session to a caller-chosen id at spawn
# (proposal 081). A one-shot query (via :attr:`QueryOptions.session_id`) or a
# session (via ``create_session(session_id=…)``) prepends ``--session-id <id>``
# so the id is known *a priori* rather than scraped off a screen that may never
# paint it; a session respawn then reconnects with ``_RESUME_FLAG`` instead, so
# the two flags are mutually exclusive on any single spawn (``--resume`` wins).
_SESSION_ID_FLAG = "--session-id"


class _Runner(Protocol):
    """The runner surface the client drives — satisfied structurally by
    :class:`PtyRunner` and by the fake runner used in tests.

    A :class:`typing.Protocol` rather than a base class so neither the real
    runner nor a test double has to inherit anything: the client touches exactly
    these six members and nothing else.
    """

    @property
    def winsize(self) -> tuple[int, int]: ...
    @property
    def returncode(self) -> int | None: ...
    async def start(self) -> None: ...
    async def send_text(self, text: str, *, prev: str | None = None) -> str | None: ...
    async def aread_chunk(self) -> bytes: ...
    async def aclose(self) -> None: ...


RunnerFactory = Callable[[Sequence[str], QueryOptions], _Runner]
"""Builds a runner for a command and the call's options (the pty-spawn seam)."""


@dataclass(frozen=True)
class ClientMarkers:
    """Layout policy for whole-screen conditions the client acts on.

    Mirrors :class:`maestro_sdk.ParserMarkers` and
    :class:`maestro_sdk.ApprovalMarkers`: each field is a tuple of line prefixes
    matched against every *stripped* line of a snapshot, and each defaults to
    "matches nothing" so an unconfigured client detects none of these conditions
    and treats EOF as the only turn-complete signal.

    - ``idle``: a line starting with any of these means the CLI finished the turn
      and is waiting for input again — the interactive turn-complete signal.
      Without it only EOF ends a turn, so a persistent CLI needs an idle marker
      or every query stalls.
    - ``rate_limit``: a line starting with any of these is a rate-limit notice →
      :class:`RateLimitError` (category ``"rate_limit"``).
    - ``session_expired``: a line starting with any of these is an expired-session
      notice → :class:`SessionExpiredError` (category ``"stale_session"``).
    - ``session_id``: a line starting with any of these carries the CLI-reported
      session id as the remainder of the *stripped* line; the client extracts it
      onto :attr:`QueryResult.session_id` so a later turn can ``--resume`` it.
      Empty by default (no id is extracted), since the id glyph is CLI-version
      specific — the same mechanism-plus-configuration split as the other fields.
      Even when empty, a spawn-pinned id (proposal 081, via
      :attr:`QueryOptions.session_id` / ``create_session(session_id=…)``) still
      populates the result as the *floor* an on-screen capture here overrides.
    """

    idle: tuple[str, ...] = ()
    rate_limit: tuple[str, ...] = ()
    session_expired: tuple[str, ...] = ()
    session_id: tuple[str, ...] = ()


DEFAULT_CLIENT_MARKERS = ClientMarkers()
"""Empty markers: no on-screen condition is detected, so EOF is the only done-signal."""

_DEFAULT_OPTIONS = QueryOptions()
_DEFAULT_COMMAND: tuple[str, ...] = ("claude",)


class AsyncClient:
    """Drives the ``claude`` CLI through a pty at human cadence.

    Construct once with a :class:`Cadence` and the layout markers tuned to your
    CLI version; then call :meth:`query` for a one-shot turn or
    :meth:`create_session` for a persistent :class:`Session`. A one-shot
    :meth:`query` owns no process between calls — it spawns, drives, and reaps its
    own runner — so a single client is safe to reuse across many queries; a
    :class:`Session` keeps one runner alive across its turns and owns that
    process until closed.
    """

    def __init__(
        self,
        cadence: Cadence,
        *,
        command: Sequence[str] = _DEFAULT_COMMAND,
        options: QueryOptions = _DEFAULT_OPTIONS,
        parser_markers: ParserMarkers = DEFAULT_MARKERS,
        approval_markers: ApprovalMarkers = DEFAULT_APPROVAL_MARKERS,
        approval_policy: ApprovalPolicy | None = None,
        client_markers: ClientMarkers = DEFAULT_CLIENT_MARKERS,
        env: Mapping[str, str] | None = None,
        runner_factory: RunnerFactory | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must be a non-empty sequence")
        self._cadence = cadence
        self._command = list(command)
        self._options = options
        self._parser_markers = parser_markers
        self._approval = ApprovalHandler(approval_markers, approval_policy)
        self._markers = client_markers
        self._env = dict(env) if env is not None else None
        self._runner_factory = runner_factory or self._default_runner_factory
        # Layout-drift sentinel (proposal 075): one-shot latch so the
        # idle-marker-missed WARNING fires once per client, not once per turn —
        # the drift is a property of the CLI layout, not of any single turn.
        self._idle_drift_warned = False

    # -- public surface ----------------------------------------------------

    async def query(
        self, prompt: str, *, options: QueryOptions | None = None, session_id: str | None = None
    ) -> QueryResult:
        """Run one self-contained turn: spawn, type ``prompt``, read, tear down.

        ``options`` overrides the client's defaults for this call only.

        ``session_id`` pins a caller-chosen id at spawn (proposal 081): when set,
        the runner spawns ``claude --session-id <id>`` and the returned
        :attr:`QueryResult.session_id` is that id even with no
        :attr:`ClientMarkers.session_id` marker configured — the pinned id is the
        floor (an on-screen capture, when one matches, still overrides it). ``None``
        (the default) leaves the spawn command and the result id exactly as before.
        Equivalent to passing ``options=QueryOptions(session_id=…)`` — if both are
        set, ``session_id`` wins (the explicit parameter overrides the options field).
        """
        opts = options or self._options
        if session_id is not None:
            opts = replace(opts, session_id=session_id)
        return await self._execute(prompt, opts)

    def create_session(
        self,
        initial_prompt: str | None = None,
        *,
        resume: str | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Open a persistent :class:`Session` bound to this client.

        Unlike :meth:`query` — which spawns, drives, and reaps its own runner per
        call — a session keeps one runner alive across turns (see
        :class:`maestro_sdk.session.Session`), so configure
        :attr:`ClientMarkers.idle` (or an
        :attr:`QueryOptions.idle_quiescence_seconds` window) before opening one
        against a persistent interactive CLI or every turn after the first would
        stall.

        ``resume`` is a CLI session id: when set, the session spawns
        ``claude --resume <id>`` so the child reconnects to that prior transcript.
        ``session_id`` instead *pins a fresh* id (proposal 081): the first spawn
        runs ``claude --session-id <id>`` (creating the session under that id),
        every turn's :attr:`QueryResult.session_id` reports it with no on-screen
        marker (the floor), and any respawn reconnects with ``--resume <id>``.
        ``resume`` and ``session_id`` are mutually exclusive — reconnecting to an
        existing transcript and creating a new pinned one are contradictory — and
        passing both raises :class:`ValueError`.
        ``initial_prompt``, when set, is sent as the session's first turn (ahead
        of the caller's first :meth:`Session.query` prompt), mirroring the
        Anthropic provider's connect-time seed message. All default to ``None``,
        which leaves the one-shot-equivalent spawn untouched.
        """
        return Session(self, initial_prompt=initial_prompt, resume=resume, session_id=session_id)

    # -- runner construction ----------------------------------------------

    def _default_runner_factory(self, command: Sequence[str], options: QueryOptions) -> _Runner:
        return PtyRunner(command, self._cadence, cwd=options.cwd, env=self._env)

    def _spawn(
        self, options: QueryOptions, *, resume: str | None = None, session_id: str | None = None
    ) -> _Runner:
        """Build the spawn command and hand it to the runner factory.

        When ``resume`` is set, ``--resume <id>`` is appended so the child
        reconnects to that transcript. Otherwise, when ``session_id`` is set, a
        ``--session-id <id>`` pins a *fresh* session to that caller-chosen id at
        spawn (proposal 081). ``resume`` wins when both are set: a session's first
        spawn creates the pinned id with ``--session-id`` and its respawns
        reconnect with ``--resume`` once an id has been captured/floored, so the
        two never need to coexist on one command. With neither set the bare
        configured command is used — the one-shot :meth:`_execute` never resumes,
        and an unpinned :class:`Session` spawns fresh.
        """
        command = list(self._command)
        if resume is not None:
            command += [_RESUME_FLAG, resume]
        elif session_id is not None:
            command += [_SESSION_ID_FLAG, session_id]
        return self._runner_factory(command, options)

    # -- the query machine -------------------------------------------------

    async def _execute(self, prompt: str, options: QueryOptions) -> QueryResult:
        self._warn_if_instant()
        runner = self._spawn(options, session_id=options.session_id)
        try:
            await runner.start()
            result, _ended_on_eof, _exited_early = await self._drive_turn(
                runner, prompt, options, pinned_session_id=options.session_id
            )
            return result
        finally:
            # The sole teardown site for a one-shot query: a normal return, a
            # raised timeout/condition, or a cancellation all reach it, so no path
            # leaves the child or the master fd alive.
            await runner.aclose()

    def _warn_if_instant(self) -> None:
        """Log once, loudly, if driving a real CLI with a pacing-defeating cadence.

        An :class:`InstantCadence` zeroes every pacing delay; shipping one to a
        real CLI defeats the entire point of this package, so say so rather than
        silently behaving like a bulk-write SDK.
        """
        if isinstance(self._cadence, InstantCadence):
            _log.warning("AsyncClient is driving the CLI with InstantCadence: all pacing delays are zero")

    async def _drive_turn(
        self,
        runner: _Runner,
        prompt: str,
        options: QueryOptions,
        *,
        _session_id_out: list[str | None] | None = None,
        pinned_session_id: str | None = None,
    ) -> tuple[QueryResult, bool, bool]:
        """Run one turn on an already-started ``runner``: send, stream, heartbeat.

        Shared by the one-shot :meth:`_execute` and :class:`Session`, so neither
        re-implements the send → stream → heartbeat sequence. Spawning and reaping
        the runner is the *caller's* job — a one-shot query reaps after every turn,
        a session keeps the child alive across turns. Returns the turn's
        :class:`QueryResult` (carrying any CLI-reported session id extracted via
        :attr:`ClientMarkers.session_id`), whether it ended on EOF (the child
        exited), and whether it exited early on a ``complete_on`` predicate (the
        child is alive but mid-generation). A session reaps the runner on *either*
        flag — both mean the child cannot safely serve another turn.

        ``pinned_session_id``, when set, is the spawn-pinned id (proposal 081):
        it becomes :attr:`QueryResult.session_id` whenever no on-screen marker
        captured one this turn — the pinned id is the *floor*, so an extracted id
        still overrides it. ``None`` keeps today's "id only from a marker"
        behaviour, leaving the result id ``None`` when nothing was captured.

        ``_session_id_out``, when provided, is a one-element list updated in-place
        each time a CLI-reported session id is seen during streaming.  Because the
        update happens before :meth:`_check_conditions` can raise, the captured id
        is accessible to the caller even when the method raises instead of
        returning — enabling :class:`Session._run_turn` to preserve ``_resume_id``
        across a :class:`RateLimitError` or :class:`SessionExpiredError`.

        ``pinned_session_id`` is a caller-chosen id pinned at spawn (proposal 081).
        It is the **floor** for the result's session id: when streaming captured no
        on-screen :attr:`ClientMarkers.session_id`, the result reports the pinned id
        so a spawn-pinned caller has an id without the layout ever painting one. An
        on-screen capture still wins for the turn (the pin is a floor, not a
        ceiling); ``None`` leaves the result id exactly what streaming produced.
        """
        screen = Screen(*runner.winsize)
        parser = Parser(self._parser_markers)
        send_ms = await self._send_prompt(runner, prompt)
        stream_ms, ended_on_eof, exited_early, session_id = await self._stream(
            runner, screen, parser, options, _session_id_out=_session_id_out
        )
        if session_id is None:
            # No on-screen id matched this turn → fall back to the spawn-pinned id
            # (proposal 081); ``None`` when nothing was pinned, leaving today's id.
            session_id = pinned_session_id
        heartbeat_ms = await self._heartbeat(len(parser.response))
        # Layout-drift sentinel (proposal 075): a configured idle marker that
        # never matched before EOF means the marker no longer fits the CLI's
        # painted layout (e.g. a CLI upgrade). EOF stays the honest fallback
        # done-signal — the turn is valid — but the degradation is recorded on
        # the result and warned once, never silent. An idle-marker break sets
        # ``ended_on_eof=False`` in ``_stream``, so this is exactly "idle was
        # configured and did not fire"; a ``complete_on`` early exit is a
        # caller-requested stop, not drift, and ``ended_on_eof`` is ``False``
        # there too.
        idle_marker_missed = ended_on_eof and bool(self._markers.idle)
        if idle_marker_missed and not self._idle_drift_warned:
            self._idle_drift_warned = True
            _log.warning(
                "configured idle marker never matched before the turn ended on "
                "EOF — possible CLI layout change (upgrade?); session "
                "continuity degrades to EOF-only turns"
            )
        result = QueryResult(
            response=parser.response,
            tool_calls=parser.tool_calls,
            usage=None,
            # The spawn-pinned id is the floor: an on-screen marker capture wins,
            # else fall back to the pinned id (proposal 081), else None.
            session_id=session_id if session_id is not None else pinned_session_id,
            raw_screen=screen.display(),
            timing=QueryTiming(send_ms=send_ms, stream_ms=stream_ms, heartbeat_ms=heartbeat_ms),
            idle_marker_missed=idle_marker_missed,
        )
        return result, ended_on_eof, exited_early

    async def _send_prompt(self, runner: _Runner, prompt: str) -> float:
        """Type the prompt and Enter at cadence; return the elapsed milliseconds."""
        start = time.monotonic()
        await runner.send_text(prompt + _SUBMIT_KEY)
        return (time.monotonic() - start) * 1000.0

    async def _stream(
        self,
        runner: _Runner,
        screen: Screen,
        parser: Parser,
        options: QueryOptions,
        *,
        _session_id_out: list[str | None] | None = None,
    ) -> tuple[float, bool, bool, str | None]:
        """Read until the turn completes; return ``(elapsed_ms, ended_on_eof, exited_early, session_id)``.

        The first read is bounded by ``startup_timeout_seconds`` and every later
        one by ``stream_idle_timeout_seconds``; whichever fires kills the turn
        with a :class:`CLITimeoutError` of the matching category.

        ``ended_on_eof`` is ``True`` when the turn completed because the child
        exited (end-of-stream) and ``False`` when the configured idle marker
        redrew while the child stayed alive. A session reads it to tell a
        reusable persistent child from a spent one-shot invocation.

        ``session_id`` is the most recent CLI-reported id seen via
        :attr:`ClientMarkers.session_id`, or ``None`` when no marker is configured
        or none matched. Capturing per-snapshot (rather than off the final screen)
        survives the id banner scrolling off the top during a long answer.

        ``_session_id_out``, when provided, is updated in-place with each newly
        captured id **before** :meth:`_check_conditions` is called, so the latest
        id remains accessible to the caller even when this method raises instead of
        returning (e.g. on :class:`RateLimitError`).

        An ``options.complete_on`` predicate, when set, is evaluated against the
        accumulated ``parser.response`` after each snapshot (after the rate-limit /
        expired checks, so a hard condition still wins), but only when the idle
        marker did **not** also fire on that snapshot — a single pty flush can
        deliver the final answer text and the idle-prompt redraw in the same chunk,
        and in that case the idle path wins: the child is at a clean prompt, not
        mid-generation. The first truthy ``complete_on`` result on a non-idle
        snapshot breaks the loop with ``ended_on_eof`` left ``False`` and
        ``exited_early`` set ``True``. Unlike the idle-marker break (turn finished,
        child back at a clean prompt), an early exit stops the child *mid-generation*
        with unconsumed pty output still pending — so it is **not** safe to reuse.
        ``exited_early`` tells the caller to reap it now: the one-shot
        :meth:`_execute` ``finally`` reaps unconditionally, and :class:`Session`
        treats ``exited_early`` like EOF (reap + respawn next turn) rather than like
        the idle marker (keep the child).

        ``options.idle_quiescence_seconds``, when set and no idle marker is
        configured, adds a **fourth completion outcome** (proposal 081): the turn is
        complete once the rendered ``screen.display()`` snapshot has stopped changing
        for the window while the child is alive — the same shape as the idle break
        (``ended_on_eof`` and ``exited_early`` both left ``False``, child clean and
        resident), but layout-free. Quiescence is measured on the rendered snapshot,
        not raw bytes, so a cursor blink or an identical repaint counts as quiet
        while any snapshot-changing repaint resets the window. Precedence is
        unchanged: a configured idle marker disables quiescence outright, a
        ``complete_on`` predicate still wins per snapshot, and EOF still wins; the
        per-read window never exceeds ``stream_idle_timeout_seconds``, the hard outer
        bound it sits inside.
        """
        start = time.monotonic()
        received_any = False
        ended_on_eof = False
        exited_early = False
        session_id: str | None = None
        # Identity of the approval modal we last answered, so a repaint of the
        # same modal before the CLI consumes our reply is not answered twice.
        answered: tuple[object, ...] | None = None
        # Output-quiescence completion (proposal 081): active only when a window is
        # configured and no idle marker competes for the turn-complete role. Track
        # the snapshot as of its last change and when that change happened, so a
        # non-changing repaint (cursor blink) does not keep resetting the window.
        quiet_window = options.idle_quiescence_seconds if not self._markers.idle else None
        quiet_anchor: str | None = None
        quiet_since = start
        while True:
            if not received_any:
                timeout = options.startup_timeout_seconds
            elif quiet_window is not None:
                # The window is the per-read bound, never above the stream-idle
                # outer bound it sits inside.
                timeout = min(quiet_window, options.stream_idle_timeout_seconds)
            else:
                timeout = options.stream_idle_timeout_seconds
            try:
                chunk = await asyncio.wait_for(runner.aread_chunk(), timeout)
            except TimeoutError:
                if quiet_window is not None and received_any:
                    break  # window elapsed with no output: turn complete, child resident
                raise self._timeout_error(received_any, timeout, screen) from None
            if not chunk:
                self._check_clean_exit(runner, screen)
                ended_on_eof = True
                break
            received_any = True
            screen.feed(chunk)
            snapshot = screen.display()
            parser.feed(snapshot)
            extracted = self._extract_session_id(snapshot)
            if extracted:
                session_id = extracted
                if _session_id_out is not None:
                    _session_id_out[0] = extracted
            self._check_conditions(snapshot)
            modal = self._approval.detect(snapshot)
            if modal is None:
                answered = None
                if self._matches(snapshot, self._markers.idle):
                    break  # turn is complete; child is clean; no exited_early
                if options.complete_on is not None and options.complete_on(parser.response):
                    exited_early = True
                    break
                if quiet_window is not None:
                    now = time.monotonic()
                    if snapshot != quiet_anchor:
                        quiet_anchor = snapshot
                        quiet_since = now
                    elif now - quiet_since >= quiet_window:
                        break  # snapshot unchanged for the window: complete, resident
                continue
            identity = self._modal_identity(modal)
            if identity == answered:
                continue  # the same modal repainting; answering twice double-fires the tool
            keys = self._approval.resolve(snapshot)
            assert keys is not None  # detect() found a modal, so resolve returns a key or raises
            await runner.send_text(keys + _SUBMIT_KEY)
            answered = identity
        return (time.monotonic() - start) * 1000.0, ended_on_eof, exited_early, session_id

    async def _heartbeat(self, answer_chars: int) -> float:
        """Pause for the cadence-prescribed heartbeat; return elapsed milliseconds."""
        start = time.monotonic()
        await asyncio.sleep(self._cadence.heartbeat_for(answer_chars))
        return (time.monotonic() - start) * 1000.0

    # -- condition checks --------------------------------------------------

    @staticmethod
    def _timeout_error(received_any: bool, timeout: float, screen: Screen) -> CLITimeoutError:
        snapshot = screen.display()
        if received_any:
            return CLITimeoutError(
                f"stream idle for {timeout}s before completing the turn",
                screen=snapshot,
                category="stream_stall",
            )
        return CLITimeoutError(
            f"no CLI output within {timeout}s of sending the prompt",
            screen=snapshot,
            category="timeout",
        )

    @staticmethod
    def _modal_identity(request: ApprovalRequest) -> tuple[object, ...]:
        """A modal's identity for dedup, excluding the raw screen so a cosmetic
        repaint (spinner, cursor) still counts as the same modal."""
        return (request.prompt, request.options, request.tool)

    def _extract_session_id(self, snapshot: str) -> str | None:
        """Return the CLI-reported session id in ``snapshot``, or ``None``.

        Scans every stripped line for a :attr:`ClientMarkers.session_id` prefix and
        returns the (non-empty) remainder of the *last* match — the id the CLI
        printed after that label. Last (not first) wins so a snapshot that still
        shows a stale id banner near the top yields to a newer id printed lower,
        matching the "most recent id" intent the :meth:`_stream` loop relies on.
        ``None`` when no marker is configured, no line matches, or every matching
        remainder is blank (a bare label with no id is not an id).
        """
        found: str | None = None
        for line in snapshot.split("\n"):
            stripped = line.strip()
            for marker in self._markers.session_id:
                if stripped.startswith(marker):
                    value = stripped[len(marker):].strip()
                    if value:
                        found = value
        return found

    def _check_conditions(self, snapshot: str) -> None:
        """Raise for a rate-limit or expired-session notice visible in ``snapshot``."""
        if self._matches(snapshot, self._markers.rate_limit):
            raise RateLimitError("CLI surfaced a rate-limit notice", screen=snapshot)
        if self._matches(snapshot, self._markers.session_expired):
            raise SessionExpiredError("CLI reported the session expired", screen=snapshot)

    @staticmethod
    def _check_clean_exit(runner: _Runner, screen: Screen) -> None:
        """Treat a non-zero exit at EOF as a crash mid-turn.

        ``returncode`` may still be ``None`` if the child has not been reaped yet;
        that is read as "no evidence of a crash" rather than guessed at, so this
        only fires on an observed non-zero code.
        """
        code = runner.returncode
        if code is not None and code != 0:
            raise CLIExitedError(
                f"claude exited with code {code} before completing the turn",
                screen=screen.display(),
                category="cli_crash",
            )

    @staticmethod
    def _matches(snapshot: str, markers: tuple[str, ...]) -> bool:
        return any(
            line.strip().startswith(marker) for line in snapshot.split("\n") for marker in markers
        )
