"""Tests for the public :class:`AsyncClient`.

The client is the one layer that orchestrates the others, so its tests mock the
bottom of the stack — the pty runner — and drive the real :class:`Screen`,
:class:`Parser`, and :class:`ApprovalHandler` above it. ``FakeRunner`` replays a
scripted sequence of read outcomes (chunks, EOF, a simulated read timeout, or an
un-cancellable hang) so every branch of the read loop is exercised without a real
subprocess, a real pty, or a real ``sleep``: an :class:`InstantCadence` zeroes
every pacing delay.

A note on the timeout tests: rather than wait real wall-clock for
``asyncio.wait_for`` to fire, ``FakeRunner`` raises ``TimeoutError`` from inside
``aread_chunk``. ``wait_for`` propagates an exception its awaitable raises, so the
client reaches exactly the ``except TimeoutError`` branch it would on a genuine
timeout — deterministically and instantly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable

import pytest

from maestro_sdk import (
    ApprovalMarkers,
    ApprovalRequiredError,
    AsyncClient,
    Cadence,
    ClientMarkers,
    CLIExitedError,
    CLINotFoundError,
    CLITimeoutError,
    InstantCadence,
    ParserMarkers,
    PtyRunner,
    QueryOptions,
    RateLimitError,
    SessionExpiredError,
    approve_all,
)

pytestmark = pytest.mark.asyncio

# Read-script sentinels.
_TIMEOUT = object()  # aread_chunk raises TimeoutError (simulated wait_for timeout)
_HANG = object()  # aread_chunk awaits forever (cancellable, for the cancel test)
_EOF = b""


class FakeRunner:
    """A scripted stand-in for :class:`PtyRunner`.

    ``script`` is a list of read outcomes consumed in order by :meth:`aread_chunk`:
    ``bytes`` is a chunk (``b""`` is EOF), ``_TIMEOUT`` raises ``TimeoutError``,
    and ``_HANG`` suspends forever. Sent text is recorded in :attr:`sent`,
    :attr:`closed` flips once :meth:`aclose` runs, and :attr:`reads` counts every
    :meth:`aread_chunk` call so a test can assert the whole script was consumed —
    an early turn-completion that leaves a trailing chunk unread shows up as a
    short count even when the response and completion flags are unchanged.
    """

    def __init__(
        self,
        script: list[object],
        *,
        winsize: tuple[int, int] = (80, 24),
        returncode: int | None = 0,
        start_error: Exception | None = None,
    ) -> None:
        self._script = iter(script)
        self._winsize = winsize
        self._returncode = returncode
        self._start_error = start_error
        self.sent: list[str] = []
        self.closed = False
        self.started = False
        self.reads = 0

    @property
    def winsize(self) -> tuple[int, int]:
        return self._winsize

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    async def send_text(self, text: str, *, prev: str | None = None) -> str | None:
        self.sent.append(text)
        return text[-1] if text else prev

    async def aread_chunk(self) -> bytes:
        self.reads += 1
        action = next(self._script)
        if action is _TIMEOUT:
            raise TimeoutError
        if action is _HANG:
            await asyncio.Event().wait()  # never set; cancelled by the test
        assert isinstance(action, bytes)
        return action

    async def aclose(self) -> None:
        self.closed = True


def _client(runner: FakeRunner, **kwargs: object) -> AsyncClient:
    """An :class:`AsyncClient` wired to ``runner`` and recording the build args."""
    factory_calls: list[tuple[object, QueryOptions]] = []

    def factory(command: object, options: QueryOptions) -> FakeRunner:
        factory_calls.append((command, options))
        return runner

    client = AsyncClient(InstantCadence(), runner_factory=factory, **kwargs)  # type: ignore[arg-type]
    client._factory_calls = factory_calls  # type: ignore[attr-defined]
    return client


class _CountingClock:
    """A monotonic stand-in that advances by 1.0 on every call.

    Patch ``maestro_sdk.client.time.monotonic`` with one of these and pick an
    output-quiescence window *below* the per-call delta (1.0) to cross it on the
    very next snapshot without a real ``sleep`` — robust because completion then
    fires on the immediately following read regardless of how many times asyncio
    samples the clock. Do **not** use this for a window that must span several
    reads: patching ``time.monotonic`` is process-wide, so asyncio's loop shares
    it and consumes an unfixed number of ticks per ``wait_for``, making multi-read
    window arithmetic non-deterministic. Use :class:`_PerReadClock` for that.
    """

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        self._t += 1.0
        return self._t


class _PerReadClock:
    """A monotonic stand-in whose logical time is ``runner.reads`` (1.0 per read).

    Because patching ``time.monotonic`` is process-wide, asyncio's event loop reads
    the same clock; a per-*call* counter (:class:`_CountingClock`) is therefore
    nudged by however many times ``wait_for`` samples the clock per await, which is
    not fixed. Tying logical time to the number of chunks actually read makes the
    output-quiescence window arithmetic depend only on read count, so a test can
    pin the exact read on which quiescence completes — which is what makes the
    per-change window reset (``quiet_since = now``) observable.
    """

    def __init__(self, runner: FakeRunner) -> None:
        self._runner = runner

    def __call__(self) -> float:
        return float(self._runner.reads)


# ---------------------------------------------------------------------------
# Construction


async def test_empty_command_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AsyncClient(InstantCadence(), command=())


async def test_default_runner_factory_builds_a_pty_runner() -> None:
    # The default factory (no injection) must construct a real PtyRunner from the
    # command and options; constructing one does not spawn anything.
    client = AsyncClient(InstantCadence(), env={"FOO": "bar"})
    runner = client._default_runner_factory(["claude", "--flag"], QueryOptions(cwd="/tmp"))
    assert isinstance(runner, PtyRunner)
    assert runner._command == ["claude", "--flag"]


# ---------------------------------------------------------------------------
# Happy path


async def test_query_returns_accumulated_response() -> None:
    # Chunks are raw pty bytes the Screen appends, not full snapshots: "Hello "
    # then "world" render "Hello world", which the Parser streams incrementally.
    runner = FakeRunner([b"Hello ", b"world", _EOF])
    result = await _client(runner).query("hi")
    assert result.response == "Hello world"
    assert result.tool_calls == []
    # The CLI surfaces no counts through the TUI, so usage is honestly None.
    assert result.usage is None
    assert result.session_id is None
    assert result.raw_screen == "Hello world"
    assert result.timing is not None
    assert result.timing.total_ms >= 0.0
    assert runner.closed is True


async def test_query_types_prompt_then_enter() -> None:
    runner = FakeRunner([b"ok", _EOF])
    await _client(runner).query("do it")
    assert runner.sent == ["do it\r"]


async def test_query_extracts_tool_calls() -> None:
    markers = ParserMarkers(tool_call=r"^(?P<name>[A-Za-z]+)\((?P<args>.*)\)$")
    runner = FakeRunner([b"Bash(ls -la)", _EOF])
    result = await _client(runner, parser_markers=markers).query("run ls")
    assert [call.name for call in result.tool_calls] == ["Bash"]
    assert result.tool_calls[0].args == {"raw": "ls -la"}


async def test_idle_marker_completes_turn_before_eof() -> None:
    # With an idle marker, the turn ends when the prompt redraws — the trailing
    # _TIMEOUT would raise if the loop kept reading, proving it stopped at idle.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"answer\r\nREADY", _TIMEOUT])
    result = await _client(runner, client_markers=markers).query("q")
    assert "answer" in result.response
    assert runner.closed is True


# ---------------------------------------------------------------------------
# layout-drift sentinel (proposal 075)


async def test_idle_marker_match_leaves_drift_flag_unset() -> None:
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"answer\r\nREADY", _TIMEOUT])
    result = await _client(runner, client_markers=markers).query("q")
    assert result.idle_marker_missed is False


async def test_idle_marker_miss_sets_flag_and_warns_once_across_turns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A configured idle marker that never matches before EOF is the layout-drift
    # signature: every such turn carries the miss flag, but the WARNING is a
    # one-shot per client — repeated turns must not storm the log.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"answer one", _EOF, b"answer two", _EOF])
    client = _client(runner, client_markers=markers)
    with caplog.at_level(logging.WARNING, logger="maestro_sdk.client"):
        first = await client.query("q1")
        second = await client.query("q2")
    assert first.idle_marker_missed is True
    assert second.idle_marker_missed is True
    drift_warnings = [
        r for r in caplog.records
        if "idle marker never matched" in r.getMessage()
    ]
    assert len(drift_warnings) == 1


async def test_unconfigured_client_eof_turn_logs_no_drift_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # With no idle marker configured EOF is the *expected* done-signal, so an
    # EOF-ended turn is not a miss: no flag, no warning.
    runner = FakeRunner([b"answer", _EOF])
    with caplog.at_level(logging.WARNING, logger="maestro_sdk.client"):
        result = await _client(runner).query("q")
    assert result.idle_marker_missed is False
    assert not [
        r for r in caplog.records
        if "idle marker never matched" in r.getMessage()
    ]


async def test_complete_on_early_exit_is_not_a_drift_miss() -> None:
    # A caller-requested early exit stops the child mid-generation; the idle
    # marker not having fired there is expected, not layout drift.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"answer DONE", _TIMEOUT])
    options = QueryOptions(complete_on=lambda text: "DONE" in text)
    result = await _client(runner, client_markers=markers).query(
        "q", options=options
    )
    assert result.idle_marker_missed is False


# ---------------------------------------------------------------------------
# complete_on early-exit


async def test_complete_on_stops_stream_at_first_satisfying_chunk() -> None:
    # The predicate fires once the accumulated answer contains "DONE": the first
    # chunk ("work ") fails it (exercising the not-None-but-False branch), the
    # second ("more DONE") satisfies it, so the loop breaks there. A trailing
    # _TIMEOUT (no EOF, no idle marker) would raise if the loop kept reading,
    # proving early-exit returned the accumulated text and stopped.
    runner = FakeRunner([b"work ", b"more DONE", _TIMEOUT])
    options = QueryOptions(complete_on=lambda text: "DONE" in text)
    result = await _client(runner).query("q", options=options)
    assert result.response == "work more DONE"
    # ended_on_eof was False, so the child was still mid-generation; the one-shot
    # finally reaps it rather than waiting for EOF or the idle marker.
    assert runner.closed is True


# ---------------------------------------------------------------------------
# Output-quiescence turn completion (proposal 081)


async def test_quiescence_completes_turn_as_resident_child() -> None:
    # idle_quiescence_seconds set, no idle marker: a stream that produces output
    # then goes silent (the trailing _TIMEOUT stands in for the read window
    # elapsing) completes the turn as a resident child — not EOF, not exited_early.
    runner = FakeRunner([b"answer", _TIMEOUT])
    client = _client(runner)
    await runner.start()
    options = QueryOptions(idle_quiescence_seconds=2.0)
    result, ended_on_eof, exited_early = await client._drive_turn(runner, "q", options)
    assert result.response == "answer"
    assert ended_on_eof is False
    assert exited_early is False


async def test_quiescence_completes_on_stable_snapshot_repaint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A repaint that re-renders identical content (clear + same text) leaves the
    # rendered snapshot unchanged; once the window has elapsed since the last real
    # change, that quiet repaint completes the turn. The counting clock advances
    # 1.0 per read, crossing the 0.5 window between the answer and the repaint —
    # so quiescence completes on the repaint chunk itself, with no read timeout.
    monkeypatch.setattr("maestro_sdk.client.time.monotonic", _CountingClock())
    clear = b"\x1b[2J\x1b[H"  # erase display + home cursor → an identical redraw
    runner = FakeRunner([b"answer", clear + b"answer"])
    client = _client(runner)
    await runner.start()
    options = QueryOptions(idle_quiescence_seconds=0.5)
    result, ended_on_eof, exited_early = await client._drive_turn(runner, "q", options)
    assert result.response == "answer"
    assert ended_on_eof is False
    assert exited_early is False


async def test_quiescence_blink_within_window_does_not_complete_until_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Identical repaints (a "blinking" prompt) keep arriving but do not change the
    # snapshot, so they must not reset the window — yet completion must wait until
    # the window has elapsed since the last *real* change. The per-change reset
    # (`quiet_since = now` on the snapshot-changed branch) is what anchors the
    # window to that last change; dropping it would anchor the window to the turn's
    # start instead and complete a read early. Pinning that requires logical time
    # to advance only on reads — asyncio shares the process-wide patched clock, so
    # a per-call counter is perturbed by `wait_for` (see _PerReadClock). With a 1.5
    # window at 1.0/read, the first chunk anchors the window (t=1.0); the first
    # repaint is still inside it (2.0-1.0=1.0<1.5, no completion) and the second
    # crosses it (3.0-1.0=2.0>=1.5). Correct code reads all three chunks; dropping
    # the reset measures from the start (t=0.0) and completes on the *first* repaint
    # (2.0-0.0>=1.5), leaving the last chunk unread. response/ended_on_eof/
    # exited_early are identical either way, so only the read count fails the
    # regression; if nothing completed at all, aread_chunk StopIterations.
    clear = b"\x1b[2J\x1b[H"
    script = [b"answer", clear + b"answer", clear + b"answer"]
    runner = FakeRunner(script)
    monkeypatch.setattr("maestro_sdk.client.time.monotonic", _PerReadClock(runner))
    client = _client(runner)
    await runner.start()
    options = QueryOptions(idle_quiescence_seconds=1.5)
    result, ended_on_eof, exited_early = await client._drive_turn(runner, "q", options)
    assert result.response == "answer"
    assert ended_on_eof is False
    assert exited_early is False
    assert runner.reads == len(script)


async def test_quiescence_does_not_complete_while_snapshot_changes() -> None:
    # Each chunk changes the rendered snapshot, resetting the window; the turn
    # completes on EOF, never early on quiescence. The trailing _EOF (not _TIMEOUT)
    # proves completion came from end-of-stream while the stream kept changing.
    runner = FakeRunner([b"a", b"b", b"c", _EOF])
    client = _client(runner)
    await runner.start()
    options = QueryOptions(idle_quiescence_seconds=2.0)
    result, ended_on_eof, exited_early = await client._drive_turn(runner, "q", options)
    assert result.response == "abc"
    assert ended_on_eof is True
    assert exited_early is False


async def test_idle_marker_takes_precedence_over_quiescence() -> None:
    # With both an idle marker and a quiescence window configured, the idle marker
    # owns turn completion: quiescence is disabled outright. The marker matches and
    # the turn ends there (the trailing _TIMEOUT would raise if it kept reading).
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"answer\r\nREADY", _TIMEOUT])
    options = QueryOptions(idle_quiescence_seconds=2.0)
    result = await _client(runner, client_markers=markers).query("q", options=options)
    assert "answer" in result.response
    assert result.idle_marker_missed is False


async def test_complete_on_takes_precedence_over_quiescence() -> None:
    # On a single snapshot, a satisfied complete_on predicate wins over quiescence:
    # the turn exits early (mid-generation) rather than waiting out the window.
    runner = FakeRunner([b"work DONE", _TIMEOUT])
    client = _client(runner)
    await runner.start()
    options = QueryOptions(
        idle_quiescence_seconds=2.0, complete_on=lambda text: "DONE" in text
    )
    result, ended_on_eof, exited_early = await client._drive_turn(runner, "q", options)
    assert result.response == "work DONE"
    assert exited_early is True
    assert ended_on_eof is False


async def test_quiescence_startup_timeout_still_raises() -> None:
    # Quiescence only completes a turn that produced output. If the very first read
    # times out (the CLI never came up), that is still a startup timeout — not a
    # quiescent turn — because nothing was ever rendered to go quiet.
    runner = FakeRunner([_TIMEOUT])
    options = QueryOptions(idle_quiescence_seconds=2.0)
    with pytest.raises(CLITimeoutError) as excinfo:
        await _client(runner).query("q", options=options)
    assert excinfo.value.category == "timeout"
    assert runner.closed is True


async def test_quiescence_none_keeps_stream_stall_timeout() -> None:
    # idle_quiescence_seconds=None leaves the read loop byte-identical to today: a
    # chunk then silence is still a stream stall, never a quiescence completion.
    runner = FakeRunner([b"partial", _TIMEOUT])
    options = QueryOptions(idle_quiescence_seconds=None)
    with pytest.raises(CLITimeoutError) as excinfo:
        await _client(runner).query("q", options=options)
    assert excinfo.value.category == "stream_stall"
    assert runner.closed is True


async def test_quiescence_per_read_timeout_is_min_of_window_and_stream_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The quiescence branch bounds each post-startup read at
    # min(idle_quiescence_seconds, stream_idle_timeout_seconds): with the
    # recommended window < stream-idle, that smaller window is what makes a silent
    # stream complete at the window rather than at the much later stream-idle outer
    # bound. FakeRunner ignores the timeout it is read under, and the snapshot tests
    # complete via the patched clock — so no other test observes the value actually
    # passed to asyncio.wait_for. Capture it and pin min() in both directions; a
    # regression that dropped the min() would surface here while every other
    # quiescence test still passed.
    recorded: list[float] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(aw: Awaitable[bytes], timeout: float) -> bytes:
        recorded.append(timeout)
        return await real_wait_for(aw, timeout)

    monkeypatch.setattr("maestro_sdk.client.asyncio.wait_for", recording_wait_for)
    startup = QueryOptions().startup_timeout_seconds

    # Window (2.0) below the stream-idle outer bound (10.0): the post-startup read
    # is bounded at the window, not the full stream-idle (a dropped min() → 10.0).
    runner = FakeRunner([b"answer", _TIMEOUT])
    client = _client(runner)
    await runner.start()
    await client._drive_turn(
        runner, "q", QueryOptions(idle_quiescence_seconds=2.0, stream_idle_timeout_seconds=10.0)
    )
    assert recorded == [startup, 2.0]

    # Window (20.0) above the outer bound (10.0): min() clamps the per-read bound to
    # stream-idle — quiescence never reads longer than the hard outer bound.
    recorded.clear()
    runner = FakeRunner([b"answer", _TIMEOUT])
    client = _client(runner)
    await runner.start()
    await client._drive_turn(
        runner, "q", QueryOptions(idle_quiescence_seconds=20.0, stream_idle_timeout_seconds=10.0)
    )
    assert recorded == [startup, 10.0]


# ---------------------------------------------------------------------------
# Spawn-pinned session id (proposal 081)


async def test_pinned_session_id_reaches_spawn_argv_and_result() -> None:
    # A one-shot query with QueryOptions.session_id spawns `claude --session-id
    # <id>` and reports that pinned id on the result even with no on-screen
    # session_id marker — the id is known a priori, not scraped off a screen.
    runner = FakeRunner([b"answer", _EOF])
    client = _client(runner)
    result = await client.query("q", options=QueryOptions(session_id="pinned-id"))
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]
    assert command == ["claude", "--session-id", "pinned-id"]
    assert result.session_id == "pinned-id"


async def test_unset_session_id_leaves_spawn_argv_and_result_unchanged() -> None:
    # The default (session_id=None) leaves the spawn command bare and the result
    # id None — byte-identical to a pre-081 one-shot query.
    runner = FakeRunner([b"answer", _EOF])
    client = _client(runner)
    result = await client.query("q")
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]
    assert command == ["claude"]
    assert result.session_id is None


async def test_on_screen_capture_overrides_pinned_session_id() -> None:
    # The pinned id is the floor, not a ceiling: when a session_id marker matches,
    # the captured id wins for that turn even though an id was pinned at spawn.
    markers = ClientMarkers(session_id=("Session:",))
    runner = FakeRunner([b"Session: captured-id\r\nanswer", _EOF])
    result = await _client(runner, client_markers=markers).query(
        "q", options=QueryOptions(session_id="pinned-id")
    )
    assert result.session_id == "captured-id"


# ---------------------------------------------------------------------------
# Timeout safety nets


async def test_startup_timeout_raises_timeout_category() -> None:
    runner = FakeRunner([_TIMEOUT])
    with pytest.raises(CLITimeoutError) as excinfo:
        await _client(runner).query("q")
    assert excinfo.value.category == "timeout"
    assert excinfo.value.screen is not None
    assert runner.closed is True


async def test_stream_idle_timeout_raises_stream_stall_category() -> None:
    # A chunk arrives (startup satisfied), then the stream goes silent: a stall.
    runner = FakeRunner([b"partial", _TIMEOUT])
    with pytest.raises(CLITimeoutError) as excinfo:
        await _client(runner).query("q")
    assert excinfo.value.category == "stream_stall"
    assert runner.closed is True


# ---------------------------------------------------------------------------
# Exit and on-screen conditions


async def test_nonzero_exit_at_eof_raises_cli_exited() -> None:
    runner = FakeRunner([b"oops", _EOF], returncode=2)
    with pytest.raises(CLIExitedError) as excinfo:
        await _client(runner).query("q")
    assert excinfo.value.category == "cli_crash"
    assert "code 2" in str(excinfo.value)
    assert runner.closed is True


async def test_clean_exit_with_unknown_returncode_does_not_raise() -> None:
    # A not-yet-reaped child (returncode None) is "no evidence of a crash".
    runner = FakeRunner([b"done", _EOF], returncode=None)
    result = await _client(runner).query("q")
    assert result.response == "done"


async def test_rate_limit_screen_raises() -> None:
    markers = ClientMarkers(rate_limit=("Rate limit reached",))
    runner = FakeRunner([b"Rate limit reached, retry later", _EOF])
    with pytest.raises(RateLimitError) as excinfo:
        await _client(runner, client_markers=markers).query("q")
    assert excinfo.value.category == "rate_limit"
    assert excinfo.value.screen is not None
    assert runner.closed is True


async def test_session_expired_screen_raises() -> None:
    markers = ClientMarkers(session_expired=("Session expired",))
    runner = FakeRunner([b"Session expired, please log in", _EOF])
    with pytest.raises(SessionExpiredError) as excinfo:
        await _client(runner, client_markers=markers).query("q")
    assert excinfo.value.category == "stale_session"
    assert runner.closed is True


# ---------------------------------------------------------------------------
# Approvals


_APPROVAL_MARKERS = ApprovalMarkers(
    prompt=("Do you want to proceed",),
    option=r"^(?P<key>\d+)\.\s*(?P<label>.+)$",
    approve_keys=("1",),
)
_MODAL = b"Do you want to proceed?\r\n1. Yes\r\n2. No"


async def test_approval_approved_sends_keystroke_then_completes() -> None:
    runner = FakeRunner([_MODAL, _EOF])
    await _client(
        runner, approval_markers=_APPROVAL_MARKERS, approval_policy=approve_all
    ).query("q")
    # The prompt, then the approval keystroke + Enter when the modal appeared.
    assert runner.sent == ["q\r", "1\r"]


async def test_repainted_modal_is_answered_only_once() -> None:
    # The CLI repaints the same modal before it consumes our reply; the client
    # must not type the approval keystroke twice and double-fire the tool.
    clear = b"\x1b[2J\x1b[H"  # erase display + home cursor → an identical redraw
    runner = FakeRunner([_MODAL, clear + _MODAL, clear + b"done", _EOF])
    await _client(
        runner, approval_markers=_APPROVAL_MARKERS, approval_policy=approve_all
    ).query("q")
    assert runner.sent == ["q\r", "1\r"]


async def test_approval_denied_by_default_raises() -> None:
    runner = FakeRunner([_MODAL, _EOF])
    with pytest.raises(ApprovalRequiredError) as excinfo:
        await _client(runner, approval_markers=_APPROVAL_MARKERS).query("q")
    # Approval-required is a control signal, not a transient failure.
    assert excinfo.value.category == "unknown"
    assert runner.closed is True


# ---------------------------------------------------------------------------
# Cadence warning


async def test_instant_cadence_warns(caplog: pytest.LogCaptureFixture) -> None:
    runner = FakeRunner([b"x", _EOF])
    with caplog.at_level(logging.WARNING, logger="maestro_sdk"):
        await _client(runner).query("q")
    assert any("InstantCadence" in record.message for record in caplog.records)


async def test_real_cadence_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    # A real (zero-heartbeat) Cadence is not an InstantCadence, so no warning —
    # zero heartbeat keeps the test free of any real sleep.
    cadence = Cadence(heartbeat_min_ms=0.0, heartbeat_per_answer_char_ms=0.0, seed=0)
    runner = FakeRunner([b"x", _EOF])
    client = AsyncClient(cadence, runner_factory=lambda _cmd, _opt: runner)  # type: ignore[arg-type,return-value]
    with caplog.at_level(logging.WARNING, logger="maestro_sdk"):
        await client.query("q")
    assert not any("InstantCadence" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# start() failure and cancellation both reclaim the runner


async def test_cli_not_found_propagates_and_closes_runner() -> None:
    runner = FakeRunner([], start_error=CLINotFoundError("not found"))
    with pytest.raises(CLINotFoundError):
        await _client(runner).query("q")
    # Even when start() raises, the finally still reclaims the runner.
    assert runner.closed is True


async def test_cancellation_mid_stream_closes_runner() -> None:
    runner = FakeRunner([_HANG])
    task = asyncio.ensure_future(_client(runner).query("q"))
    # Let the query reach the suspended read before cancelling it.
    for _ in range(3):
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.closed is True


# ---------------------------------------------------------------------------
# Per-call options override


async def test_query_options_override_reaches_runner_factory() -> None:
    runner = FakeRunner([b"x", _EOF])
    client = _client(runner)
    override = QueryOptions(cwd="/work", startup_timeout_seconds=1.0)
    await client.query("q", options=override)
    (_command, options), = client._factory_calls  # type: ignore[attr-defined]
    assert options is override


# ---------------------------------------------------------------------------
# Spawn-pinned session id (proposal 081)


async def test_query_session_id_pins_spawn_and_floors_result() -> None:
    # A one-shot query with a pinned id spawns `claude --session-id <id>` and reports
    # that id on the result even with no session_id marker configured — the pin is
    # the floor, so a metering caller has an id without the layout painting one.
    runner = FakeRunner([b"answer", _EOF])
    client = _client(runner)
    result = await client.query("q", session_id="pinned-1")
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]
    assert command == ["claude", "--session-id", "pinned-1"]
    assert result.session_id == "pinned-1"


async def test_query_without_session_id_leaves_spawn_and_result_unchanged() -> None:
    # The unset path is byte-identical to today: no --session-id on the spawn and a
    # None result id (the honest "no id" default).
    runner = FakeRunner([b"answer", _EOF])
    client = _client(runner)
    result = await client.query("q")
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]
    assert command == ["claude"]
    assert result.session_id is None


async def test_query_on_screen_id_overrides_pinned_floor() -> None:
    # The pin is a floor, not a ceiling: when a session_id marker actually matches on
    # screen, that captured id wins for the turn even though an id was pinned at spawn.
    markers = ClientMarkers(session_id=("Session:",))
    runner = FakeRunner([b"Session: scraped\r\nanswer", _EOF])
    client = _client(runner, client_markers=markers)
    result = await client.query("q", session_id="pinned-1")
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]
    assert command == ["claude", "--session-id", "pinned-1"]  # first spawn still pins
    assert result.session_id == "scraped"  # but the on-screen capture wins
