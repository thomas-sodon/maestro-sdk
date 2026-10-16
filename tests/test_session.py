"""Tests for the persistent :class:`maestro_sdk.Session`.

A session keeps one runner alive across turns, so its tests mock the bottom of
the stack — the pty runner — and drive the real :class:`Screen`,
:class:`Parser`, and :class:`ApprovalHandler` above it. ``FakeRunner`` replays a
scripted sequence of read outcomes per turn; an :class:`InstantCadence` zeroes
every pacing delay so no real ``sleep`` runs. A factory hands out one runner per
spawn and records how many times the client asked for one, so a test can prove a
session reused a single runner (one spawn) versus respawned per turn.

The whole-screen ``idle`` marker is what makes a child reusable: a turn that ends
on the idle redraw leaves the child alive for the next turn, while a turn that
ends on EOF spends it and forces a respawn. Both paths are exercised here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence

import pytest

from maestro_sdk import (
    AsyncClient,
    ClientMarkers,
    CLIExitedError,
    CLITimeoutError,
    InstantCadence,
    QueryOptions,
    QueryResult,
    RateLimitError,
    SessionExpiredError,
)

pytestmark = pytest.mark.asyncio

_EOF = b""
# A read-script sentinel: aread_chunk raises TimeoutError, the same way a real
# stalled stream trips asyncio.wait_for, so the wedged-child path is exercised
# instantly and without a real sleep (see test_client.py for the same trick).
_TIMEOUT = object()

# A script entry that computes its bytes from the runner's accumulated state, so
# a turn's output can reflect what earlier turns typed into the *same* runner —
# the fake stand-in for the persistent CLI remembering prior context.
_ContextFn = Callable[["FakeRunner"], bytes]


def _echo_context(runner: FakeRunner) -> bytes:
    """Reflect every prompt typed into this runner so far, then redraw idle."""
    seen = " ".join(text.strip() for text in runner.sent)
    return (seen + "\r\nREADY").encode()


class FakeRunner:
    """A scripted stand-in for :class:`PtyRunner`, reused across session turns.

    ``script`` is a list of read outcomes consumed in order by :meth:`aread_chunk`
    across *all* turns this runner serves: ``bytes`` is a chunk (``b""`` is EOF),
    a callable is invoked with this runner to compute its bytes from prior state,
    and ``_TIMEOUT`` raises ``TimeoutError`` (a simulated ``wait_for`` stall).
    ``start_error``, if set, makes :meth:`start` raise it; ``send_error``, if set,
    makes :meth:`send_text` raise it (a turn that fails or is cancelled while
    typing the prompt); ``aclose_error``, if set, makes :meth:`aclose` raise it
    *after* bumping :attr:`close_count` (a reap that itself fails — the real
    :meth:`PtyRunner.aclose` can re-raise a non-EIO ``OSError`` or propagate a
    ``CancelledError`` from teardown). :attr:`sent` accumulates every typed string,
    :attr:`close_count` counts reaps, and :attr:`start_count` counts successful
    spawns.
    """

    def __init__(
        self,
        script: Sequence[bytes | _ContextFn | object],
        *,
        winsize: tuple[int, int] = (80, 24),
        returncode: int | None = None,
        start_error: Exception | None = None,
        send_error: BaseException | None = None,
        aclose_error: BaseException | None = None,
    ) -> None:
        self._script: Iterator[bytes | _ContextFn | object] = iter(script)
        self._winsize = winsize
        self._returncode = returncode
        self._start_error = start_error
        self._send_error = send_error
        self._aclose_error = aclose_error
        self.sent: list[str] = []
        self.start_count = 0
        self.close_count = 0

    @property
    def winsize(self) -> tuple[int, int]:
        return self._winsize

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.start_count += 1

    async def send_text(self, text: str, *, prev: str | None = None) -> str | None:
        if self._send_error is not None:
            raise self._send_error
        self.sent.append(text)
        return text[-1] if text else prev

    async def aread_chunk(self) -> bytes:
        action = next(self._script)
        if action is _TIMEOUT:
            raise TimeoutError
        if isinstance(action, bytes):
            return action
        assert callable(action)
        return action(self)

    async def aclose(self) -> None:
        # Bump the counter *before* raising so a test can still assert the reap
        # was attempted even when teardown itself fails (the double-fault path).
        self.close_count += 1
        if self._aclose_error is not None:
            raise self._aclose_error


def _client(runners: Sequence[FakeRunner], **kwargs: object) -> AsyncClient:
    """An :class:`AsyncClient` whose factory hands out ``runners`` in order and
    records each build, so a test can count spawns."""
    queue = iter(runners)
    calls: list[tuple[object, QueryOptions]] = []

    def factory(command: object, options: QueryOptions) -> FakeRunner:
        calls.append((command, options))
        return next(queue)

    client = AsyncClient(InstantCadence(), runner_factory=factory, **kwargs)  # type: ignore[arg-type]
    client._factory_calls = calls  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# Persistence: one runner across turns


async def test_persistent_session_reuses_one_runner() -> None:
    # With an idle marker, each turn ends on the READY redraw with the child
    # still alive, so two queries drive the *same* runner — one spawn, no reap
    # until aclose — and the second turn's output reflects the first turn's
    # prompt, proving the runner carried context across turns.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"a1\r\nREADY", _echo_context])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    first = await session.query("first")
    second = await session.query("second")

    assert "a1" in first.response
    assert "first" in second.response  # the second turn sees the first turn's context
    assert runner.sent == ["first\r", "second\r"]  # both turns drove the one runner
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # one spawn
    assert runner.start_count == 1
    assert runner.close_count == 0  # idle-terminated turns leave the child alive

    await session.aclose()
    assert runner.close_count == 1  # one reap, at close


async def test_eof_terminated_turn_reaps_and_respawns() -> None:
    # With no idle marker, every turn ends on EOF: the child exits, so the
    # session reaps it and the next query spawns a fresh runner — degrading to
    # one-spawn-per-turn, exactly the one-shot behaviour.
    runner1 = FakeRunner([b"r1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session()

    first = await session.query("q1")
    second = await session.query("q2")

    assert first.response == "r1"
    assert second.response == "r2"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned
    assert runner1.sent == ["q1\r"]
    assert runner2.sent == ["q2\r"]
    assert runner1.close_count == 1  # reaped at end of its turn
    assert runner2.close_count == 1

    await session.aclose()  # no live runner survives the last EOF turn
    assert runner2.close_count == 1  # so close adds no extra reap


# ---------------------------------------------------------------------------
# complete_on early-exit on a session


async def test_idle_marker_wins_over_complete_on_in_same_snapshot() -> None:
    # When idle marker and complete_on both fire on the same snapshot, idle wins.
    # This is the 'single pty flush' scenario: the CLI writes its final answer
    # and immediately repaints the READY prompt in the same kernel buffer.
    # complete_on fires on parser.response, but because the same snapshot also
    # shows the idle marker, the child is in a CLEAN state — not mid-generation.
    # The idle path must win: break without exited_early, keeping the runner
    # alive for the next turn.
    #
    # Bug: complete_on was checked BEFORE the idle-marker check, so exited_early=True
    # was set even though the child was at a clean idle prompt, causing an unnecessary
    # reap and (when session_id=None) a spurious respawn that lost conversation context.
    markers = ClientMarkers(idle=("READY",))
    # First chunk satisfies both complete_on (contains "DONE") AND the idle marker
    # ("READY"). Since idle wins, the runner stays alive and serves Q2 directly.
    runner = FakeRunner([b"work DONE\r\nREADY", b"second\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()
    options = QueryOptions(complete_on=lambda text: "DONE" in text)

    first = await session.query("q1", options=options)
    assert "DONE" in first.response

    # The idle marker was also in that snapshot — child is clean, must NOT be reaped.
    assert runner.close_count == 0, (
        f"runner was reaped (close_count={runner.close_count}) — "
        "exited_early was set True even though the idle marker was present in "
        "the same snapshot: complete_on fired before the idle-marker check"
    )
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # no spurious respawn

    second = await session.query("q2")
    assert "second" in second.response
    assert runner.sent == ["q1\r", "q2\r"]  # both turns drove the same runner
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]

    await session.aclose()
    assert runner.close_count == 1


async def test_complete_on_early_exit_reaps_and_respawns_session() -> None:
    # complete_on fires mid-generation: the child is still alive (the satisfying
    # chunk carries no idle redraw and no EOF, so without complete_on the session
    # would keep it), but it holds unconsumed pty output and cannot safely serve
    # another turn. The session must reap it on the early exit and spawn a fresh
    # runner for the next query — never reuse a mid-generation child, which would
    # corrupt the next turn's read with the previous turn's leftover output.
    markers = ClientMarkers(idle=("READY",))
    runner1 = FakeRunner([b"work DONE"])  # satisfies complete_on; no idle redraw, no EOF
    runner2 = FakeRunner([b"second\r\nREADY"])
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()
    options = QueryOptions(complete_on=lambda text: "DONE" in text)

    first = await session.query("q1", options=options)
    assert first.response == "work DONE"
    assert runner1.close_count == 1  # reaped despite neither EOF nor the idle marker

    second = await session.query("q2")  # no complete_on: ends on the idle redraw
    assert "second" in second.response
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned, not reused
    assert runner1.sent == ["q1\r"]
    assert runner2.sent == ["q2\r"]
    assert runner2.close_count == 0  # idle-terminated, kept alive

    await session.aclose()
    assert runner2.close_count == 1  # one reap, at close


# ---------------------------------------------------------------------------
# Lifecycle


async def test_aclose_reaps_a_live_runner() -> None:
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"hi\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    assert runner.close_count == 0  # idle-terminated, kept alive

    await session.aclose()
    assert runner.close_count == 1


async def test_aclose_double_fault_clears_slot_and_stays_idempotent() -> None:
    # aclose() reaps the live child at session close; if that reap's own aclose()
    # raises (a non-EIO OSError or a teardown CancelledError), the slot must still
    # be cleared *before* the reap so the broken runner is never left installed on
    # the closed session (issue #2400). The raise propagates, but the slot is clear,
    # so a second aclose() reaps nothing — the reorder makes close idempotent even
    # when the reap itself fails.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"hi\r\nREADY"], aclose_error=RuntimeError("teardown blew up"))
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.query("q")  # idle-terminated, child kept alive
    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.aclose()

    assert runner.close_count == 1  # reap was attempted despite the raise
    assert session._runner is None  # slot cleared even though aclose() raised

    await session.aclose()  # idempotent: nothing left to reap after a failed close
    assert runner.close_count == 1  # no second reap


async def test_closed_session_refuses_query_and_aclose_without_runner() -> None:
    # Closing a never-queried session reaps nothing (no runner was spawned) and
    # still flips the session closed, so a later query raises.
    runner = FakeRunner([b"hi", _EOF])
    client = _client([runner])
    session = client.create_session()

    await session.aclose()
    assert runner.close_count == 0  # never spawned, nothing to reap

    with pytest.raises(RuntimeError, match="closed"):
        await session.query("q")


async def test_per_call_options_override_drives_the_spawn() -> None:
    runner = FakeRunner([b"x", _EOF])
    client = _client([runner])
    session = client.create_session()

    override = QueryOptions(cwd="/work", startup_timeout_seconds=1.0)
    await session.query("q", options=override)

    (_command, options), = client._factory_calls  # type: ignore[attr-defined]
    assert options is override


# ---------------------------------------------------------------------------
# Failure paths: a turn that crashes or wedges must not leave a broken child
# installed for the next query to reuse, and a failed spawn must not leak.


async def test_crashed_child_is_reaped_and_next_query_respawns() -> None:
    # A non-zero exit at EOF raises CLIExitedError mid-turn. The dead child can
    # never serve another turn, so the session must reap it and clear its slot —
    # the next query then spawns a fresh runner instead of driving a corpse.
    runner1 = FakeRunner([_EOF], returncode=1)
    runner2 = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session()

    with pytest.raises(CLIExitedError):
        await session.query("q1")
    assert runner1.close_count == 1  # crashed child reaped, not left installed

    second = await session.query("q2")  # spawns fresh rather than reusing runner1
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]
    assert runner2.start_count == 1
    assert runner1.close_count == 1  # still just the one crash-time reap


async def test_wedged_child_is_reaped_and_next_query_respawns() -> None:
    # A stream that goes silent past the idle timeout raises CLITimeoutError; the
    # wedged child is just as unusable as a crashed one, so the same reap-and-
    # respawn path applies.
    runner1 = FakeRunner([_TIMEOUT])
    runner2 = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session()

    with pytest.raises(CLITimeoutError):
        await session.query("q1")
    assert runner1.close_count == 1  # wedged child reaped

    second = await session.query("q2")
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned


async def test_failed_spawn_is_reaped_and_not_installed() -> None:
    # If start() raises, the half-built runner can still hold a child or fd, so
    # the session must reap it and leave no runner installed — a later query then
    # tries a clean spawn rather than reusing the failed one.
    boom = FakeRunner([b"x", _EOF], start_error=RuntimeError("spawn failed"))
    healthy = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([boom, healthy])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="spawn failed"):
        await session.query("q1")
    assert boom.close_count == 1  # reaped despite never finishing start()

    second = await session.query("q2")  # not stuck on the failed runner
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]


async def test_cancelled_turn_reaps_and_respawns() -> None:
    # A CancelledError mid-turn must not strand the child: a runner left installed
    # would hold partial pty output for the next query to misparse. The session
    # reaps and clears it — parity with the one-shot _execute's finally-aclose —
    # so the next query spawns fresh. BaseException (not Exception) is what makes
    # this catch a cancellation at all.
    cancelled = FakeRunner([b"x", _EOF], send_error=asyncio.CancelledError())
    healthy = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([cancelled, healthy])
    session = client.create_session()

    with pytest.raises(asyncio.CancelledError):
        await session.query("q1")
    assert cancelled.close_count == 1  # cancelled child reaped, not left installed
    assert session._runner is None  # slot cleared so the next query respawns

    second = await session.query("q2")
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]


async def test_unexpected_error_reaps_and_respawns() -> None:
    # An unexpected (non-SDK) error mid-turn is just as unusable as a crash: the
    # runner can't safely serve another turn, so the session reaps and clears it
    # rather than keeping a corrupt child installed.
    boom = FakeRunner([b"x", _EOF], send_error=RuntimeError("typing blew up"))
    healthy = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([boom, healthy])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="typing blew up"):
        await session.query("q1")
    assert boom.close_count == 1  # reaped, not left installed
    assert session._runner is None

    second = await session.query("q2")
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]


async def test_rate_limit_keeps_runner_for_retry() -> None:
    # A rate-limit notice is raised at a clean point with the child still alive,
    # so — unlike a crash — the session deliberately keeps the runner installed:
    # the caller can retry the turn on the same session once the limit clears,
    # driving the same child rather than respawning.
    markers = ClientMarkers(rate_limit=("RATE LIMIT",), idle=("READY",))
    runner = FakeRunner([b"RATE LIMIT", b"ok\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    with pytest.raises(RateLimitError):
        await session.query("q1")
    assert runner.close_count == 0  # kept alive for retry, not reaped
    assert session._runner is runner  # same runner stays installed

    second = await session.query("q2")  # retry drives the same child
    assert "ok" in second.response  # idle-terminated, so the response trails READY
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # no respawn
    assert runner.start_count == 1
    assert runner.close_count == 0

    await session.aclose()
    assert runner.close_count == 1  # reaped once, at close


async def test_double_fault_mid_turn_clears_slot_and_respawns() -> None:
    # Double fault: the turn fails mid-send AND the reap's own aclose() raises. A
    # raising aclose() must not strand the broken runner in the slot: the slot is
    # cleared before the reap, so even when aclose() blows up the exception still
    # propagates and the next query spawns fresh instead of driving a dead child
    # whose master fd aclose already closed (issue #2400). The propagating error is
    # the aclose failure, with the original mid-turn error chained as __context__.
    boom = FakeRunner(
        [b"x", _EOF],
        send_error=RuntimeError("typing blew up"),
        aclose_error=RuntimeError("teardown blew up"),
    )
    healthy = FakeRunner([b"ok", _EOF], returncode=0)
    client = _client([boom, healthy])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="teardown blew up") as exc_info:
        await session.query("q1")
    assert isinstance(exc_info.value.__context__, RuntimeError)
    assert "typing blew up" in str(exc_info.value.__context__)  # original error preserved
    assert boom.close_count == 1  # reap was attempted despite the raise
    assert session._runner is None  # slot cleared even though aclose() raised

    second = await session.query("q2")  # respawns fresh rather than reusing the corpse
    assert second.response == "ok"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]


async def test_double_fault_on_eof_reap_clears_slot_and_respawns() -> None:
    # The EOF-terminated turn completes, then reaps the spent child — and that reap's
    # aclose() raises. The slot must be cleared before the reap so a failing aclose()
    # cannot leave the dead runner installed: the raise propagates, but the next query
    # still respawns instead of reusing a runner whose fd was already closed.
    runner1 = FakeRunner([b"r1", _EOF], returncode=0, aclose_error=RuntimeError("teardown blew up"))
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.query("q1")
    assert runner1.close_count == 1  # EOF reap was attempted
    assert session._runner is None  # slot cleared despite the raising aclose()

    second = await session.query("q2")  # respawns fresh
    assert second.response == "r2"
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Resume + initial prompt + session-id capture (proposal 061 Step 2)


async def test_resume_initial_prompt_and_session_id_capture() -> None:
    # The acceptance bundle: create_session(resume=...) spawns `claude --resume
    # <id>`, an initial_prompt is sent as the first turn ahead of the caller's
    # prompt, and the CLI-reported session id is captured onto the result. One
    # idle-terminated runner serves both turns, so there is exactly one spawn to
    # assert the --resume argv against.
    markers = ClientMarkers(idle=("READY",), session_id=("Session:",))
    runner = FakeRunner([b"Session: sess-1\r\nseeded\r\nREADY", b"Session: sess-2\r\nanswer\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session("seed prompt", resume="abc")

    result = await session.query("real prompt")

    (command, _options), = client._factory_calls  # type: ignore[attr-defined]  # one spawn
    assert command == ["claude", "--resume", "abc"]
    assert runner.sent == ["seed prompt\r", "real prompt\r"]  # seed first, caller second
    assert result.session_id == "sess-2"  # latest CLI-reported id, from the caller turn
    assert "answer" in result.response


async def test_resume_id_refreshes_and_carries_into_respawn() -> None:
    # With no idle marker every turn ends on EOF and respawns. The first turn
    # captures the CLI-reported id, so the *second* spawn resumes from it even
    # though the session was created without an explicit resume id — that is how
    # continuity survives the one-spawn-per-turn degrade path.
    markers = ClientMarkers(session_id=("Session:",))
    runner1 = FakeRunner([b"Session: sess-1\r\nr1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()

    first = await session.query("q1")
    await session.query("q2")

    assert first.session_id == "sess-1"
    (cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert "--resume" not in cmd1  # first spawn had no id yet
    assert cmd2 == ["claude", "--resume", "sess-1"]  # respawn resumes the captured id


async def test_initial_prompt_fires_once() -> None:
    # The seed is delivered only ahead of the *first* turn; a later query must not
    # resend it. One reused runner (idle marker) sees seed, then each caller turn.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner([b"seedout\r\nREADY", b"a1\r\nREADY", b"a2\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session("seed")

    await session.query("q1")
    await session.query("q2")

    assert runner.sent == ["seed\r", "q1\r", "q2\r"]  # seed only ahead of the first turn
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # one spawn for all turns


async def test_no_session_id_marker_leaves_result_id_none() -> None:
    # Without a session_id marker the client extracts nothing even when the screen
    # shows an id-shaped line, so the result id stays None and a respawn carries no
    # --resume (the honest "did not capture" default).
    runner1 = FakeRunner([b"Session: sess-9\r\nr1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])  # no client_markers → no session_id marker
    session = client.create_session()

    first = await session.query("q1")
    await session.query("q2")

    assert first.session_id is None
    (_cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert "--resume" not in cmd2  # nothing captured, so nothing to resume


async def test_initial_prompt_resent_after_failed_seed_turn() -> None:
    # The seed must survive a seed turn that *fails*: a rate-limit notice on the
    # very first turn keeps the runner alive but delivers nothing, so the caller's
    # retry has to re-send the seed (cleared only after a turn returns, never
    # before). Otherwise the conversation would start without its connect-time
    # seed on exactly the failure path resume exists for.
    markers = ClientMarkers(rate_limit=("RATE LIMIT",), idle=("READY",))
    runner = FakeRunner([b"RATE LIMIT", b"seedout\r\nREADY", b"answer\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session("seed")

    with pytest.raises(RateLimitError):
        await session.query("real")  # seed turn trips the rate limit, raises
    assert runner.sent == ["seed\r"]  # seed was typed but not delivered
    assert session._runner is runner  # kept alive for retry

    result = await session.query("real")  # retry re-sends the still-pending seed
    assert runner.sent == ["seed\r", "seed\r", "real\r"]  # seed resent, then caller
    assert "answer" in result.response
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # one spawn throughout


async def test_initial_prompt_seed_turn_ignores_caller_complete_on() -> None:
    # The connect-time seed must be governed by the session's spawn-time factory
    # options, never the caller's first-turn complete_on. If the caller predicate
    # leaked onto the seed turn (the bug), a chunk of the *seed's* streamed text
    # that satisfies it — before the seed's idle redraw — would trip an early exit
    # and reap the child mid-seed; a fresh spawn would then serve the caller against
    # a conversation that never received the full seed.
    #
    # The seed streams a predicate-satisfying chunk ("seed DONE") ahead of its idle
    # redraw ("\r\nREADY") in two chunks, so under the bug the seed turn early-exits
    # on the first chunk and respawns for the caller, while correctly it reads on to
    # the idle marker and reuses the one runner. Assert on the complete_on driving
    # each turn (recorded per turn) as the primary check: a same-snapshot idle-vs-
    # complete_on case would otherwise let the idle marker win and mask the leak.
    markers = ClientMarkers(idle=("READY",))
    seed_runner = FakeRunner([b"seed DONE", b"\r\nREADY", b"answer\r\nREADY"])
    respawn_runner = FakeRunner([b"answer\r\nREADY"])  # only spawned if the bug reaps the seed
    client = _client([seed_runner, respawn_runner], client_markers=markers)
    session = client.create_session("seed")

    # Record the complete_on governing each turn, in order, by wrapping _drive_turn.
    driven_predicates: list[Callable[[str], bool] | None] = []
    original_drive_turn = client._drive_turn

    async def recording_drive_turn(
        runner: object,
        prompt: str,
        options: QueryOptions,
        **kwargs: object,
    ) -> tuple[QueryResult, bool, bool]:
        driven_predicates.append(options.complete_on)
        return await original_drive_turn(runner, prompt, options, **kwargs)  # type: ignore[arg-type]

    client._drive_turn = recording_drive_turn  # type: ignore[method-assign]

    def caller_predicate(text: str) -> bool:
        return "DONE" in text

    await session.query("real", options=QueryOptions(complete_on=caller_predicate))

    # (a) the seed turn is governed by spawn-time options, not the caller predicate.
    assert driven_predicates[0] is None
    # (b) the caller turn honours the caller's predicate.
    assert driven_predicates[1] is caller_predicate
    # (c) the seed was delivered in full ahead of the caller on one reused runner —
    # never reaped mid-seed and respawned.
    assert seed_runner.sent == ["seed\r", "real\r"]
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # one spawn for both turns
    assert respawn_runner.start_count == 0  # the respawn runner was never needed

    await session.aclose()


async def test_session_id_last_match_wins_within_snapshot() -> None:
    # When one snapshot shows two id lines — a stale banner near the top and a
    # newer id below — the *last* non-empty match wins, so the result carries the
    # most recent id rather than the leftover banner.
    markers = ClientMarkers(idle=("READY",), session_id=("Session:",))
    runner = FakeRunner([b"Session: old\r\nSession: new\r\nanswer\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    result = await session.query("q")

    assert result.session_id == "new"  # last id on screen, not the stale top banner


# ---------------------------------------------------------------------------
# Spawn-pinned session id (proposal 081)


async def test_pinned_session_spawns_with_session_id_then_resumes() -> None:
    # The acceptance bundle: a session created with a pinned id spawns
    # `claude --session-id <id>` on the first turn, every turn's result id is the
    # pinned id with no marker configured, and the respawn (no idle marker → EOF
    # each turn) carries `--resume <id>` reusing the same id.
    runner1 = FakeRunner([b"r1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])  # no client_markers → no on-screen id
    session = client.create_session(session_id="pinned-1")

    first = await session.query("q1")
    second = await session.query("q2")

    assert first.session_id == "pinned-1"  # floored from the pin, no marker needed
    assert second.session_id == "pinned-1"
    (cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert cmd1 == ["claude", "--session-id", "pinned-1"]  # first spawn registers the id
    assert cmd2 == ["claude", "--resume", "pinned-1"]  # respawn reconnects to it


async def test_pinned_session_resumes_pinned_id_after_first_turn_crash() -> None:
    # If the first pinned turn crashes before producing a result, _resume_id is still
    # the pinned id (seeded at construction), so the respawn reconnects to the
    # registered session rather than spawning a blank one — continuity on the exact
    # failure path where no result id was captured.
    runner1 = FakeRunner([b"boom", _EOF], returncode=1)  # nonzero exit → CLIExitedError
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session(session_id="pinned-2")

    with pytest.raises(CLIExitedError):
        await session.query("q1")
    await session.query("q2")

    (cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert cmd1 == ["claude", "--session-id", "pinned-2"]  # first spawn registers
    assert cmd2 == ["claude", "--resume", "pinned-2"]  # respawn resumes the seeded pin


async def test_pinned_session_on_screen_id_overrides_floor_and_drives_resume() -> None:
    # The pin is a floor, not a ceiling: when a session_id marker actually matches on
    # screen, that captured id wins for the turn and refreshes the resume id, so the
    # respawn `--resume`s the captured id rather than the pin.
    markers = ClientMarkers(session_id=("Session:",))
    runner1 = FakeRunner([b"Session: scraped\r\nr1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session(session_id="pinned-3")

    first = await session.query("q1")
    await session.query("q2")

    assert first.session_id == "scraped"  # on-screen capture wins over the pin
    (cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert cmd1 == ["claude", "--session-id", "pinned-3"]  # first spawn still pins
    assert cmd2 == ["claude", "--resume", "scraped"]  # respawn resumes the captured id


async def test_create_session_rejects_both_resume_and_session_id() -> None:
    # ``resume`` and ``session_id`` are mutually exclusive intents — reconnect an
    # existing transcript vs register a fresh pinned one — so passing both fails loud
    # at construction, before any spawn.
    client = _client([])  # Session.__init__ raises before a runner is ever requested
    with pytest.raises(ValueError, match="at most one"):
        client.create_session(resume="a", session_id="b")


async def test_unpinned_session_spawn_argv_unchanged() -> None:
    # The unset path is byte-identical to today: no pinned id means no --session-id
    # on the spawn and a None result id under the empty default markers.
    runner = FakeRunner([b"r1", _EOF], returncode=0)
    client = _client([runner])
    session = client.create_session()

    result = await session.query("q1")

    assert result.session_id is None
    (cmd1, _o1), = client._factory_calls  # type: ignore[attr-defined]
    assert cmd1 == ["claude"]  # no --session-id, no --resume


async def test_session_expired_keeps_runner_for_retry() -> None:
    # An expired-session notice, like a rate limit, is raised at a clean point
    # with the child still alive, so the session keeps the runner installed for a
    # retry on the same handle rather than reaping it. (Both notices share one
    # except clause in Session.query; this pins the SessionExpiredError half.)
    markers = ClientMarkers(session_expired=("SESSION EXPIRED",), idle=("READY",))
    runner = FakeRunner([b"SESSION EXPIRED", b"ok\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    with pytest.raises(SessionExpiredError):
        await session.query("q1")
    assert runner.close_count == 0  # kept alive for retry, not reaped
    assert session._runner is runner  # same runner stays installed

    second = await session.query("q2")  # retry drives the same child
    assert "ok" in second.response
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # no respawn
    assert runner.start_count == 1
    assert runner.close_count == 0

    await session.aclose()
    assert runner.close_count == 1  # reaped once, at close


async def test_session_id_captured_before_rate_limit_is_preserved_in_resume_id() -> None:
    # Chunk 1 delivers the session id; chunk 2 delivers the rate-limit marker on a
    # fresh terminal row (\r\n ensures _matches sees it at the start of a line).
    # After re-raising RateLimitError, _resume_id must hold the id captured before
    # the error so any subsequent respawn can use --resume without losing the
    # conversation — the exact continuity-loss scenario this fix targets.
    markers = ClientMarkers(
        rate_limit=("RATE LIMIT",),
        idle=("READY",),
        session_id=("Session:",),
    )
    runner = FakeRunner([b"Session: sess-1\r\npartial answer", b"\r\nRATE LIMIT"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    with pytest.raises(RateLimitError):
        await session.query("q1")

    # The session id was visible in chunk 1 before the rate-limit fired in chunk 2.
    # After re-raising RateLimitError, _resume_id must hold that captured id so
    # that any subsequent respawn will use --resume.
    assert session._resume_id == "sess-1", (
        f"_resume_id should be 'sess-1' but is {session._resume_id!r} — "
        "partial session id captured before the rate-limit error was discarded"
    )


async def test_session_id_captured_before_session_expired_is_preserved_in_resume_id() -> None:
    # Mirror of the rate-limit variant: a session-expired notice on chunk 2 must
    # not discard the session id captured from chunk 1.
    markers = ClientMarkers(
        session_expired=("SESSION EXPIRED",),
        idle=("READY",),
        session_id=("Session:",),
    )
    runner = FakeRunner([b"Session: sess-2\r\npartial answer", b"\r\nSESSION EXPIRED"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    with pytest.raises(SessionExpiredError):
        await session.query("q1")

    assert session._resume_id == "sess-2", (
        f"_resume_id should be 'sess-2' but is {session._resume_id!r} — "
        "partial session id captured before the session-expired error was discarded"
    )


# ---------------------------------------------------------------------------
# Spawn-pinned session id (proposal 081 Step 2)


async def test_pinned_session_id_spawns_and_reports_across_quiescent_turns() -> None:
    # create_session(session_id=...) pins a fresh id at spawn: the first spawn
    # carries `--session-id <id>`, and with a quiescence window (and no idle
    # marker) each turn completes on the silent stream as a resident child, so two
    # queries drive the *one* runner — one spawn — and every turn reports the
    # pinned id with no on-screen marker (the floor). _TIMEOUT stands in for the
    # quiescence window elapsing after each answer.
    runner = FakeRunner([b"a1", _TIMEOUT, b"a2", _TIMEOUT])
    client = _client([runner])  # no idle marker, no session_id marker
    session = client.create_session(session_id="pinned-uuid")
    options = QueryOptions(idle_quiescence_seconds=2.0)

    first = await session.query("q1", options=options)
    second = await session.query("q2", options=options)

    assert first.response == "a1"
    assert second.response == "a2"
    assert first.session_id == "pinned-uuid"  # pinned floor, no marker configured
    assert second.session_id == "pinned-uuid"
    (command, _options), = client._factory_calls  # type: ignore[attr-defined]  # one spawn
    assert command == ["claude", "--session-id", "pinned-uuid"]
    assert runner.sent == ["q1\r", "q2\r"]  # both turns drove the one runner
    assert runner.start_count == 1
    assert runner.close_count == 0  # quiescence-terminated turns leave the child alive

    await session.aclose()
    assert runner.close_count == 1  # one reap, at close


async def test_pinned_session_id_carries_into_respawn_as_resume() -> None:
    # With no idle marker and no quiescence, every turn ends on EOF and respawns.
    # The pinned id creates the session on the *first* spawn (`--session-id`) and
    # floors the reported id, so it refreshes the resume id — and the respawn then
    # reconnects with `--resume <id>` to the same pinned transcript.
    runner1 = FakeRunner([b"r1", _EOF], returncode=0)
    runner2 = FakeRunner([b"r2", _EOF], returncode=0)
    client = _client([runner1, runner2])
    session = client.create_session(session_id="pinned-uuid")

    first = await session.query("q1")
    second = await session.query("q2")

    assert first.session_id == "pinned-uuid"  # pinned floor on the first turn
    assert second.session_id == "pinned-uuid"
    (cmd1, _o1), (cmd2, _o2) = client._factory_calls  # type: ignore[attr-defined]
    assert cmd1 == ["claude", "--session-id", "pinned-uuid"]  # first spawn creates it
    assert cmd2 == ["claude", "--resume", "pinned-uuid"]  # respawn reconnects to it


async def test_resume_and_session_id_are_mutually_exclusive() -> None:
    # Reconnecting to an existing transcript and pinning a fresh one are
    # contradictory spawn intents, so create_session refuses both at once.
    runner = FakeRunner([b"x", _EOF])
    client = _client([runner])
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.create_session(resume="abc", session_id="xyz")


# ---------------------------------------------------------------------------
# Slash-command metrics probe (proposal 062 Step 1)


async def test_run_command_probes_live_runner_and_leaves_state_unchanged() -> None:
    # A probe is NOT a conversational turn: run_command("/cost") drives the *same*
    # live runner the preceding query brought up, returns its redraw snapshot, and
    # leaves the session's resume id untouched (the probe's own Session: line must
    # NOT refresh it) so the next query continues the conversation as if the probe
    # had never run — one spawn throughout, no reap until aclose.
    markers = ClientMarkers(idle=("READY",), session_id=("Session:",))
    runner = FakeRunner(
        [
            b"Session: sess-1\r\na1\r\nREADY",  # query "first"
            b"Session: sess-probe\r\nCOST $0.12\r\nREADY",  # probe "/cost"
            b"Session: sess-2\r\na2\r\nREADY",  # query "second"
        ]
    )
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    first = await session.query("first")
    assert "a1" in first.response
    assert session._resume_id == "sess-1"

    snapshot = await session.run_command("/cost")

    assert "COST $0.12" in snapshot  # the probe returns the rendered redraw
    assert session._resume_id == "sess-1"  # the probe did NOT refresh the resume id
    assert runner.close_count == 0  # idle redraw kept the child alive — no reap
    assert session._runner is runner  # same live runner stays installed
    assert runner.sent == ["first\r", "/cost\r"]  # the probe drove the same runner

    second = await session.query("second")
    assert "a2" in second.response
    assert runner.sent == ["first\r", "/cost\r", "second\r"]  # all on the one runner
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # one spawn throughout
    assert runner.start_count == 1

    await session.aclose()
    assert runner.close_count == 1  # one reap, at close


async def test_run_command_eof_reaps_and_clears_runner() -> None:
    # A probe that ends on EOF spent the child (the CLI exited), so run_command
    # reaps it and clears the slot — the next query then respawns rather than
    # driving a corpse, mirroring the EOF-terminated turn path.
    runner1 = FakeRunner([b"a\r\nREADY", b"COST", _EOF], returncode=0)
    runner2 = FakeRunner([b"a2\r\nREADY"])
    markers = ClientMarkers(idle=("READY",))
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    snapshot = await session.run_command("/cost")

    assert "COST" in snapshot  # the redraw is still returned before the reap
    assert runner1.close_count == 1  # EOF spent the child, so the probe reaped it
    assert session._runner is None  # slot cleared for the next query to respawn

    second = await session.query("q2")
    assert "a2" in second.response
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned


async def test_run_command_rate_limit_keeps_runner_for_retry() -> None:
    # A rate-limit notice surfaced during a probe is raised with the child still
    # alive (parity with query), so run_command keeps the runner installed: the
    # caller can retry once the limit clears, driving the same child — a probe must
    # never tear down the live conversation just because it could not read a metric.
    markers = ClientMarkers(rate_limit=("RATE LIMIT",), idle=("READY",))
    runner = FakeRunner([b"ok\r\nREADY", b"RATE LIMIT", b"ok2\r\nREADY"])
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    with pytest.raises(RateLimitError):
        await session.run_command("/cost")

    assert runner.close_count == 0  # kept alive for retry, not reaped
    assert session._runner is runner  # same runner stays installed

    second = await session.query("q2")  # the conversation continues on the same child
    assert "ok2" in second.response
    assert len(client._factory_calls) == 1  # type: ignore[attr-defined]  # no respawn


async def test_run_command_unexpected_error_reaps_and_clears_runner() -> None:
    # Any other mid-probe failure — here a wedged stream that trips the idle
    # timeout — leaves a runner that cannot safely serve another turn, so the probe
    # reaps and clears it (like a crashed/cancelled turn) and the next query
    # respawns.
    runner1 = FakeRunner([b"a\r\nREADY", _TIMEOUT])
    runner2 = FakeRunner([b"a2\r\nREADY"])
    markers = ClientMarkers(idle=("READY",))
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    with pytest.raises(CLITimeoutError):
        await session.run_command("/cost")

    assert runner1.close_count == 1  # wedged child reaped, not left installed
    assert session._runner is None  # slot cleared so the next query respawns

    second = await session.query("q2")
    assert "a2" in second.response
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned


async def test_run_command_double_fault_clears_slot_and_respawns() -> None:
    # Double fault on a probe: the probe wedges (mid-probe failure) AND its reap's
    # aclose() raises. As on a turn, the slot is cleared before the reap, so a
    # failing aclose() cannot strand the broken runner — the raise propagates and
    # the next query respawns from the untouched resume id instead of driving a
    # dead child (issue #2400). The mid-probe error is chained as __context__.
    runner1 = FakeRunner([b"a\r\nREADY", _TIMEOUT], aclose_error=RuntimeError("teardown blew up"))
    runner2 = FakeRunner([b"a2\r\nREADY"])
    markers = ClientMarkers(idle=("READY",))
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    with pytest.raises(RuntimeError, match="teardown blew up") as exc_info:
        await session.run_command("/cost")
    assert isinstance(exc_info.value.__context__, CLITimeoutError)  # wedge preserved

    assert runner1.close_count == 1  # reap was attempted despite the raise
    assert session._runner is None  # slot cleared even though aclose() raised

    second = await session.query("q2")  # respawns fresh
    assert "a2" in second.response
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned


async def test_run_command_double_fault_on_eof_clears_slot_and_respawns() -> None:
    # A probe that ends on EOF reaps the spent child; if that reap's aclose() raises,
    # the slot must be cleared *before* the reap so the dead runner (its master fd
    # already closed) is never left installed — the raise propagates and the next
    # query respawns instead of driving a corpse (issue #2400). Mirrors the mid-probe
    # double-fault, but on the EOF/early-exit reap branch of run_command.
    runner1 = FakeRunner(
        [b"a\r\nREADY", b"COST", _EOF],
        returncode=0,
        aclose_error=RuntimeError("teardown blew up"),
    )
    runner2 = FakeRunner([b"a2\r\nREADY"])
    markers = ClientMarkers(idle=("READY",))
    client = _client([runner1, runner2], client_markers=markers)
    session = client.create_session()

    await session.query("q")
    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.run_command("/cost")

    assert runner1.close_count == 1  # EOF reap was attempted despite the raise
    assert session._runner is None  # slot cleared even though aclose() raised

    second = await session.query("q2")  # respawns fresh
    assert "a2" in second.response
    assert len(client._factory_calls) == 2  # type: ignore[attr-defined]  # respawned


async def test_run_command_on_closed_session_raises() -> None:
    # A closed session refuses a probe exactly as it refuses a query — the closed
    # check precedes the live-runner check.
    runner = FakeRunner([b"hi\r\nREADY"])
    markers = ClientMarkers(idle=("READY",))
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await session.run_command("/cost")


async def test_run_command_without_live_runner_raises() -> None:
    # A probe has nothing to measure before a query has brought the session up, so
    # run_command on a never-queried session raises rather than spawning a child
    # just to probe a blank conversation.
    runner = FakeRunner([b"hi\r\nREADY"])
    client = _client([runner])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="no live runner"):
        await session.run_command("/cost")


# ---------------------------------------------------------------------------
# Double fault: the reap itself raises. Every reap-and-clear site must still
# leave self._runner is None (so the next query spawns fresh instead of reusing
# a broken child), with the teardown exception propagating loudly — never
# swallowed. Covers all four reap sites: the _run_turn except handler, the
# _run_turn EOF reap, the run_command EOF reap, and the run_command except
# handler.


async def test_double_fault_turn_clears_runner_when_aclose_raises() -> None:
    # The exact scenario the issue describes: the turn fails (send_text raises)
    # AND the reap that follows fails (aclose raises). The slot must still clear —
    # the try/finally guarantees self._runner is None so the next query spawns
    # fresh — while the teardown exception surfaces rather than being swallowed.
    # Covers the _run_turn `except BaseException` handler.
    runner = FakeRunner(
        [b"x", _EOF],
        send_error=RuntimeError("turn failed"),
        aclose_error=RuntimeError("teardown blew up"),
    )
    client = _client([runner])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.query("q1")
    assert runner.close_count == 1  # the reap was still attempted
    assert session._runner is None  # slot cleared despite the raising teardown


async def test_double_fault_eof_reap_clears_runner_when_aclose_raises() -> None:
    # The turn itself succeeds but ends on EOF (no idle marker → the child exited),
    # so _run_turn reaps at its end-of-turn branch — and that reap raises. The slot
    # must still clear so the next query respawns, with the teardown error
    # surfacing. Covers the _run_turn EOF/early-exit reap.
    runner = FakeRunner([b"r1", _EOF], returncode=0, aclose_error=RuntimeError("teardown blew up"))
    client = _client([runner])
    session = client.create_session()

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.query("q1")
    assert runner.close_count == 1  # the EOF reap was attempted
    assert session._runner is None  # slot cleared despite the raising teardown


async def test_double_fault_probe_clears_runner_when_aclose_raises() -> None:
    # Bring the session up with a clean idle-terminated query, then drive a probe
    # that ends on EOF (the child exited) *and* whose reap raises. run_command must
    # still clear the slot so the next query respawns, surfacing the teardown
    # error. Covers the run_command reap-and-clear.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner(
        [b"a\r\nREADY", b"COST", _EOF],
        returncode=0,
        aclose_error=RuntimeError("teardown blew up"),
    )
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.query("q")  # idle-terminated: child stays alive, no reap yet
    assert runner.close_count == 0

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.run_command("/cost")
    assert runner.close_count == 1  # the probe's EOF reap was attempted
    assert session._runner is None  # slot cleared despite the raising teardown


async def test_double_fault_probe_except_handler_clears_runner_when_aclose_raises() -> None:
    # Bring the session up cleanly, then drive a probe whose stream wedges
    # (CLITimeoutError, hitting run_command's `except BaseException`) *and* whose
    # reap then raises. This is the one reap site the other double-fault tests do
    # not exercise with a raising aclose. The slot must still clear so the next
    # query respawns, and the teardown error surfaces over the in-flight
    # CLITimeoutError (kept as its __context__) rather than being swallowed.
    # Covers the run_command except-handler reap.
    markers = ClientMarkers(idle=("READY",))
    runner = FakeRunner(
        [b"a\r\nREADY", _TIMEOUT],
        aclose_error=RuntimeError("teardown blew up"),
    )
    client = _client([runner], client_markers=markers)
    session = client.create_session()

    await session.query("q")  # idle-terminated: child stays alive, no reap yet
    assert runner.close_count == 0

    with pytest.raises(RuntimeError, match="teardown blew up"):
        await session.run_command("/cost")
    assert runner.close_count == 1  # the except-handler reap was attempted
    assert session._runner is None  # slot cleared despite the raising teardown
