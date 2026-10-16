"""Persistent conversational session over an :class:`~maestro_sdk.AsyncClient`.

A one-shot :meth:`AsyncClient.query` spawns, drives, and reaps its own runner per
call, so two queries never share a process — correct for a stateless invocation
but it throws away conversation state between turns. A :class:`Session` keeps
**one** runner alive across its turns instead: the first :meth:`query` spawns the
child, every later one types into the *same* live pty, and the child is reaped
once at :meth:`aclose`. Because the child stays resident, its in-process
conversation state carries forward — the pty itself is the transcript.

**Continuity survives a respawn via ``--resume``.** When a turn ends on EOF (or a
crash) the child is spent, so the next turn spawns fresh — which would otherwise
start a blank conversation. A session created with ``resume`` spawns
``claude --resume <id>`` from the first turn, and every turn updates the stored id
from the CLI-reported :attr:`~maestro_sdk.QueryResult.session_id`, so a respawn
reconnects to the same transcript rather than losing it. A session created with
``session_id`` instead *pins* a fresh id (proposal 081): its first spawn runs
``claude --session-id <id>`` (creating the session under that id) and that pinned
id floors every turn's reported id, so the first turn refreshes the stored id to
it and any respawn reconnects with ``--resume <id>`` — continuity with no on-screen
id to scrape. ``resume`` and ``session_id`` are mutually exclusive. An
``initial_prompt`` is sent as the session's first turn (ahead of the caller's
first :meth:`query` prompt), mirroring the Anthropic provider's connect-time seed
message.

**A spawn-pinned id needs no on-screen marker (proposal 081).** A session created
with ``session_id`` instead registers a caller-chosen id at the *first* spawn
(``claude --session-id <id>``), reports that id on every turn's
:attr:`~maestro_sdk.QueryResult.session_id` even with no marker configured, and
reuses it as ``--resume <id>`` on a respawn — so continuity (and the transcript
sidecar metering keyed on the id) work on a layout that paints no session-id line.
``resume`` and ``session_id`` are mutually exclusive.

**Turn boundaries decide reuse.** :meth:`AsyncClient._drive_turn` reports whether
a turn ended on EOF (the child exited) or on the configured
:attr:`~maestro_sdk.ClientMarkers.idle` redraw (the child stayed alive). A turn
that ended on idle leaves a reusable child, so the next query types into it; a
turn that ended on EOF spent the child, so the session reaps it and the next
query spawns a fresh one. A client with **no** idle marker therefore ends every
turn on EOF and degrades to one-spawn-per-turn — exactly the one-shot behaviour —
which is why a persistent interactive CLI must configure an idle marker or every
turn after the first would stall waiting for an EOF that never comes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from maestro_sdk.errors import RateLimitError, SessionExpiredError

if TYPE_CHECKING:
    from maestro_sdk.client import AsyncClient, _Runner
    from maestro_sdk.types import QueryOptions, QueryResult


class Session:
    """A persistent conversational session bound to one :class:`AsyncClient`.

    Holds at most one live runner at a time; reuses it across turns while the
    child survives and respawns transparently once a turn reaps it. A closed
    session refuses further queries.
    """

    def __init__(
        self,
        client: AsyncClient,
        *,
        initial_prompt: str | None = None,
        resume: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if resume is not None and session_id is not None:
            # ``resume`` reconnects an existing transcript; ``session_id`` registers
            # a fresh caller-pinned one. Contradictory spawn intents — refuse rather
            # than silently picking one.
            raise ValueError("resume and session_id are mutually exclusive: pass at most one")
        self._client = client
        self._runner: _Runner | None = None
        self._closed = False
        # Pending seed turn, sent ahead of the caller's first prompt and cleared
        # once delivered so it never re-fires on a later query.
        self._initial_prompt = initial_prompt
        # The spawn-pinned id (proposal 081): when set, the *first* spawn creates
        # the session with ``--session-id <id>`` and floors every turn's reported id.
        # ``None`` leaves spawns bare.
        self._pin_session_id = session_id
        # Whether the first spawn has happened. The first spawn of a pinned session
        # carries ``--session-id`` (registering the id); every later respawn carries
        # ``--resume``. Gating on this keeps the registering spawn distinct from the
        # reconnecting ones.
        self._has_spawned = False
        # The id the next spawn resumes from; seeded from ``resume`` (or the pinned
        # id so a crash-before-first-result respawn still reconnects rather than
        # spawning blank) and refreshed after every turn. ``None`` → spawn fresh.
        self._resume_id = resume if session_id is None else session_id

    async def query(self, prompt: str, *, options: QueryOptions | None = None) -> QueryResult:
        """Run one turn on this session, reusing the live child or spawning one.

        ``options`` overrides the client's defaults for this call; the runner is
        spawned with whichever options drive the turn that first brings it up.
        Only per-turn streaming bounds (``startup_timeout_seconds`` /
        ``stream_idle_timeout_seconds``) take effect on a *reused* runner —
        spawn-time fields (``cwd``, ``system``) bind at the first spawn and are
        ignored by later turns, since the child they would configure already exists.

        A pending ``initial_prompt`` is delivered as a preceding turn the first
        time the session is driven (its result is discarded, but its CLI-reported
        session id is captured for resume continuity); the caller's ``prompt`` is
        then the next turn and its result is the one returned.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        opts = options if options is not None else self._client._options
        if self._initial_prompt is not None:
            # Clear only *after* the seed turn returns: a turn that raises
            # (rate-limit/expired keep the runner alive; any other error reaps and
            # respawns) leaves the seed undelivered, so keeping it pending lets the
            # caller's retry re-send it rather than silently starting the
            # conversation without its connect-time seed.
            #
            # Drive the seed with the session's spawn-time factory options
            # (``self._client._options``), never the per-call ``opts``: the seed
            # predates any caller streaming, so a caller-scoped ``complete_on`` (or
            # a per-call idle override) must not govern it. A predicate satisfied by
            # the seed's streamed text before its idle redraw would trip an early
            # exit and reap the child mid-seed, leaving a fresh spawn to serve the
            # caller against a conversation that never received the full seed. The
            # factory options carry the baked-in spawn-time fields but no per-call
            # predicate — the same contract the ``run_command`` probe honours for
            # its non-caller turn. A future refactor must not collapse this back
            # onto the shared ``opts``.
            await self._run_turn(self._initial_prompt, self._client._options)
            self._initial_prompt = None
        return await self._run_turn(prompt, opts)

    async def run_command(self, text: str) -> str:
        """Issue a slash command as a probe on the live runner; return the redraw.

        Unlike :meth:`query`, a probe is **not a conversational turn**: it types
        ``text`` (e.g. ``/cost``) into the *already-live* runner this session keeps
        resident, reads the redraw to the idle marker through the same
        :meth:`AsyncClient._drive_turn` / :meth:`Screen.display` path a turn uses,
        and returns the rendered screen snapshot. It deliberately leaves the
        session's resume id and any pending seed untouched, so the next
        :meth:`query` continues the conversation exactly as if the probe had never
        run — the slash command's own output is scraped here, never folded into
        turn state. The metrics live in the TUI redraw, not the answer region, so
        the caller (a slash-command extractor) parses the returned snapshot.

        Requires a live runner: a probe has nothing to measure before a turn has
        brought the session up, so call it only after a :meth:`query`. A probe that
        ends on EOF or a ``complete_on`` early exit spent the child, and any other
        mid-probe error (a crash, a wedge, a cancellation) leaves it unusable; both
        reap the runner and clear the slot so the next query respawns — the
        untouched ``_resume_id`` (from the last real turn) carries the respawn back
        to the same transcript. A rate-limit or expired-session notice is raised
        with the child still alive (parity with :meth:`query`), so the runner stays
        installed for a retry.
        """
        if self._closed:
            raise RuntimeError("session is closed")
        runner = self._runner
        if runner is None:
            raise RuntimeError("cannot probe a session with no live runner; run a query first")
        try:
            result, ended_on_eof, exited_early = await self._client._drive_turn(
                runner, text, self._client._options
            )
        except (RateLimitError, SessionExpiredError):
            # Raised at a clean point with the child still alive, so keep the runner
            # installed for a retry — mirroring _run_turn. A probe never touches
            # _resume_id, so there is no captured id to preserve here.
            raise
        except BaseException:
            # Any other mid-probe exit — a dead, wedged, or cancelled child — leaves
            # a runner that cannot safely serve another turn. Reap and clear it so
            # the next query spawns fresh, with _resume_id (from the last real turn)
            # intact for the respawn. BaseException, not Exception, so a cancelled
            # probe reaps too.
            await self._reap_and_clear(runner)
            raise
        if ended_on_eof or exited_early:
            # The probe spent the child: a slash command should redraw to idle, but
            # an EOF (the child exited) or a complete_on early exit leaves it
            # unusable. Reap and let the next query respawn from the unchanged
            # _resume_id; the probe never refreshes it (a probe is not a turn).
            await self._reap_and_clear(runner)
        return result.raw_screen

    async def _run_turn(self, prompt: str, opts: QueryOptions) -> QueryResult:
        """Drive one turn: spawn (resuming if needed) or reuse, then send/stream/reap.

        Shared by the seed turn and every caller turn so the spawn/reuse,
        reap-on-failure, and resume-id bookkeeping live in one place.
        """
        runner = self._runner
        if runner is None:
            self._client._warn_if_instant()
            if self._pin_session_id is not None and not self._has_spawned:
                # First spawn of a pinned session: register the caller-chosen id
                # with --session-id (proposal 081). Later respawns fall to the
                # --resume branch, since _resume_id is seeded/updated to the pin.
                runner = self._client._spawn(opts, session_id=self._pin_session_id)
            else:
                runner = self._client._spawn(opts, resume=self._resume_id)
            try:
                await runner.start()
            except BaseException:
                # A spawn that fails part-way can still leave a child or master
                # fd to reap; mirror the one-shot path's "no path leaves the
                # child alive" guarantee rather than stranding the runner because
                # self._runner was not yet set. (BaseException, not Exception, so
                # a cancelled start reaps too.) ``_has_spawned`` stays unset so a
                # retry re-registers the pinned id rather than --resuming an id the
                # failed spawn may never have created.
                await runner.aclose()
                raise
            self._has_spawned = True
            self._runner = runner
        # A one-element list used as a side-channel so that any CLI-reported
        # session id captured inside _stream (before a condition check raises)
        # is visible here even when _drive_turn raises instead of returning.
        # Without this, a RateLimitError or SessionExpiredError raised on chunk
        # N would discard ids captured from chunks 1…N-1, leaving _resume_id
        # stale and causing a later respawn to lose the conversation.
        session_id_out: list[str | None] = [None]
        try:
            result, ended_on_eof, exited_early = await self._client._drive_turn(
                runner,
                prompt,
                opts,
                _session_id_out=session_id_out,
                pinned_session_id=self._pin_session_id,
            )
        except (RateLimitError, SessionExpiredError):
            # A rate-limit or expired-session notice is raised at a clean point
            # with the child still alive, so deliberately keep the runner
            # installed: the caller can retry this turn on the same session once
            # the condition clears.  Preserve any session id captured before the
            # error so a later respawn can still --resume the same transcript.
            if session_id_out[0] is not None:
                self._resume_id = session_id_out[0]
            raise
        except BaseException:
            # Any other mid-turn exit — a dead child (CLIExitedError), a wedged one
            # (CLITimeoutError), a denied approval, a cancellation, or an
            # unexpected error — leaves a runner that cannot safely serve another
            # turn (it may hold partial, unconsumed pty output). Reap it and clear
            # the slot so the next query spawns fresh, mirroring the one-shot
            # _execute's finally-aclose guarantee. BaseException, not Exception, so
            # a cancelled turn reaps too.
            await self._reap_and_clear(runner)
            raise
        if result.session_id is not None:
            # Refresh the resume id from the CLI-reported one so a later respawn
            # (after this turn's EOF, or a future crash) reconnects to the same
            # transcript instead of starting a blank conversation.
            self._resume_id = result.session_id
        if ended_on_eof or exited_early:
            # The child cannot serve another turn, so reap it now and let the next
            # query spawn fresh. Two cases reach here: EOF (the child exited at end
            # of turn — the path a client with no idle marker always takes, one
            # spawn per turn) and a ``complete_on`` early exit (the child is still
            # alive but mid-generation, holding unconsumed pty output that would
            # corrupt the next turn's read if reused). The refreshed ``_resume_id``
            # above lets the respawn reconnect to the same transcript.
            await self._reap_and_clear(runner)
        return result

    async def _reap_and_clear(self, runner: _Runner) -> None:
        """Reap ``runner`` and clear the session's slot, even if teardown raises.

        Every reap-and-clear site must leave ``self._runner is None`` so the next
        query spawns fresh rather than reusing a spent or broken child. Teardown
        walks a SIGTERM→SIGKILL ladder over a pty and can itself raise; the
        ``finally`` guarantees the slot clears whether or not ``aclose()`` fails,
        while the teardown exception (or, on the error paths, the in-flight
        exception it becomes the ``__context__`` of) still propagates loudly —
        nothing is swallowed, so the fail-loud invariant holds.
        """
        try:
            await runner.aclose()
        finally:
            self._runner = None

    async def aclose(self) -> None:
        """Close the session and reap the live child, if any; queries then raise.

        Walks the runner's existing SIGTERM→SIGKILL teardown ladder exactly once,
        so a session that ended its last turn on the idle marker leaks neither
        child nor descriptor.
        """
        self._closed = True
        # Reap the live child through the shared _reap_and_clear helper so a raising
        # aclose() (a non-EIO OSError or a teardown CancelledError) still propagates
        # but never leaves the broken runner installed on the closed session,
        # mirroring the turn/probe reap sites (issue #2400).
        runner = self._runner
        if runner is not None:
            await self._reap_and_clear(runner)
