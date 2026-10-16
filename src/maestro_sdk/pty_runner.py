"""PTY runner — spawns the ``claude`` CLI in a pseudo-terminal and drives it.

This is the lowest layer of the CLI-driving SDK: it owns the child process and
the master side of a pty, nothing more. It does not understand the screen, the
output, or approvals — those live in later slices. Its single responsibility is
to put bytes *into* the terminal at human cadence and hand raw bytes *out* of
it, with a teardown discipline strict enough that a cancelled query leaves no
orphaned child and no leaked file descriptor.

Why a pty rather than pipes: the ``claude`` CLI is a full-screen TUI. It only
draws its interactive interface when stdin/stdout is a terminal, and it sizes
that interface from the window dimensions. So the runner allocates a real pty,
sizes it from :meth:`Cadence.winsize`, and writes one keystroke at a time with
the per-key delay :class:`Cadence` prescribes — there is no bulk write anywhere.

Reading is delegated to a single background task that drains the master fd into
an unbounded queue; :meth:`aread_chunk` pops from that queue and reports EOF as
an empty ``bytes``. On Linux the master read raises ``OSError`` (EIO) once the
child closes its slave end rather than returning empty bytes, so both outcomes
are funnelled to the same end-of-stream signal.

Cancel discipline (the part that must not regress): :meth:`aclose` walks the
child through a ``SIGTERM`` → ``SIGKILL`` ladder, tears the reader task down to
completion, and closes the master fd — in that order, and idempotently — so a
caller cancelled mid-query can always reclaim the runner cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import os
import signal
import struct
import termios
from collections.abc import Mapping, Sequence

from maestro_sdk.cadence import Cadence
from maestro_sdk.errors import CLINotFoundError

_DEFAULT_READ_CHUNK = 4096
_DEFAULT_KILL_GRACE_SECONDS = 2.0


def _set_winsize(fd: int, cols: int, rows: int) -> None:
    """Apply ``(cols, rows)`` to the tty ``fd`` via the ``TIOCSWINSZ`` ioctl.

    The kernel ``struct winsize`` orders the fields ``rows`` then ``cols`` (then
    two pixel dimensions we leave zero), which is the reverse of how the rest of
    this package speaks of size — hence the explicit pack here rather than at
    the call site.
    """
    packed = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


class PtyRunner:
    """Drives a child process attached to a pseudo-terminal.

    Construct with the command to spawn and a :class:`Cadence`; call
    :meth:`start` to launch it, then interleave :meth:`send_char` and
    :meth:`aread_chunk`. Always pair construction with :meth:`aclose` (directly
    or via ``async with``) so the child and the master fd are reclaimed.
    """

    def __init__(
        self,
        command: Sequence[str],
        cadence: Cadence,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        read_chunk_size: int = _DEFAULT_READ_CHUNK,
        kill_grace_seconds: float = _DEFAULT_KILL_GRACE_SECONDS,
    ) -> None:
        if not command:
            raise ValueError("command must be a non-empty sequence")
        self._command = list(command)
        self._cadence = cadence
        self._cwd = cwd
        self._env = dict(env) if env is not None else None
        self._read_chunk_size = read_chunk_size
        self._kill_grace_seconds = kill_grace_seconds

        self._proc: asyncio.subprocess.Process | None = None
        self._master_fd: int | None = None
        self._reader_task: asyncio.Task[None] | None = None
        # Unbounded by design: this layer never blocks the child to apply
        # backpressure. A consumer that stops calling aread_chunk while the
        # child keeps emitting will grow this queue without bound — pacing
        # reads is the caller's responsibility (the client slice's), not the
        # runner's.
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._winsize = (0, 0)
        self._eof_seen = False
        self._closed = False

    # -- read surface ------------------------------------------------------

    @property
    def pid(self) -> int | None:
        """The child's PID, or ``None`` before :meth:`start`."""
        return self._proc.pid if self._proc is not None else None

    @property
    def returncode(self) -> int | None:
        """The child's exit code: ``None`` while running, negative if signalled."""
        return self._proc.returncode if self._proc is not None else None

    @property
    def winsize(self) -> tuple[int, int]:
        """The ``(cols, rows)`` drawn from :class:`Cadence` and applied to the pty."""
        return self._winsize

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Allocate the pty, size it, and spawn the child.

        Raises :class:`CLINotFoundError` if the command is not on ``PATH`` — a
        missing CLI is an operator-facing precondition failure, not a transient
        runtime error.

        Cleans up after itself on any failure: if the child has already spawned
        when a later step raises, it is walked through the kill ladder and the
        master fd is closed before the error propagates. That matters because
        ``start()`` runs inside ``__aenter__`` — when it raises, the ``async
        with`` body is never entered and :meth:`aclose` never runs, so ``start``
        is the only place that can reclaim its own partially-built state.
        """
        if self._proc is not None:
            raise RuntimeError("PtyRunner already started")
        self._loop = asyncio.get_running_loop()
        cols, rows = self._cadence.winsize()
        self._winsize = (cols, rows)
        master_fd, slave_fd = os.openpty()
        try:
            _set_winsize(slave_fd, cols, rows)
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *self._command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=self._cwd,
                    env=self._env,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise CLINotFoundError(f"command not found: {self._command[0]!r}") from exc
            os.set_blocking(master_fd, False)
            self._master_fd = master_fd
            self._reader_task = asyncio.create_task(self._read_loop())
        except BaseException:
            # A failure (or a cancellation) after the child is ours must not
            # leak it or the master fd. A cancellation can only land at the
            # spawn await, where ``self._proc`` is still None and asyncio has
            # already reaped its own half-built child — so the terminate below
            # is reachable only for a live child from a synchronous post-spawn
            # failure, and never awaits while a CancelledError is in flight.
            if self._proc is not None:
                await self._terminate_child(self._proc)
            os.close(master_fd)
            self._master_fd = None
            raise
        finally:
            # The parent never touches the slave end again; the child owns it.
            os.close(slave_fd)

    async def aclose(self) -> None:
        """Tear everything down, idempotently.

        Order matters: the child is signalled first (so a reader still mid-read
        unblocks as the pty drains), then the reader task is cancelled and
        awaited to completion so its fd registration is torn down, then the
        master fd is closed. Calling more than once is a no-op.

        The master fd is closed in a ``finally`` so the no-leaked-descriptor
        guarantee holds even on the error path: if the reader task surfaced a
        non-EIO ``OSError`` (rather than a clean EOF), awaiting it here
        re-raises that error, but the fd is reclaimed before it propagates.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc is not None:
                await self._terminate_child(self._proc)
            if self._reader_task is not None:
                self._reader_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._reader_task
                self._reader_task = None
        finally:
            if self._master_fd is not None:
                os.close(self._master_fd)
                self._master_fd = None

    async def __aenter__(self) -> PtyRunner:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _terminate_child(self, proc: asyncio.subprocess.Process) -> None:
        """Walk ``proc`` through the SIGTERM → SIGKILL ladder."""
        if proc.returncode is not None:
            return
        self._signal_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._kill_grace_seconds)
        except TimeoutError:
            self._signal_group(proc, signal.SIGKILL)
            await proc.wait()

    @staticmethod
    def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        """Signal the child's process group, tolerating an already-dead child.

        ``start_new_session=True`` makes the child its own group leader, so
        signalling the group reaches any grandchildren it spawned. A
        ``ProcessLookupError`` means the process is already gone — there is
        nothing left to signal, which is exactly the state we wanted.
        """
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), sig)

    # -- writing -----------------------------------------------------------

    async def send_char(self, char: str, prev: str | None) -> None:
        """Wait the cadence-prescribed delay, then type one ``char``.

        ``prev`` is the previously typed character (or ``None`` to start) so the
        :class:`Cadence` can lengthen the pause after word and sentence
        boundaries.
        """
        if self._master_fd is None:
            raise RuntimeError("PtyRunner not started")
        await asyncio.sleep(self._cadence.delay_for(char, prev))
        await self._write_all(char.encode("utf-8"))

    async def _write_all(self, data: bytes) -> None:
        """Write every byte of ``data`` to the master fd, pacing past a full buffer.

        The master fd is non-blocking (the reader needs it that way), so a write
        can land short or raise ``BlockingIOError`` when the child has not yet
        drained its input. Loop over partial writes and suspend until the fd is
        writable again — so a multi-byte keystroke is never torn mid-character
        and a stalled child cannot silently swallow input.
        """
        assert self._master_fd is not None  # send_char checked before calling
        view = memoryview(data)
        while view:
            try:
                written = os.write(self._master_fd, view)
            except BlockingIOError:
                await self._wait_writable(self._master_fd)
                continue
            view = view[written:]

    async def send_text(self, text: str, *, prev: str | None = None) -> str | None:
        """Type ``text`` character by character, threading ``prev`` between keys.

        Returns the last character typed so a caller can chain successive
        :meth:`send_text` calls and keep the boundary-pause behaviour correct.
        """
        for char in text:
            await self.send_char(char, prev)
            prev = char
        return prev

    # -- reading -----------------------------------------------------------

    async def aread_chunk(self) -> bytes:
        """Return the next chunk of output, or empty ``bytes`` at end of stream.

        EOF is sticky: once the stream ends, every subsequent call returns empty
        bytes immediately rather than blocking on a queue that will never fill.
        """
        if self._eof_seen:
            return b""
        chunk = await self._queue.get()
        if not chunk:
            self._eof_seen = True
        return chunk

    async def _read_loop(self) -> None:
        """Drain the master fd into the queue until EOF, then enqueue the sentinel."""
        try:
            while True:
                data = await self._await_readable()
                if not data:
                    break
                await self._queue.put(data)
        finally:
            await self._queue.put(b"")

    async def _await_readable(self) -> bytes:
        """Read the next chunk, suspending until the fd is readable.

        Returns empty ``bytes`` at end of stream — either a genuine empty read
        or the ``OSError`` (EIO on Linux) the kernel raises on the master once
        the child closes its slave end.
        """
        assert self._master_fd is not None  # set by start() before the reader runs
        while True:
            try:
                return os.read(self._master_fd, self._read_chunk_size)
            except BlockingIOError:
                await self._wait_readable(self._master_fd)
            except OSError as exc:
                # EIO is the kernel's way of reporting EOF on a pty master once
                # the child closes its slave end — the expected end-of-stream.
                # Any other OSError is unexpected and must not be disguised as a
                # clean close, so it propagates and fails the reader loudly.
                if exc.errno == errno.EIO:
                    return b""
                raise

    async def _wait_readable(self, fd: int) -> None:
        """Suspend until ``fd`` is readable, cleaning the reader registration off any exit."""
        assert self._loop is not None  # set by start() before the reader runs
        fut: asyncio.Future[None] = self._loop.create_future()
        self._loop.add_reader(fd, self._wake, fut)
        try:
            await fut
        finally:
            self._loop.remove_reader(fd)

    async def _wait_writable(self, fd: int) -> None:
        """Suspend until ``fd`` is writable, cleaning the writer registration off any exit."""
        assert self._loop is not None  # set by start() before any send
        fut: asyncio.Future[None] = self._loop.create_future()
        self._loop.add_writer(fd, self._wake, fut)
        try:
            await fut
        finally:
            self._loop.remove_writer(fd)

    @staticmethod
    def _wake(fut: asyncio.Future[None]) -> None:
        if not fut.done():
            fut.set_result(None)
