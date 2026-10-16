"""Tests for :class:`PtyRunner`.

These exercise the runner against a *real* pty allocated inside the test
process (``os.openpty`` via the runner's own ``start``), with the child replaced
by a small Python fixture script rather than the real ``claude`` CLI. That keeps
the tests hermetic — no network, no installed CLI — while still driving genuine
pty bytes, signals, and fd lifecycle, which is the whole point of this layer.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import signal
import struct
import sys
import termios

import pytest

from maestro_sdk import CLINotFoundError, InstantCadence, PtyRunner

# A child that puts its tty in raw mode (no echo, no line buffering) and
# uppercases everything it reads. Raw mode matters: in the default canonical
# mode the slave would not deliver input until a newline, so a byte-at-a-time
# round trip would hang. The case differs from the input so a test can be sure
# it is seeing the child's output, not a stray echo.
_ECHO_UPPER = """
import os, termios
attrs = termios.tcgetattr(0)
attrs[3] &= ~(termios.ECHO | termios.ICANON)
attrs[6][termios.VMIN] = 1
attrs[6][termios.VTIME] = 0
termios.tcsetattr(0, termios.TCSANOW, attrs)
while True:
    data = os.read(0, 4096)
    if not data:
        break
    os.write(1, data.upper())
"""

_SLEEP_FOREVER = "import time; time.sleep(3600)"

# Announces readiness only after the SIGTERM handler is installed, so a test
# can wait for it and be sure the first SIGTERM will be ignored — otherwise the
# default disposition would win the startup race and the SIGKILL escalation
# under test would never be exercised.
_IGNORE_SIGTERM = """
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.write(1, b"READY\\n")
time.sleep(3600)
"""

_QUICK_EXIT = "import sys; sys.exit(0)"


class RecordingCadence(InstantCadence):
    """An :class:`InstantCadence` that records every ``delay_for`` argument."""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "calls", [])

    def delay_for(self, char: str, prev: str | None) -> float:
        self.calls.append((char, prev))  # type: ignore[attr-defined]
        return 0.0


def _child(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def _read_pty_winsize(fd: int) -> tuple[int, int]:
    packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
    rows, cols, _xpix, _ypix = struct.unpack("HHHH", packed)
    return cols, rows


async def _read_until(runner: PtyRunner, needle: bytes, *, timeout: float = 4.0) -> bytes:
    acc = bytearray()

    async def _loop() -> None:
        while needle not in acc:
            chunk = await runner.aread_chunk()
            if not chunk:
                break
            acc.extend(chunk)

    await asyncio.wait_for(_loop(), timeout=timeout)
    return bytes(acc)


def test_construction_rejects_empty_command() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PtyRunner([], InstantCadence())


def test_pid_and_returncode_before_start() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    assert runner.pid is None
    assert runner.returncode is None
    assert runner.winsize == (0, 0)


@pytest.mark.asyncio
async def test_start_applies_cadence_winsize() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence(cols=77, rows=29))
    await runner.start()
    try:
        assert runner.winsize == (77, 29)
        assert runner.pid is not None
        assert runner.returncode is None
        assert _read_pty_winsize(runner._master_fd) == (77, 29)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_start_twice_raises() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            await runner.start()
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_missing_command_raises_cli_not_found_without_leaking_fds() -> None:
    runner = PtyRunner(["maestro-no-such-binary-xyz"], InstantCadence())
    before = set(os.listdir("/proc/self/fd"))
    with pytest.raises(CLINotFoundError, match="not found"):
        await runner.start()
    # Both the master and slave pty fds opened during the failed spawn must be
    # reclaimed — a failed start leaves nothing behind.
    assert set(os.listdir("/proc/self/fd")) == before


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_start_reclaims_spawned_child_when_post_spawn_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    before = set(os.listdir("/proc/self/fd"))
    real_set_blocking = os.set_blocking

    def boom(fd: int, blocking: bool) -> None:
        # Fail only the runner's own post-spawn call (once the child exists),
        # not any set_blocking asyncio itself makes while spawning.
        if runner._proc is not None:
            raise OSError(errno.EBADF, "bad file descriptor")
        real_set_blocking(fd, blocking)

    monkeypatch.setattr(os, "set_blocking", boom)
    with pytest.raises(OSError, match="bad file descriptor"):
        await runner.start()
    # The child spawned before the failure must be reaped and neither pty fd may
    # leak: start() reclaims its own partial state because __aexit__/aclose
    # never runs when start() raises out of __aenter__.
    assert runner.returncode is not None
    assert runner._master_fd is None
    assert set(os.listdir("/proc/self/fd")) == before


@pytest.mark.asyncio
async def test_send_char_before_start_raises() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    with pytest.raises(RuntimeError, match="not started"):
        await runner.send_char("a", None)


@pytest.mark.asyncio
async def test_send_text_threads_prev_through_cadence() -> None:
    cadence = RecordingCadence()
    runner = PtyRunner(_child(_ECHO_UPPER), cadence)
    await runner.start()
    try:
        last = await runner.send_text("ab")
        assert last == "b"
        assert cadence.calls == [("a", None), ("b", "a")]  # type: ignore[attr-defined]
    finally:
        await runner.aclose()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_round_trip_echo() -> None:
    runner = PtyRunner(_child(_ECHO_UPPER), InstantCadence())
    await runner.start()
    try:
        await runner.send_text("hello")
        out = await _read_until(runner, b"HELLO")
        assert b"HELLO" in out
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_send_char_reassembles_partial_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    try:
        out = bytearray()
        counts = iter([1, 1])  # force the two-byte char out one byte per write

        def chunked(fd: int, data: memoryview) -> int:
            n = next(counts, len(bytes(data)))
            out.extend(bytes(data)[:n])
            return n

        monkeypatch.setattr(os, "write", chunked)
        await runner.send_char("é", None)  # 0xC3 0xA9 — two UTF-8 bytes
        # A short write must not tear the character: every byte still lands.
        assert bytes(out) == "é".encode()
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_send_char_retries_after_blocking_io(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    try:
        real_write = os.write
        attempts = {"n": 0}

        def flaky(fd: int, data: memoryview) -> int:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise BlockingIOError
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", flaky)
        await runner.send_char("x", None)
        # The first write raised EAGAIN; the runner waited for writability and
        # retried rather than dropping the keystroke or surfacing the error.
        assert attempts["n"] == 2
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_await_readable_reraises_non_eio_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    runner._master_fd = 999  # os.read is mocked; the value is never dereferenced

    def boom(fd: int, n: int) -> bytes:
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(os, "read", boom)
    # Only EIO means EOF on a pty master; any other OSError must propagate
    # rather than masquerade as a clean end of stream.
    with pytest.raises(OSError, match="not permitted"):
        await runner._await_readable()


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_aclose_kills_running_child_and_closes_fd() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    fd = runner._master_fd
    assert fd is not None
    await runner.aclose()
    assert runner.returncode is not None
    assert runner._master_fd is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.asyncio
async def test_aclose_is_idempotent() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    await runner.aclose()
    # Second call must be a clean no-op, not a double-close or double-kill.
    await runner.aclose()


@pytest.mark.asyncio
async def test_aclose_without_start_is_safe() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.aclose()
    assert runner.returncode is None


@pytest.mark.asyncio
async def test_aclose_closes_fd_even_when_reader_raised() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())

    async def _boom() -> None:
        raise OSError(errno.EPERM, "operation not permitted")

    task = asyncio.ensure_future(_boom())
    runner._reader_task = task
    with pytest.raises(OSError, match="not permitted"):
        await task  # drive the reader task to a done-with-exception state

    fd = os.open(os.devnull, os.O_RDONLY)
    runner._master_fd = fd  # no real start(); _proc stays None so terminate is skipped
    # aclose must surface the reader's error, but only after reclaiming the fd:
    # the no-leaked-descriptor guarantee has to hold on the error path too.
    with pytest.raises(OSError, match="not permitted"):
        await runner.aclose()
    assert runner._master_fd is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_cancel_mid_read_then_aclose_reclaims_child_and_fd() -> None:
    runner = PtyRunner(_child(_SLEEP_FOREVER), InstantCadence())
    await runner.start()
    fd = runner._master_fd
    assert fd is not None

    reader = asyncio.ensure_future(runner.aread_chunk())
    await asyncio.sleep(0)  # let the read suspend on the empty queue
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader

    # A query cancelled mid-read still reclaims everything through aclose:
    # no orphaned child, no leaked descriptor.
    await runner.aclose()
    assert runner.returncode is not None
    assert runner._master_fd is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_kill_on_timeout_escalates_to_sigkill() -> None:
    runner = PtyRunner(
        _child(_IGNORE_SIGTERM),
        InstantCadence(),
        kill_grace_seconds=0.1,
    )
    await runner.start()
    await _read_until(runner, b"READY")
    await runner.aclose()
    assert runner.returncode == -signal.SIGKILL


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_terminate_skips_already_exited_child() -> None:
    runner = PtyRunner(_child(_QUICK_EXIT), InstantCadence())
    await runner.start()
    assert runner._proc is not None
    await runner._proc.wait()
    # The child is already gone; aclose must reach the early-return in the
    # terminate ladder rather than re-signalling a reaped pid.
    await runner.aclose()
    assert runner.returncode == 0


@pytest.mark.timeout(15)
@pytest.mark.asyncio
async def test_signal_group_tolerates_reaped_child() -> None:
    runner = PtyRunner(_child(_QUICK_EXIT), InstantCadence())
    await runner.start()
    assert runner._proc is not None
    await runner._proc.wait()
    # Signalling a reaped pid raises ProcessLookupError internally, which the
    # runner swallows because "already dead" is the desired end state.
    runner._signal_group(runner._proc, signal.SIGTERM)
    await runner.aclose()


@pytest.mark.asyncio
async def test_eof_is_sticky_after_close() -> None:
    runner = PtyRunner(_child(_QUICK_EXIT), InstantCadence())
    await runner.start()
    await runner.aclose()
    assert await runner.aread_chunk() == b""
    # A second read does not block on the drained queue: EOF is remembered.
    assert await runner.aread_chunk() == b""


@pytest.mark.asyncio
async def test_async_context_manager() -> None:
    async with PtyRunner(_child(_SLEEP_FOREVER), InstantCadence()) as runner:
        assert runner.pid is not None
    assert runner.returncode is not None
    assert runner._master_fd is None


def test_wake_sets_result_at_most_once() -> None:
    loop = asyncio.new_event_loop()
    try:
        fut: asyncio.Future[None] = loop.create_future()
        PtyRunner._wake(fut)
        assert fut.done()
        assert fut.result() is None
        # A second wake must not raise InvalidStateError on the resolved future.
        PtyRunner._wake(fut)
        assert fut.done()
    finally:
        loop.close()
