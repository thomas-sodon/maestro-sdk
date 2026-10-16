"""Screen emulator — a minimal vt100 subset, pure over bytes.

The CLI repaints its terminal: it does not append text, it moves the cursor,
erases regions, and re-renders. To tell *incremental streaming* apart from a
*re-render*, the parser (a later slice) needs a stable picture of what the
terminal would show, not the raw byte stream. This module is that picture.

It is deliberately a *subset*: cursor movement, ED/EL erasure, the scroll
region (DECSTBM), and SGR (parsed then discarded — colour does not change the
text). Anything else is consumed and ignored rather than mis-rendered. There is
no I/O here and no dependency on a third-party emulator (``pyte`` et al.) so the
package keeps ``dependencies = []``; the cost is that exotic escapes are
swallowed, which is acceptable for the streaming/re-render discrimination this
exists to serve.

:class:`Screen` is *stateful across feeds* on purpose: a split escape sequence
or a truncated UTF-8 lead byte at the end of one ``feed`` is carried into the
next, mirroring how bytes actually arrive off a pty.
"""

from __future__ import annotations

import enum

# C0 control bytes we act on; every other control byte is ignored.
_BS = 0x08
_TAB = 0x09
_LF = 0x0A
_CR = 0x0D
_ESC = 0x1B

_TAB_WIDTH = 8
_REPLACEMENT = "�"


class _State(enum.Enum):
    """Escape-sequence parser state, persisted across ``feed`` calls."""

    GROUND = enum.auto()
    ESC = enum.auto()
    CSI = enum.auto()


class Screen:
    """A fixed-size character grid driven by a byte stream.

    ``cols`` and ``rows`` fix the grid; both must be at least 1. Feed raw pty
    bytes with :meth:`feed`; read the rendered text with :meth:`display` and
    the cursor with :attr:`cursor_row` / :attr:`cursor_col`.
    """

    def __init__(self, cols: int, rows: int) -> None:
        if cols < 1 or rows < 1:
            raise ValueError(f"screen must be at least 1x1, got {cols}x{rows}")
        self.cols = cols
        self.rows = rows
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        """Clear the grid, home the cursor, reset scroll region and parser."""
        self._buffer: list[list[str]] = [self._blank_row() for _ in range(self.rows)]
        self._cursor_row = 0
        self._cursor_col = 0
        self._scroll_top = 0
        self._scroll_bottom = self.rows - 1
        self._pending_wrap = False
        self._state = _State.GROUND
        self._csi_params = bytearray()
        self._pending_bytes = bytearray()

    def _blank_row(self) -> list[str]:
        return [" "] * self.cols

    # -- read surface ------------------------------------------------------

    @property
    def cursor_row(self) -> int:
        return self._cursor_row

    @property
    def cursor_col(self) -> int:
        return self._cursor_col

    @property
    def scroll_region(self) -> tuple[int, int]:
        """``(top, bottom)`` row indices, 0-based and inclusive."""
        return self._scroll_top, self._scroll_bottom

    def display(self) -> str:
        """Rendered text: rows joined by newlines, trailing blanks trimmed.

        Trailing whitespace on each line and trailing all-blank lines are
        dropped so two equivalent screens compare equal regardless of how much
        empty grid sits to the right of or below the content.
        """
        lines = ["".join(row).rstrip() for row in self._buffer]
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    # -- write surface -----------------------------------------------------

    def feed(self, data: bytes) -> None:
        """Advance the emulator by ``data`` bytes."""
        buf = bytes(self._pending_bytes) + data
        self._pending_bytes = bytearray()
        i = 0
        n = len(buf)
        while i < n:
            b = buf[i]
            if self._state is _State.GROUND:
                if b == _ESC:
                    self._state = _State.ESC
                    i += 1
                elif b < 0x20:
                    self._exec_control(b)
                    i += 1
                elif b < 0x80:
                    self._put(chr(b))
                    i += 1
                else:
                    consumed = self._feed_utf8(buf, i)
                    if consumed == 0:
                        # Truncated multibyte lead at the buffer tail: stash the
                        # remainder so the next feed can complete it.
                        self._pending_bytes = bytearray(buf[i:])
                        return
                    i += consumed
            elif self._state is _State.ESC:
                if b == 0x5B:  # '['
                    self._state = _State.CSI
                    self._csi_params = bytearray()
                else:
                    # Unsupported ESC introducer (ESC c, ESC 7, ...): swallow it.
                    self._state = _State.GROUND
                i += 1
            else:  # _State.CSI
                if 0x30 <= b <= 0x3F:  # parameter bytes: digits ; ? : < = >
                    self._csi_params.append(b)
                    i += 1
                elif 0x20 <= b <= 0x2F:  # intermediate bytes: collected, ignored
                    i += 1
                elif 0x40 <= b <= 0x7E:  # final byte: dispatch
                    self._dispatch_csi(chr(b))
                    self._state = _State.GROUND
                    i += 1
                else:
                    # Malformed: abort the sequence and reprocess this byte in
                    # GROUND (do not advance i).
                    self._state = _State.GROUND

    def _feed_utf8(self, buf: bytes, i: int) -> int:
        """Decode one UTF-8 char at ``buf[i]``; return bytes consumed (0 if truncated)."""
        b = buf[i]
        if 0xC0 <= b <= 0xDF:
            length = 2
        elif 0xE0 <= b <= 0xEF:
            length = 3
        elif 0xF0 <= b <= 0xF7:
            length = 4
        else:
            # Stray continuation byte or invalid lead: emit replacement, resync.
            self._put(_REPLACEMENT)
            return 1
        if i + length > len(buf):
            return 0
        try:
            self._put(buf[i : i + length].decode("utf-8"))
        except UnicodeDecodeError:
            # Bad continuation bytes: emit replacement, consume only the lead.
            self._put(_REPLACEMENT)
            return 1
        return length

    # -- control characters ------------------------------------------------

    def _exec_control(self, b: int) -> None:
        if b == _CR:
            self._cursor_col = 0
            self._pending_wrap = False
        elif b == _LF:
            self._index()
            self._pending_wrap = False
        elif b == _BS:
            if self._cursor_col > 0:
                self._cursor_col -= 1
            self._pending_wrap = False
        elif b == _TAB:
            self._cursor_col = min(self.cols - 1, (self._cursor_col // _TAB_WIDTH + 1) * _TAB_WIDTH)
            self._pending_wrap = False
        # Every other control byte (NUL, BEL, ...) is ignored.

    def _put(self, ch: str) -> None:
        """Write one printable char at the cursor with deferred autowrap.

        Deferred (rather than immediate) wrap matches real terminals: a char in
        the last column leaves the cursor pinned there until the *next*
        printable arrives. This keeps full-width lines followed by CR/LF from
        double-advancing — exactly the case a re-rendering TUI produces.
        """
        if self._pending_wrap:
            self._cursor_col = 0
            self._index()
            self._pending_wrap = False
        self._buffer[self._cursor_row][self._cursor_col] = ch
        if self._cursor_col + 1 >= self.cols:
            self._pending_wrap = True
        else:
            self._cursor_col += 1

    def _index(self) -> None:
        """Move down one row, scrolling the region if at its bottom margin."""
        if self._cursor_row == self._scroll_bottom:
            self._scroll_up()
        elif self._cursor_row < self.rows - 1:
            self._cursor_row += 1
        # Below the scroll region at the last row: clamp (no movement).

    def _scroll_up(self) -> None:
        region = self._buffer[self._scroll_top : self._scroll_bottom + 1]
        self._buffer[self._scroll_top : self._scroll_bottom + 1] = [*region[1:], self._blank_row()]

    # -- CSI dispatch ------------------------------------------------------

    def _csi_args(self) -> tuple[list[int], bool]:
        """Parse collected CSI parameters into ints; report the private marker.

        Empty / non-numeric parameters become ``0`` so callers can apply their
        own per-command default. A leading ``?`` marks a private sequence.
        """
        raw = self._csi_params.decode("ascii", errors="ignore")
        private = raw.startswith("?")
        if private:
            raw = raw[1:]
        args: list[int] = []
        for part in raw.split(";"):
            try:
                args.append(int(part))
            except ValueError:
                args.append(0)
        return args, private

    def _dispatch_csi(self, final: str) -> None:
        args, private = self._csi_args()
        if private:
            # Cursor show/hide, alt-screen toggles, etc.: outside the subset.
            return
        if final == "A":  # CUU — cursor up
            self._cursor_row = max(0, self._cursor_row - max(1, args[0]))
            self._pending_wrap = False
        elif final == "B":  # CUD — cursor down
            self._cursor_row = min(self.rows - 1, self._cursor_row + max(1, args[0]))
            self._pending_wrap = False
        elif final == "C":  # CUF — cursor forward
            self._cursor_col = min(self.cols - 1, self._cursor_col + max(1, args[0]))
            self._pending_wrap = False
        elif final == "D":  # CUB — cursor back
            self._cursor_col = max(0, self._cursor_col - max(1, args[0]))
            self._pending_wrap = False
        elif final in ("H", "f"):  # CUP — cursor position (1-based row;col)
            row = args[0] or 1
            col = (args[1] if len(args) > 1 else 0) or 1
            self._cursor_row = min(self.rows - 1, max(0, row - 1))
            self._cursor_col = min(self.cols - 1, max(0, col - 1))
            self._pending_wrap = False
        elif final == "J":  # ED — erase in display
            self._erase_display(args[0])
        elif final == "K":  # EL — erase in line
            self._erase_line(args[0])
        elif final == "r":  # DECSTBM — set scroll region
            self._set_scroll_region(args)
        elif final == "m":  # SGR — colour/style, intentionally discarded
            pass
        # Any other final byte is an unsupported command: ignored.

    def _erase_display(self, mode: int) -> None:
        if mode == 1:  # start of screen to cursor (inclusive)
            for y in range(self._cursor_row):
                self._buffer[y] = self._blank_row()
            for x in range(self._cursor_col + 1):
                self._buffer[self._cursor_row][x] = " "
        elif mode == 2:  # entire screen
            self._buffer = [self._blank_row() for _ in range(self.rows)]
        else:  # mode 0 (default): cursor to end of screen
            for x in range(self._cursor_col, self.cols):
                self._buffer[self._cursor_row][x] = " "
            for y in range(self._cursor_row + 1, self.rows):
                self._buffer[y] = self._blank_row()

    def _erase_line(self, mode: int) -> None:
        if mode == 1:  # start of line to cursor (inclusive)
            for x in range(self._cursor_col + 1):
                self._buffer[self._cursor_row][x] = " "
        elif mode == 2:  # entire line
            self._buffer[self._cursor_row] = self._blank_row()
        else:  # mode 0 (default): cursor to end of line
            for x in range(self._cursor_col, self.cols):
                self._buffer[self._cursor_row][x] = " "

    def _set_scroll_region(self, args: list[int]) -> None:
        top = args[0] or 1
        bottom = (args[1] if len(args) > 1 else 0) or self.rows
        top0 = top - 1
        bottom0 = bottom - 1
        if 0 <= top0 < bottom0 <= self.rows - 1:
            self._scroll_top = top0
            self._scroll_bottom = bottom0
            # DECSTBM homes the cursor.
            self._cursor_row = 0
            self._cursor_col = 0
            self._pending_wrap = False
        # An inverted or out-of-range region is rejected, leaving state intact.
