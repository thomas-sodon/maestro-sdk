"""Tests for the vt100-subset screen emulator.

The emulator is a pure function over bytes, so every test here is a table test:
feed a byte stream, assert the rendered text, cursor, or scroll region. No I/O,
no sleep, no subprocess. ESC sequences are written as raw bytes to keep the
inputs honest about what arrives off a pty.
"""

from __future__ import annotations

import pytest

from maestro_sdk import Screen

# ---------------------------------------------------------------------------
# Construction


def test_rejects_zero_columns() -> None:
    with pytest.raises(ValueError, match="at least 1x1"):
        Screen(cols=0, rows=4)


def test_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="at least 1x1"):
        Screen(cols=4, rows=0)


def test_fresh_screen_is_blank_and_homed() -> None:
    screen = Screen(cols=10, rows=3)
    assert screen.display() == ""
    assert (screen.cursor_row, screen.cursor_col) == (0, 0)
    assert screen.scroll_region == (0, 2)


# ---------------------------------------------------------------------------
# Plain text and the C0 controls


def test_plain_text_advances_cursor() -> None:
    screen = Screen(cols=20, rows=2)
    screen.feed(b"hello")
    assert screen.display() == "hello"
    assert (screen.cursor_row, screen.cursor_col) == (0, 5)


def test_carriage_return_overwrites_from_column_zero() -> None:
    screen = Screen(cols=20, rows=2)
    screen.feed(b"hello\rH")
    assert screen.display() == "Hello"
    assert screen.cursor_col == 1


def test_line_feed_moves_down_without_resetting_column() -> None:
    screen = Screen(cols=20, rows=3)
    screen.feed(b"ab\ncd")
    # LF moves down only; the column is preserved, matching raw pty output.
    assert screen.display() == "ab\n  cd"
    assert (screen.cursor_row, screen.cursor_col) == (1, 4)


def test_crlf_starts_next_line_at_column_zero() -> None:
    screen = Screen(cols=20, rows=3)
    screen.feed(b"ab\r\ncd")
    assert screen.display() == "ab\ncd"


def test_backspace_moves_left_and_stops_at_margin() -> None:
    screen = Screen(cols=20, rows=2)
    screen.feed(b"abc\b\b\b\b\bX")
    # Three backspaces reach column 0; the extra two are clamped.
    assert screen.display() == "Xbc"
    assert screen.cursor_col == 1


def test_tab_advances_to_next_eight_column_stop() -> None:
    screen = Screen(cols=40, rows=2)
    screen.feed(b"ab\tc")
    assert screen.cursor_col == 9  # tab from col 2 -> col 8, then 'c' -> col 9
    assert screen.display() == "ab      c"


def test_tab_clamps_at_right_margin() -> None:
    screen = Screen(cols=5, rows=2)
    screen.feed(b"ab\t")
    assert screen.cursor_col == 4  # next stop (8) clamps to cols-1


def test_unhandled_control_byte_is_ignored() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"a\x07b")  # BEL between letters
    assert screen.display() == "ab"


# ---------------------------------------------------------------------------
# Deferred autowrap


def test_last_column_defers_wrap_until_next_char() -> None:
    screen = Screen(cols=3, rows=3)
    screen.feed(b"abc")
    # 'c' lands in the last column; the cursor is pinned, not yet wrapped.
    assert screen.display() == "abc"
    assert (screen.cursor_row, screen.cursor_col) == (0, 2)
    screen.feed(b"d")
    assert screen.display() == "abc\nd"
    assert (screen.cursor_row, screen.cursor_col) == (1, 1)


def test_full_width_line_then_crlf_does_not_double_advance() -> None:
    screen = Screen(cols=3, rows=4)
    screen.feed(b"abc\r\nxyz")
    # Without deferred wrap this would skip a row; with it the rows are adjacent.
    assert screen.display() == "abc\nxyz"


# ---------------------------------------------------------------------------
# Cursor positioning (CUU/CUD/CUF/CUB, CUP)


def test_cursor_up_and_down_clamp_to_screen() -> None:
    screen = Screen(cols=10, rows=4)
    screen.feed(b"\x1b[2B")  # down 2
    assert screen.cursor_row == 2
    screen.feed(b"\x1b[10B")  # down past bottom -> clamp
    assert screen.cursor_row == 3
    screen.feed(b"\x1b[1A")  # up 1
    assert screen.cursor_row == 2
    screen.feed(b"\x1b[10A")  # up past top -> clamp
    assert screen.cursor_row == 0


def test_cursor_forward_and_back_default_to_one() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"\x1b[C")  # forward, default 1
    assert screen.cursor_col == 1
    screen.feed(b"\x1b[5C")  # forward 5
    assert screen.cursor_col == 6
    screen.feed(b"\x1b[2D")  # back 2
    assert screen.cursor_col == 4
    screen.feed(b"\x1b[99D")  # back past margin -> clamp
    assert screen.cursor_col == 0


def test_cursor_forward_clamps_to_right_margin() -> None:
    screen = Screen(cols=5, rows=2)
    screen.feed(b"\x1b[99C")
    assert screen.cursor_col == 4


@pytest.mark.parametrize("final", [b"H", b"f"])
def test_cup_positions_one_based(final: bytes) -> None:
    screen = Screen(cols=10, rows=5)
    screen.feed(b"\x1b[3;4" + final)
    assert (screen.cursor_row, screen.cursor_col) == (2, 3)


def test_cup_without_params_homes_cursor() -> None:
    screen = Screen(cols=10, rows=5)
    screen.feed(b"\x1b[3;4H")
    screen.feed(b"\x1b[H")
    assert (screen.cursor_row, screen.cursor_col) == (0, 0)


def test_cup_clamps_out_of_range_target() -> None:
    screen = Screen(cols=4, rows=3)
    screen.feed(b"\x1b[99;99H")
    assert (screen.cursor_row, screen.cursor_col) == (2, 3)


# ---------------------------------------------------------------------------
# Erase in display (ED) and erase in line (EL)


def test_ed_default_clears_from_cursor_to_end() -> None:
    screen = Screen(cols=5, rows=3)
    screen.feed(b"aaaaa\r\nbbbbb\r\nccccc")
    screen.feed(b"\x1b[2;3H")  # row 2, col 3 (0-based 1,2)
    screen.feed(b"\x1b[J")  # default mode 0
    assert screen.display() == "aaaaa\nbb"


def test_ed_mode_one_clears_from_start_to_cursor() -> None:
    screen = Screen(cols=5, rows=3)
    screen.feed(b"aaaaa\r\nbbbbb\r\nccccc")
    screen.feed(b"\x1b[2;3H")
    screen.feed(b"\x1b[1J")
    assert screen.display() == "\n   bb\nccccc"


def test_ed_mode_two_clears_whole_screen() -> None:
    screen = Screen(cols=5, rows=3)
    screen.feed(b"aaaaa\r\nbbbbb\r\nccccc")
    screen.feed(b"\x1b[2J")
    assert screen.display() == ""


def test_el_default_clears_to_end_of_line() -> None:
    screen = Screen(cols=8, rows=2)
    screen.feed(b"abcdefgh\r\x1b[3C")  # back to col 0 via CR, forward 3
    screen.feed(b"\x1b[K")
    assert screen.display() == "abc"


def test_el_mode_one_clears_to_cursor() -> None:
    screen = Screen(cols=8, rows=2)
    screen.feed(b"abcdefgh")
    screen.feed(b"\x1b[1;4H")  # col 4 (0-based 3)
    screen.feed(b"\x1b[1K")
    assert screen.display() == "    efgh"


def test_el_mode_two_clears_whole_line() -> None:
    screen = Screen(cols=8, rows=2)
    screen.feed(b"abcdefgh\r\nsecond")
    screen.feed(b"\x1b[1;1H\x1b[2K")
    assert screen.display() == "\nsecond"


# ---------------------------------------------------------------------------
# Scroll region (DECSTBM) and scrolling


def test_scroll_region_is_set_and_homes_cursor() -> None:
    screen = Screen(cols=10, rows=5)
    screen.feed(b"\x1b[2;4r")
    assert screen.scroll_region == (1, 3)
    assert (screen.cursor_row, screen.cursor_col) == (0, 0)


def test_scroll_region_without_params_resets_to_full_screen() -> None:
    screen = Screen(cols=10, rows=5)
    screen.feed(b"\x1b[2;4r")
    screen.feed(b"\x1b[r")
    assert screen.scroll_region == (0, 4)


def test_inverted_scroll_region_is_rejected() -> None:
    screen = Screen(cols=10, rows=5)
    screen.feed(b"\x1b[4;2r")  # top below bottom
    assert screen.scroll_region == (0, 4)


def test_line_feed_at_bottom_margin_scrolls_region() -> None:
    screen = Screen(cols=5, rows=4)
    screen.feed(b"r0\r\nr1\r\nr2\r\nr3")
    screen.feed(b"\x1b[1;3r")  # scroll region rows 0..2; homes cursor
    screen.feed(b"\x1b[3;1H")  # to bottom of region (0-based row 2)
    screen.feed(b"\nX")  # LF scrolls the region, X writes on the new bottom line
    # Rows 0..2 scrolled up (r0 lost, r1/r2 rise); row 3 is outside the region.
    assert screen.display() == "r1\nr2\nX\nr3"


def test_line_feed_at_last_row_scrolls_full_screen_region() -> None:
    screen = Screen(cols=5, rows=2)
    screen.feed(b"top\r\nbot")
    # The default scroll region is the whole screen, so LF at the last row
    # scrolls: "top" is lost, "bot" rises, and the new bottom line takes "more".
    screen.feed(b"\r\nmore")
    assert screen.display() == "bot\nmore"


# ---------------------------------------------------------------------------
# SGR, private sequences, and malformed input


def test_sgr_is_parsed_and_discarded() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"\x1b[1;31mred\x1b[0m")
    assert screen.display() == "red"


def test_csi_with_intermediate_byte_and_unknown_final_is_ignored() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"\x1b[ qtext")  # space is an intermediate byte; 'q' unsupported
    assert screen.display() == "text"


def test_private_sequence_is_ignored() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"hi\x1b[?25lthere")  # hide-cursor: no effect on text or cursor
    assert screen.display() == "hithere"


def test_unsupported_escape_introducer_is_swallowed() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"a\x1bcb")  # ESC c (reset) is unsupported; 'c' is the introducer
    assert screen.display() == "ab"


def test_non_numeric_csi_parameter_defaults_to_zero() -> None:
    screen = Screen(cols=10, rows=3)
    screen.feed(b"\x1b[:H")  # ':' is not a number -> arg 0 -> CUP homes
    assert (screen.cursor_row, screen.cursor_col) == (0, 0)


def test_malformed_csi_aborts_and_reprocesses_byte() -> None:
    screen = Screen(cols=10, rows=3)
    # A LF interrupts the CSI: the sequence is abandoned and the LF executes.
    screen.feed(b"a\x1b[1\nb")
    assert screen.display() == "a\n b"
    assert screen.cursor_row == 1


# ---------------------------------------------------------------------------
# UTF-8 decoding


def test_two_byte_utf8_renders_one_char() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed("café".encode())
    assert screen.display() == "café"
    assert screen.cursor_col == 4


def test_three_byte_utf8_renders_one_char() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed("a€b".encode())
    assert screen.display() == "a€b"
    assert screen.cursor_col == 3


def test_four_byte_utf8_renders_one_char() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed("x😀y".encode())
    assert screen.display() == "x😀y"
    assert screen.cursor_col == 3


def test_truncated_utf8_lead_is_stashed_across_feeds() -> None:
    screen = Screen(cols=10, rows=2)
    encoded = "é".encode()  # two bytes
    screen.feed(encoded[:1])  # lead only
    assert screen.display() == ""  # nothing rendered yet
    screen.feed(encoded[1:])  # continuation
    assert screen.display() == "é"


def test_stray_continuation_byte_becomes_replacement() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"a\xa9b")  # 0xa9 is a stray continuation byte
    assert screen.display() == "a�b"


def test_invalid_utf8_continuation_emits_replacement_and_resyncs() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"\xc3A")  # 2-byte lead followed by ASCII 'A' (not a continuation)
    assert screen.display() == "�A"


def test_invalid_high_lead_byte_becomes_replacement() -> None:
    screen = Screen(cols=10, rows=2)
    screen.feed(b"\xffz")  # 0xff is never a valid UTF-8 lead
    assert screen.display() == "�z"


# ---------------------------------------------------------------------------
# Reset and re-render discrimination


def test_reset_returns_to_blank_homed_state() -> None:
    screen = Screen(cols=10, rows=3)
    screen.feed(b"some text\x1b[2;4r")
    screen.reset()
    assert screen.display() == ""
    assert (screen.cursor_row, screen.cursor_col) == (0, 0)
    assert screen.scroll_region == (0, 2)


def test_rerender_to_identical_content_compares_equal() -> None:
    # The whole point of the emulator: a repaint that lands on the same content
    # yields the same display(), so the parser can detect "no new text."
    first = Screen(cols=10, rows=3)
    first.feed(b"line one\r\nline two")
    second = Screen(cols=10, rows=3)
    second.feed(b"\x1b[2Jline one\r\nline two\x1b[Hline one\r\nline two")
    assert first.display() == second.display()
