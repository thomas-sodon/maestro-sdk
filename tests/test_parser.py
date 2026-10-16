"""Tests for the output parser.

The parser is a pure fold over a stream of screen snapshots, so every test here
is a table test: feed a sequence of `display()`-style strings and assert the
emitted `StreamDelta`s, the accumulated `response`, or the collected
`tool_calls`. No I/O, no sleep, no subprocess.

`ParserMarkers` is the layout policy; tests pass explicit markers so they pin the
*mechanism* rather than any one CLI version's glyphs. A couple of glyphy markers
appear verbatim to prove the parser is byte-faithful after SGR stripping.
"""

from __future__ import annotations

import pytest

from maestro_sdk import DEFAULT_MARKERS, Parser, ParserMarkers, StreamDelta, ToolCall

# ---------------------------------------------------------------------------
# ParserMarkers


def test_default_markers_match_nothing() -> None:
    markers = DEFAULT_MARKERS
    assert markers.thinking_open == ()
    assert markers.thinking_close == ()
    assert markers.tool_call is None
    assert markers.chrome == ()


def test_compiled_tool_call_is_none_without_pattern() -> None:
    assert ParserMarkers().compiled_tool_call() is None


def test_compiled_tool_call_compiles_pattern() -> None:
    pattern = ParserMarkers(tool_call=r"^(?P<name>\w+)$").compiled_tool_call()
    assert pattern is not None
    assert pattern.match("Bash") is not None


def test_compiled_tool_call_rejects_pattern_without_name_group() -> None:
    # Extraction reads the "name" group unconditionally, so a pattern that omits
    # it must fail loudly at construction, not mid-stream during a query.
    markers = ParserMarkers(tool_call=r"^\w+\(.*\)$")
    with pytest.raises(ValueError, match="must define a"):
        markers.compiled_tool_call()


def test_parser_rejects_markers_without_name_group_at_construction() -> None:
    with pytest.raises(ValueError, match="must define a"):
        Parser(ParserMarkers(tool_call=r"^\w+$"))


# ---------------------------------------------------------------------------
# Degenerate (default-markers) behaviour: everything is answer


def test_default_markers_treat_all_text_as_answer() -> None:
    parser = Parser()
    deltas = parser.feed("hello world")
    assert deltas == [StreamDelta(text="hello world", kind="answer", screen_snapshot="hello world")]
    assert parser.response == "hello world"
    assert parser.tool_calls == []


def test_default_markers_do_not_extract_tool_calls() -> None:
    # A line that looks like a tool call is plain text without a tool pattern.
    parser = Parser()
    parser.feed("Bash(ls -la)")
    assert parser.tool_calls == []
    assert parser.response == "Bash(ls -la)"


# ---------------------------------------------------------------------------
# Incremental diffing


def test_append_only_growth_emits_only_the_suffix() -> None:
    parser = Parser()
    first = parser.feed("Hello")
    second = parser.feed("Hello world")
    assert first == [StreamDelta(text="Hello", kind="answer", screen_snapshot="Hello")]
    assert second == [StreamDelta(text=" world", kind="answer", screen_snapshot="Hello world")]
    assert parser.response == "Hello world"


def test_identical_repaint_emits_nothing() -> None:
    parser = Parser()
    parser.feed("steady")
    assert parser.feed("steady") == []
    assert parser.response == "steady"


def test_repaint_to_less_content_emits_nothing() -> None:
    # The new snapshot is a prefix of the old one: common prefix is the whole new
    # text, so there is no suffix to emit and the response keeps what streamed.
    parser = Parser()
    parser.feed("abcdef")
    assert parser.feed("abc") == []
    assert parser.response == "abcdef"


def test_shrink_then_regrow_does_not_duplicate() -> None:
    # A transient shrink during repaint must not cause the tail to be
    # re-emitted when the region grows back. If _prev were overwritten to the
    # shorter snapshot, the regrowth would diff against the shrunken prefix
    # and re-append the suffix to the response.
    parser = Parser()
    parser.feed("abcdef")
    parser.feed("abc")
    parser.feed("abcdef")
    assert parser.response == "abcdef"


def test_mid_string_rewrite_emits_the_divergent_tail() -> None:
    # Documented behaviour: after the common prefix, the rest of the new snapshot
    # is treated as freshly revealed even though it overwrote earlier text.
    parser = Parser()
    parser.feed("Hello world")
    deltas = parser.feed("Hello there")
    assert deltas == [StreamDelta(text="there", kind="answer", screen_snapshot="Hello there")]
    assert parser.response == "Hello worldthere"


def test_empty_snapshot_emits_nothing() -> None:
    parser = Parser()
    assert parser.feed("") == []
    assert parser.response == ""


# ---------------------------------------------------------------------------
# Thinking / answer region discrimination

_THINKING = ParserMarkers(thinking_open=("✻ Thinking",), thinking_close=("⏺",))


def test_thinking_region_is_classified_and_excluded_from_response() -> None:
    parser = Parser(_THINKING)
    snapshot = "✻ Thinking\nweigh the options\n⏺\nthe answer"
    deltas = parser.feed(snapshot)
    assert deltas == [
        StreamDelta(text="weigh the options", kind="thinking", screen_snapshot=snapshot),
        StreamDelta(text="the answer", kind="answer", screen_snapshot=snapshot),
    ]
    assert parser.response == "the answer"


def test_thinking_before_answer_within_one_feed() -> None:
    # Both regions grow in a single feed; thinking deltas come first.
    parser = Parser(_THINKING)
    deltas = parser.feed("✻ Thinking\nreason\n⏺\nanswer")
    assert [d.kind for d in deltas] == ["thinking", "answer"]


def test_marker_lines_are_consumed_not_emitted() -> None:
    parser = Parser(_THINKING)
    parser.feed("✻ Thinking\nr\n⏺\na")
    # Neither the open nor the close marker text appears in any region.
    assert "Thinking" not in parser.response
    assert "⏺" not in parser.response


def test_answer_only_when_no_thinking_marker_seen() -> None:
    parser = Parser(_THINKING)
    deltas = parser.feed("just an answer")
    assert deltas == [StreamDelta(text="just an answer", kind="answer", screen_snapshot="just an answer")]


def test_thinking_streams_incrementally_across_feeds() -> None:
    parser = Parser(_THINKING)
    parser.feed("✻ Thinking\nstep one")
    deltas = parser.feed("✻ Thinking\nstep one\nstep two")
    assert deltas == [
        StreamDelta(text="\nstep two", kind="thinking", screen_snapshot="✻ Thinking\nstep one\nstep two"),
    ]


def test_close_marker_in_answer_region_is_plain_text() -> None:
    # The close marker only closes when we are *in* the thinking region; seen in
    # the answer region with no open before it, it is ordinary answer text.
    parser = Parser(ParserMarkers(thinking_open=("THINK",), thinking_close=("END",)))
    parser.feed("END of story")
    assert parser.response == "END of story"


# ---------------------------------------------------------------------------
# Tool-call extraction

_TOOLS = ParserMarkers(tool_call=r"^[⏺●]\s*(?P<name>[A-Za-z][\w-]*)\((?P<args>.*)\)$")


def test_tool_call_is_extracted_with_raw_args() -> None:
    parser = Parser(_TOOLS)
    parser.feed("answer above\n⏺ Bash(ls -la)\nanswer below")
    assert parser.tool_calls == [ToolCall(name="Bash", args={"raw": "ls -la"})]
    # The tool line is not part of the answer text.
    assert parser.response == "answer above\nanswer below"


def test_tool_call_line_is_not_emitted_as_a_delta() -> None:
    parser = Parser(_TOOLS)
    deltas = parser.feed("⏺ Read(/etc/hosts)")
    assert deltas == []
    assert parser.tool_calls == [ToolCall(name="Read", args={"raw": "/etc/hosts"})]


def test_repeated_tool_line_is_recorded_once() -> None:
    parser = Parser(_TOOLS)
    parser.feed("⏺ Bash(ls)")
    parser.feed("⏺ Bash(ls)")  # repaint of the same line
    assert parser.tool_calls == [ToolCall(name="Bash", args={"raw": "ls"})]


def test_distinct_tool_lines_are_each_recorded() -> None:
    parser = Parser(_TOOLS)
    parser.feed("⏺ Bash(ls)\n● Read(a.txt)")
    assert parser.tool_calls == [
        ToolCall(name="Bash", args={"raw": "ls"}),
        ToolCall(name="Read", args={"raw": "a.txt"}),
    ]


def test_tool_call_without_args_group_has_empty_args() -> None:
    parser = Parser(ParserMarkers(tool_call=r"^>>>(?P<name>\w+)$"))
    parser.feed(">>>Compile")
    assert parser.tool_calls == [ToolCall(name="Compile", args={})]


def test_tool_call_with_empty_args_string_has_empty_args() -> None:
    # An args group that captures the empty string is falsy, so no "raw" key.
    parser = Parser(ParserMarkers(tool_call=r"^(?P<name>\w+)\((?P<args>.*)\)$"))
    parser.feed("Noop()")
    assert parser.tool_calls == [ToolCall(name="Noop", args={})]


def test_non_matching_line_is_not_a_tool_call() -> None:
    parser = Parser(_TOOLS)
    parser.feed("this mentions Bash(ls) mid-sentence")
    assert parser.tool_calls == []
    assert parser.response == "this mentions Bash(ls) mid-sentence"


# ---------------------------------------------------------------------------
# Chrome filtering


def test_chrome_lines_are_dropped_entirely() -> None:
    parser = Parser(ParserMarkers(chrome=("│", "? for shortcuts")))
    parser.feed("│ input box\nreal answer\n? for shortcuts")
    assert parser.response == "real answer"


def test_chrome_does_not_count_as_thinking_or_answer() -> None:
    markers = ParserMarkers(thinking_open=("THINK",), chrome=("---",))
    parser = Parser(markers)
    deltas = parser.feed("THINK\n---\nreasoning")
    assert deltas == [StreamDelta(text="reasoning", kind="thinking", screen_snapshot="THINK\n---\nreasoning")]


# ---------------------------------------------------------------------------
# Response invariant


def test_response_equals_concatenation_of_answer_deltas() -> None:
    parser = Parser()
    answer_text = ""
    for snapshot in ["a", "ab", "ab", "abc"]:
        for delta in parser.feed(snapshot):
            if delta.kind == "answer":
                answer_text += delta.text
    assert answer_text == parser.response == "abc"


def test_tool_calls_property_returns_a_copy() -> None:
    parser = Parser(_TOOLS)
    parser.feed("⏺ Bash(ls)")
    snapshot = parser.tool_calls
    snapshot.clear()
    # Mutating the returned list does not disturb the parser's own record.
    assert parser.tool_calls == [ToolCall(name="Bash", args={"raw": "ls"})]
