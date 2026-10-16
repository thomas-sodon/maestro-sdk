"""Tests for the approval handler.

Detection is a pure function of one screen snapshot and resolution is a pure
function of the request and the policy, so every test here is a table test: build
a synthetic modal snapshot, then assert the recognised :class:`ApprovalRequest`,
the keystrokes a permissive policy yields, or the :class:`ApprovalRequiredError`
the deny-by-default posture raises. No I/O, no sleep, no subprocess.

Markers are the layout policy; tests pass explicit markers so they pin the
*mechanism* rather than any one CLI version's glyphs. A couple of glyphy markers
appear verbatim to prove detection is byte-faithful after SGR stripping.
"""

from __future__ import annotations

import pytest

from maestro_sdk import (
    DEFAULT_APPROVAL_MARKERS,
    ApprovalHandler,
    ApprovalMarkers,
    ApprovalOption,
    ApprovalRequest,
    ApprovalRequiredError,
    approve_all,
)

# A layout tuned to a Claude-Code-shaped modal; tests pin the mechanism with it.
_MARKERS = ApprovalMarkers(
    prompt=("Do you want to proceed",),
    option=r"^>?\s*(?P<key>\d+)\.\s*(?P<label>.+)$",
    tool=r"^[⏺●]?\s*(?P<name>[A-Za-z][\w-]*) command$",
    approve_keys=("1",),
)

_MODAL = (
    "⏺ Bash command\n"
    "ls -la\n"
    "\n"
    "Do you want to proceed?\n"
    "> 1. Yes\n"
    "  2. Yes, and don't ask again\n"
    "  3. No, and tell Claude what to do (esc)\n"
)


# ---------------------------------------------------------------------------
# ApprovalMarkers


def test_default_markers_match_nothing() -> None:
    markers = DEFAULT_APPROVAL_MARKERS
    assert markers.prompt == ()
    assert markers.option is None
    assert markers.tool is None
    assert markers.approve_keys == ()


def test_compiled_option_is_none_without_pattern() -> None:
    assert ApprovalMarkers().compiled_option() is None


def test_compiled_option_compiles_pattern() -> None:
    pattern = ApprovalMarkers(option=r"^(?P<key>\d)\.(?P<label>.+)$").compiled_option()
    assert pattern is not None
    assert pattern.match("1.Yes") is not None


def test_compiled_option_rejects_pattern_without_key_group() -> None:
    with pytest.raises(ValueError, match=r"key.*label"):
        ApprovalMarkers(option=r"^(?P<label>.+)$").compiled_option()


def test_compiled_option_rejects_pattern_without_label_group() -> None:
    with pytest.raises(ValueError, match=r"key.*label"):
        ApprovalMarkers(option=r"^(?P<key>\d)$").compiled_option()


def test_compiled_tool_is_none_without_pattern() -> None:
    assert ApprovalMarkers().compiled_tool() is None


def test_compiled_tool_compiles_pattern() -> None:
    pattern = ApprovalMarkers(tool=r"^(?P<name>\w+) command$").compiled_tool()
    assert pattern is not None
    assert pattern.match("Bash command") is not None


def test_compiled_tool_rejects_pattern_without_name_group() -> None:
    with pytest.raises(ValueError, match="must define a"):
        ApprovalMarkers(tool=r"^\w+ command$").compiled_tool()


def test_handler_rejects_bad_markers_at_construction() -> None:
    # The handler compiles both patterns in __init__, so a misconfigured marker
    # fails loudly when the handler is built, not mid-stream during a query.
    with pytest.raises(ValueError, match=r"key.*label"):
        ApprovalHandler(ApprovalMarkers(option=r"^(?P<key>\d)$"))


# ---------------------------------------------------------------------------
# Detection


def test_default_markers_detect_no_modal() -> None:
    # Without prompt markers, even a modal-shaped snapshot is not a modal.
    handler = ApprovalHandler()
    assert handler.detect(_MODAL) is None


def test_detect_returns_request_with_prompt_options_and_tool() -> None:
    handler = ApprovalHandler(_MARKERS)
    request = handler.detect(_MODAL)
    assert request == ApprovalRequest(
        prompt="Do you want to proceed?",
        options=(
            ApprovalOption(key="1", label="Yes"),
            ApprovalOption(key="2", label="Yes, and don't ask again"),
            ApprovalOption(key="3", label="No, and tell Claude what to do (esc)"),
        ),
        tool="Bash",
        screen=_MODAL,
    )


def test_detect_returns_none_when_no_prompt_line() -> None:
    handler = ApprovalHandler(_MARKERS)
    assert handler.detect("just streaming output\nno modal here") is None


def test_detect_without_tool_pattern_leaves_tool_none() -> None:
    markers = ApprovalMarkers(prompt=("Proceed?",), option=r"^(?P<key>\d)\.\s*(?P<label>.+)$")
    handler = ApprovalHandler(markers)
    request = handler.detect("Proceed?\n1. Yes\n2. No")
    assert request is not None
    assert request.tool is None
    assert request.options == (ApprovalOption(key="1", label="Yes"), ApprovalOption(key="2", label="No"))


def test_detect_without_option_pattern_leaves_options_empty() -> None:
    handler = ApprovalHandler(ApprovalMarkers(prompt=("Proceed?",)))
    request = handler.detect("Proceed?\n1. Yes")
    assert request is not None
    assert request.options == ()


def test_detect_takes_first_prompt_line() -> None:
    handler = ApprovalHandler(ApprovalMarkers(prompt=("Proceed",)))
    request = handler.detect("Proceed first?\nProceed second?")
    assert request is not None
    assert request.prompt == "Proceed first?"


def test_detect_takes_first_tool_match() -> None:
    markers = ApprovalMarkers(prompt=("Proceed?",), tool=r"^(?P<name>\w+) command$")
    handler = ApprovalHandler(markers)
    request = handler.detect("Bash command\nEdit command\nProceed?")
    assert request is not None
    assert request.tool == "Bash"


# ---------------------------------------------------------------------------
# Resolution — deny-by-default posture


def test_resolve_returns_none_when_no_modal() -> None:
    handler = ApprovalHandler(_MARKERS, policy=approve_all)
    assert handler.resolve("plain streaming text") is None


def test_resolve_without_policy_raises() -> None:
    handler = ApprovalHandler(_MARKERS)
    with pytest.raises(ApprovalRequiredError, match="no policy configured") as exc:
        handler.resolve(_MODAL)
    assert exc.value.screen == _MODAL
    assert "tool: Bash" in str(exc.value)


def test_resolve_with_denying_policy_raises() -> None:
    handler = ApprovalHandler(_MARKERS, policy=lambda request: False)
    with pytest.raises(ApprovalRequiredError, match="denied by policy"):
        handler.resolve(_MODAL)


def test_deny_message_omits_tool_when_absent() -> None:
    handler = ApprovalHandler(ApprovalMarkers(prompt=("Proceed?",)))
    with pytest.raises(ApprovalRequiredError) as exc:
        handler.resolve("Proceed?")
    assert "tool:" not in str(exc.value)


# ---------------------------------------------------------------------------
# Resolution — approval


def test_resolve_with_permissive_policy_returns_keystroke() -> None:
    handler = ApprovalHandler(_MARKERS, policy=approve_all)
    assert handler.resolve(_MODAL) == "1"


def test_approve_all_is_permissive() -> None:
    assert approve_all(ApprovalRequest(prompt="anything")) is True


def test_resolve_selects_first_option_without_approve_keys() -> None:
    markers = ApprovalMarkers(prompt=("Proceed?",), option=r"^(?P<key>\d)\.\s*(?P<label>.+)$")
    handler = ApprovalHandler(markers, policy=approve_all)
    assert handler.resolve("Proceed?\n7. Approve\n8. Deny") == "7"


def test_resolve_selects_configured_approve_key() -> None:
    markers = ApprovalMarkers(
        prompt=("Proceed?",),
        option=r"^(?P<key>\d)\.\s*(?P<label>.+)$",
        approve_keys=("2",),
    )
    handler = ApprovalHandler(markers, policy=approve_all)
    assert handler.resolve("Proceed?\n1. No\n2. Yes") == "2"


def test_resolve_raises_when_approved_but_no_options() -> None:
    handler = ApprovalHandler(ApprovalMarkers(prompt=("Proceed?",)), policy=approve_all)
    with pytest.raises(ApprovalRequiredError, match="no approval option"):
        handler.resolve("Proceed?")


def test_resolve_raises_when_no_option_matches_approve_keys() -> None:
    markers = ApprovalMarkers(
        prompt=("Proceed?",),
        option=r"^(?P<key>\d)\.\s*(?P<label>.+)$",
        approve_keys=("9",),
    )
    handler = ApprovalHandler(markers, policy=approve_all)
    with pytest.raises(ApprovalRequiredError, match="no approval option"):
        handler.resolve("Proceed?\n1. Yes\n2. No")
