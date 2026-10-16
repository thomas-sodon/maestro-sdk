"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from maestro_sdk import (
    ApprovalRequiredError,
    CLIExitedError,
    CLINotFoundError,
    CLITimeoutError,
    MaestroSDKError,
    RateLimitError,
    SessionExpiredError,
)


@pytest.mark.parametrize(
    "cls",
    [
        CLINotFoundError,
        CLITimeoutError,
        ApprovalRequiredError,
        RateLimitError,
        SessionExpiredError,
        CLIExitedError,
    ],
)
def test_subclasses_of_base(cls: type[MaestroSDKError]) -> None:
    assert issubclass(cls, MaestroSDKError)


def test_message_round_trip() -> None:
    err = CLITimeoutError("ready prompt not seen within 30s")
    assert str(err) == "ready prompt not seen within 30s"


def test_screen_snapshot_default_is_none() -> None:
    err = CLIExitedError("pty EOF")
    assert err.screen is None


def test_screen_snapshot_attached_when_passed() -> None:
    snapshot = "$ claude\n> "
    err = RateLimitError("rate limited", screen=snapshot)
    assert err.screen == snapshot


@pytest.mark.parametrize(
    ("cls", "expected_category"),
    [
        (MaestroSDKError, "unknown"),
        (CLINotFoundError, "auth"),
        (CLITimeoutError, "stream_stall"),
        (ApprovalRequiredError, "unknown"),
        (RateLimitError, "rate_limit"),
        (SessionExpiredError, "stale_session"),
        (CLIExitedError, "cli_crash"),
    ],
)
def test_default_category_mirrors_error_category_by_name(
    cls: type[MaestroSDKError], expected_category: str
) -> None:
    # The category strings mirror maestro.sdk.ErrorCategory by name so the
    # selection seam can re-pack them without this package importing maestro.
    assert cls("boom").category == expected_category


def test_category_override_wins_over_default() -> None:
    # A startup CLITimeoutError is "timeout"; the same class defaults to
    # "stream_stall" for the mid-stream case, so the raiser overrides per call.
    err = CLITimeoutError("startup timed out", category="timeout")
    assert err.category == "timeout"


def test_category_override_coexists_with_screen() -> None:
    err = CLITimeoutError("stalled", screen="frozen", category="stream_stall")
    assert err.category == "stream_stall"
    assert err.screen == "frozen"
