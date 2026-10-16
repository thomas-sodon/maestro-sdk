"""Tests for the marker-pack registry, selection, and the CLI version probe."""

from __future__ import annotations

import dataclasses
import subprocess

import pytest

from maestro_sdk import (
    ClientMarkers,
    MarkerPack,
    detect_cli_version,
    get_marker_pack,
    select_marker_pack,
)
from maestro_sdk import marker_packs as marker_packs_module


def _pack(name: str, low: str, high: str) -> MarkerPack:
    return MarkerPack(name=name, min_version=low, max_version=high)


# -- MarkerPack construction -------------------------------------------------


def test_marker_pack_is_frozen() -> None:
    pack = _pack("p", "1.0.0", "2.0.0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        pack.name = "other"  # type: ignore[misc]


def test_marker_pack_families_default_to_empty_markers() -> None:
    pack = _pack("p", "1.0.0", "2.0.0")
    assert pack.client == ClientMarkers()
    assert pack.client.idle == ()
    assert pack.parser.chrome == ()
    assert pack.approvals.prompt == ()
    assert pack.probe.usage_plan is None


def test_marker_pack_rejects_non_numeric_min_version() -> None:
    with pytest.raises(ValueError, match="dotted-numeric"):
        _pack("p", "v2.0", "2.1.0")


def test_marker_pack_rejects_non_numeric_max_version() -> None:
    with pytest.raises(ValueError, match="dotted-numeric"):
        _pack("p", "2.0.0", "latest")


def test_marker_pack_rejects_single_component_version() -> None:
    with pytest.raises(ValueError, match="dotted-numeric"):
        _pack("p", "2", "3.0")


def test_marker_pack_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        _pack("p", "2.1.0", "2.0.0")


# -- covers ------------------------------------------------------------------


def test_covers_version_inside_range() -> None:
    assert _pack("p", "2.0.0", "2.2.0").covers("2.1.172") is True


def test_covers_is_inclusive_at_both_bounds() -> None:
    pack = _pack("p", "2.0.0", "2.2.0")
    assert pack.covers("2.0.0") is True
    assert pack.covers("2.2.0") is True


def test_covers_rejects_below_min() -> None:
    assert _pack("p", "2.0.0", "2.2.0").covers("1.9.9") is False


def test_covers_rejects_above_max() -> None:
    assert _pack("p", "2.0.0", "2.2.0").covers("2.2.1") is False


def test_covers_rejects_unparseable_version() -> None:
    pack = _pack("p", "2.0.0", "2.2.0")
    assert pack.covers("two.point.one") is False
    assert pack.covers("") is False


def test_covers_shorter_max_bound_acts_as_prefix() -> None:
    # Documented tuple-comparison semantics: (2, 1, 172) > (2, 1), so a
    # max_version of "2.1" excludes "2.1.172" — authors write full bounds.
    assert _pack("p", "2.0", "2.1").covers("2.1.172") is False


# -- get_marker_pack ---------------------------------------------------------


def test_get_marker_pack_finds_by_exact_name() -> None:
    packs = [_pack("a", "1.0", "1.9"), _pack("b", "2.0", "2.9")]
    found = get_marker_pack("b", packs)
    assert found is packs[1]


def test_get_marker_pack_returns_none_for_unknown_name() -> None:
    packs = [_pack("a", "1.0", "1.9")]
    assert get_marker_pack("zzz", packs) is None


def test_get_marker_pack_defaults_to_module_registry() -> None:
    assert get_marker_pack("definitely-not-a-registered-pack") is None


# -- select_marker_pack ------------------------------------------------------


def test_select_returns_covering_pack() -> None:
    packs = [_pack("old", "1.0.0", "1.9.9"), _pack("new", "2.0.0", "2.9.9")]
    assert select_marker_pack("2.1.172", packs) is packs[1]


def test_select_prefers_newest_of_overlapping_packs() -> None:
    wide = _pack("wide", "1.0.0", "3.0.0")
    newer = _pack("newer", "2.0.0", "2.9.9")
    assert select_marker_pack("2.5.0", [wide, newer]) is newer
    assert select_marker_pack("2.5.0", [newer, wide]) is newer


def test_select_breaks_min_version_tie_by_max_version() -> None:
    short = _pack("short", "2.0.0", "2.1.0")
    long = _pack("long", "2.0.0", "2.9.9")
    assert select_marker_pack("2.0.5", [short, long]) is long


def test_select_returns_none_for_uncovered_version() -> None:
    packs = [_pack("p", "2.0.0", "2.2.0")]
    assert select_marker_pack("3.0.0", packs) is None


def test_select_returns_none_for_none_version() -> None:
    packs = [_pack("p", "2.0.0", "2.2.0")]
    assert select_marker_pack(None, packs) is None


def test_select_returns_none_for_unparseable_version() -> None:
    packs = [_pack("p", "2.0.0", "2.2.0")]
    assert select_marker_pack("garbage", packs) is None


def test_select_defaults_to_module_registry() -> None:
    assert select_marker_pack("0.0.0") is None


# -- detect_cli_version ------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_detect_cli_version_parses_dotted_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        calls.append((argv, kwargs))
        return _FakeCompleted(0, "2.1.172 (Claude Code)\n")

    monkeypatch.setattr(marker_packs_module.subprocess, "run", fake_run)
    assert detect_cli_version(("claude",)) == "2.1.172"
    argv, kwargs = calls[0]
    assert argv == ["claude", "--version"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == marker_packs_module._PROBE_TIMEOUT_SECONDS


def test_detect_cli_version_finds_version_inside_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        marker_packs_module.subprocess,
        "run",
        lambda argv, **kwargs: _FakeCompleted(0, "Claude Code v2.1.172\n"),
    )
    assert detect_cli_version() == "2.1.172"


def test_detect_cli_version_returns_none_on_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        raise FileNotFoundError("claude")

    monkeypatch.setattr(marker_packs_module.subprocess, "run", fake_run)
    assert detect_cli_version(("claude",)) is None


def test_detect_cli_version_returns_none_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=10.0)

    monkeypatch.setattr(marker_packs_module.subprocess, "run", fake_run)
    assert detect_cli_version(("claude",)) is None


def test_detect_cli_version_returns_none_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        raise OSError("exec format error")

    monkeypatch.setattr(marker_packs_module.subprocess, "run", fake_run)
    assert detect_cli_version(("claude",)) is None


def test_detect_cli_version_returns_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        marker_packs_module.subprocess,
        "run",
        lambda argv, **kwargs: _FakeCompleted(1, "2.1.172"),
    )
    assert detect_cli_version(("claude",)) is None


def test_detect_cli_version_returns_none_on_versionless_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        marker_packs_module.subprocess,
        "run",
        lambda argv, **kwargs: _FakeCompleted(0, "no version here"),
    )
    assert detect_cli_version(("claude",)) is None


def test_detect_cli_version_returns_none_for_empty_command() -> None:
    assert detect_cli_version(()) is None
