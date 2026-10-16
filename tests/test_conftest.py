"""Tests for the ``RUN_CLAUDE_TESTS`` opt-in gate in ``conftest.py``.

Tests that drive the real ``claude`` CLI carry ``@pytest.mark.requires_claude_cli``
and must be skipped unless ``RUN_CLAUDE_TESTS=1``. The hook is exercised directly
with stand-in items, so no real CLI test is ever collected or run here.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import REQUIRES_CLAUDE_CLI, RUN_CLAUDE_TESTS_ENV, pytest_collection_modifyitems


class _FakeItem:
    """The two members of ``pytest.Item`` the hook touches."""

    def __init__(self, *keywords: str) -> None:
        self.keywords = {keyword: True for keyword in keywords}
        self.added: list[Any] = []

    def add_marker(self, marker: Any) -> None:
        self.added.append(marker)


def _run_hook(*items: _FakeItem) -> None:
    pytest_collection_modifyitems(list(items))  # type: ignore[arg-type]


def test_cli_tests_are_skipped_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUN_CLAUDE_TESTS_ENV, raising=False)
    cli_item = _FakeItem(REQUIRES_CLAUDE_CLI)
    plain_item = _FakeItem()
    _run_hook(cli_item, plain_item)
    assert len(cli_item.added) == 1
    assert cli_item.added[0].name == "skip"
    assert plain_item.added == []


def test_cli_tests_run_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_CLAUDE_TESTS_ENV, "1")
    cli_item = _FakeItem(REQUIRES_CLAUDE_CLI)
    _run_hook(cli_item)
    assert cli_item.added == []


def test_opt_in_requires_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_CLAUDE_TESTS_ENV, "true")
    cli_item = _FakeItem(REQUIRES_CLAUDE_CLI)
    _run_hook(cli_item)
    assert len(cli_item.added) == 1
