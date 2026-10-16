"""Shared pytest configuration.

Tests marked ``requires_claude_cli`` drive the real ``claude`` CLI and are
opt-in: they are skipped unless ``RUN_CLAUDE_TESTS=1`` is set, so the default
suite (and CI) never spawns the installed CLI or touches the network.
"""

from __future__ import annotations

import os

import pytest

RUN_CLAUDE_TESTS_ENV = "RUN_CLAUDE_TESTS"
REQUIRES_CLAUDE_CLI = "requires_claude_cli"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``requires_claude_cli`` tests unless ``RUN_CLAUDE_TESTS=1``."""
    if os.environ.get(RUN_CLAUDE_TESTS_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"drives the real claude CLI; set {RUN_CLAUDE_TESTS_ENV}=1 to run")
    for item in items:
        if REQUIRES_CLAUDE_CLI in item.keywords:
            item.add_marker(skip)
