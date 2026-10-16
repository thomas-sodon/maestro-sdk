"""Shared pytest fixtures.

This slice has no I/O, so the fixture set is small. Later slices will add the
``FakePtyRunner`` and recording-replay machinery here.
"""

from __future__ import annotations

import pytest

from maestro_sdk import Cadence


@pytest.fixture()
def seeded_cadence() -> Cadence:
    """A ``Cadence`` with a fixed seed for reproducible tests."""
    return Cadence(seed=12345)
