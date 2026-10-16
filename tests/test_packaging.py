"""Packaging-metadata smoke tests.

Pins PEP 561 declarations so a future packaging refactor cannot silently drop
them. Downstream type-checking (e.g. ``mypy maestro/`` over the
``maestro.providers.cli_paced`` provider, which imports ``maestro_sdk`` real
types) depends on the ``py.typed`` marker being present alongside the
installed package.
"""

from __future__ import annotations

from pathlib import Path

import maestro_sdk


def test_py_typed_marker_ships_with_package() -> None:
    """The inline ``py.typed`` marker must live next to the installed package
    so PEP 561 consumers see ``maestro_sdk`` as typed."""
    assert maestro_sdk.__file__ is not None
    pkg_dir = Path(maestro_sdk.__file__).parent
    assert (pkg_dir / "py.typed").exists()
