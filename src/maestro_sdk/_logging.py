"""Logging shim.

A single root logger named ``maestro_sdk``; child loggers are obtained via
``get_logger(name)`` and are namespaced as ``maestro_sdk.<name>``. The package
configures no handlers and no level — callers own logging configuration.
"""

from __future__ import annotations

import logging

_ROOT = "maestro_sdk"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{name}")
