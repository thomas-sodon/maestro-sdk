"""maestro_sdk — a CLI-driving Claude SDK that paces interaction like a human user.

This ships the foundation layer (Cadence, errors, types), the screen emulator,
the pty runner, the output parser, the approval handler, the public
``AsyncClient`` that composes them into a query, and the persistent ``Session``
that keeps one runner alive across turns — reconnecting to the CLI's server-side
transcript across a respawn via ``--resume`` (seeded by
``create_session(resume=...)`` and refreshed from each turn's reported
``session_id``).
"""

from __future__ import annotations

from maestro_sdk.approvals import (
    DEFAULT_APPROVAL_MARKERS,
    ApprovalHandler,
    ApprovalMarkers,
    ApprovalOption,
    ApprovalPolicy,
    ApprovalRequest,
    approve_all,
)
from maestro_sdk.cadence import Cadence, InstantCadence
from maestro_sdk.client import (
    DEFAULT_CLIENT_MARKERS,
    AsyncClient,
    ClientMarkers,
    RunnerFactory,
)
from maestro_sdk.errors import (
    ApprovalRequiredError,
    CLIExitedError,
    CLINotFoundError,
    CLITimeoutError,
    MaestroSDKError,
    RateLimitError,
    SessionExpiredError,
)
from maestro_sdk.marker_packs import (
    MARKER_PACKS,
    MarkerPack,
    detect_cli_version,
    get_marker_pack,
    select_marker_pack,
)
from maestro_sdk.mcp_config import (
    MCP_SERVERS_KEY,
    build_mcp_config,
    serialize_mcp_config,
    stdio_server,
)
from maestro_sdk.parser import DEFAULT_MARKERS, Parser, ParserMarkers
from maestro_sdk.pty_runner import PtyRunner
from maestro_sdk.screen import Screen
from maestro_sdk.session import Session
from maestro_sdk.slash_probe import (
    DEFAULT_PROBE_MARKERS,
    SlashProbeMarkers,
    parse_context,
    parse_cost,
    parse_usage,
)
from maestro_sdk.types import (
    ContextBucket,
    ContextReport,
    CostReport,
    QueryOptions,
    QueryResult,
    QueryTiming,
    QueryUsage,
    StreamDelta,
    ToolCall,
    UsageReport,
)

__all__ = [
    "DEFAULT_APPROVAL_MARKERS",
    "DEFAULT_CLIENT_MARKERS",
    "DEFAULT_MARKERS",
    "DEFAULT_PROBE_MARKERS",
    "MARKER_PACKS",
    "MCP_SERVERS_KEY",
    "ApprovalHandler",
    "ApprovalMarkers",
    "ApprovalOption",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalRequiredError",
    "AsyncClient",
    "CLIExitedError",
    "CLINotFoundError",
    "CLITimeoutError",
    "Cadence",
    "ClientMarkers",
    "ContextBucket",
    "ContextReport",
    "CostReport",
    "InstantCadence",
    "MaestroSDKError",
    "MarkerPack",
    "Parser",
    "ParserMarkers",
    "PtyRunner",
    "QueryOptions",
    "QueryResult",
    "QueryTiming",
    "QueryUsage",
    "RateLimitError",
    "RunnerFactory",
    "Screen",
    "Session",
    "SessionExpiredError",
    "SlashProbeMarkers",
    "StreamDelta",
    "ToolCall",
    "UsageReport",
    "approve_all",
    "build_mcp_config",
    "detect_cli_version",
    "get_marker_pack",
    "parse_context",
    "parse_cost",
    "parse_usage",
    "select_marker_pack",
    "serialize_mcp_config",
    "stdio_server",
]
