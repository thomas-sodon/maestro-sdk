"""``--mcp-config`` JSON builder for the ``claude`` CLI (proposal 061 Step 4).

The ``claude`` CLI reaches MCP servers through a ``--mcp-config <file>`` flag
whose payload is plain JSON: a top-level ``{"mcpServers": {<name>: <spec>}}``
object where each *spec* describes one server. For an stdio server the spec is
``{"command": str, "args": [str, ...], "env": {str: str}}`` — the same
``command`` / ``args`` / ``env`` shape :meth:`maestro.config.Config.mcp_servers_config`
already produces for the agent path.

This module is the **config-generation** seam, deliberately nothing more. It
builds and serialises that JSON and knows the wire format; it does *not* spawn
the CLI, host a server, or speak MCP. Keeping it pure JSON honours
:mod:`maestro_sdk`'s zero-dependency posture (it imports neither ``mcp`` nor
``maestro``), so the provider-side adapter
(:mod:`maestro.providers.cli_paced_tool_adapter`) can assemble a registry into
a ``--mcp-config`` without dragging a transport dependency into the SDK.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

#: Top-level key the ``claude`` CLI reads server specs from in a ``--mcp-config``
#: payload. A constant so the adapter and its tests pin the same spelling.
MCP_SERVERS_KEY = "mcpServers"


def stdio_server(
    command: str,
    *,
    args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one stdio MCP server spec for a ``--mcp-config`` payload.

    ``command`` is always emitted; ``args`` and ``env`` are included only when
    provided, so a bare ``stdio_server("cmd")`` produces ``{"command": "cmd"}``
    rather than a spec padded with empty collections. Mirrors the
    ``{command, args, env}`` shape :meth:`maestro.config.Config.mcp_servers_config`
    emits for the agent path, so external MCP entries fold in unchanged
    (proposal 061 Step 5).
    """
    spec: dict[str, Any] = {"command": command}
    if args is not None:
        spec["args"] = list(args)
    if env is not None:
        spec["env"] = dict(env)
    return spec


def build_mcp_config(servers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Wrap per-server specs into the CLI's ``{"mcpServers": {...}}`` envelope.

    Server insertion order is preserved (each spec is shallow-copied so the
    returned config does not alias the caller's input), giving a deterministic
    payload that is trivial to regression-pin.
    """
    return {MCP_SERVERS_KEY: {name: dict(spec) for name, spec in servers.items()}}


def serialize_mcp_config(config: Mapping[str, Any]) -> str:
    """Serialise a ``--mcp-config`` payload to the JSON text the CLI consumes.

    Two-space indented and key-order-preserving (``sort_keys=False``) so the
    emitted file stays human-diffable and matches :func:`build_mcp_config`'s
    deterministic ordering.
    """
    return json.dumps(config, indent=2, sort_keys=False)


__all__ = ["MCP_SERVERS_KEY", "build_mcp_config", "serialize_mcp_config", "stdio_server"]
