"""Tests for the ``--mcp-config`` JSON builder (proposal 061 Step 4).

The builder is the config-generation seam the ``cli_paced`` provider's tool
adapter uses to point a spawned ``claude`` at Maestro-hosted MCP servers. It is
pure JSON shaping — no transport, no ``mcp``/``maestro`` import — so these tests
pin the wire shape, the include/omit rules, copy semantics, and serialisation.
"""

from __future__ import annotations

import json

from maestro_sdk import (
    MCP_SERVERS_KEY,
    build_mcp_config,
    serialize_mcp_config,
    stdio_server,
)


def test_stdio_server_command_only_omits_args_and_env() -> None:
    # A bare command emits exactly {"command": ...} — no empty args/env padding.
    assert stdio_server("claude-mcp") == {"command": "claude-mcp"}


def test_stdio_server_includes_args_and_env_when_provided() -> None:
    spec = stdio_server(
        "python",
        args=["-m", "server", "journal"],
        env={"GITHUB_TOKEN": "tok"},
    )
    assert spec == {
        "command": "python",
        "args": ["-m", "server", "journal"],
        "env": {"GITHUB_TOKEN": "tok"},
    }


def test_stdio_server_empty_args_is_included_distinct_from_none() -> None:
    # An explicit empty sequence is honoured (caller asked for "no args"),
    # distinct from the None default which omits the key entirely.
    assert stdio_server("cmd", args=[]) == {"command": "cmd", "args": []}
    assert stdio_server("cmd", env={}) == {"command": "cmd", "env": {}}


def test_stdio_server_copies_args_and_env() -> None:
    # The returned spec must not alias the caller's mutable inputs.
    args = ["-m", "server"]
    env = {"K": "V"}
    spec = stdio_server("cmd", args=args, env=env)
    args.append("mutated")
    env["K"] = "mutated"
    assert spec["args"] == ["-m", "server"]
    assert spec["env"] == {"K": "V"}


def test_build_mcp_config_wraps_servers_under_top_level_key() -> None:
    config = build_mcp_config({"journal": {"command": "a"}, "memory": {"command": "b"}})
    assert config == {
        MCP_SERVERS_KEY: {"journal": {"command": "a"}, "memory": {"command": "b"}},
    }


def test_build_mcp_config_preserves_server_insertion_order() -> None:
    config = build_mcp_config(
        {"gamma": {"command": "g"}, "alpha": {"command": "a"}, "beta": {"command": "b"}},
    )
    assert list(config[MCP_SERVERS_KEY]) == ["gamma", "alpha", "beta"]


def test_build_mcp_config_empty_servers() -> None:
    assert build_mcp_config({}) == {MCP_SERVERS_KEY: {}}


def test_build_mcp_config_copies_each_spec() -> None:
    # Each spec is shallow-copied so the returned config does not alias inputs.
    spec = {"command": "cmd"}
    config = build_mcp_config({"srv": spec})
    spec["command"] = "mutated"
    assert config[MCP_SERVERS_KEY]["srv"] == {"command": "cmd"}


def test_serialize_mcp_config_round_trips_to_json() -> None:
    config = build_mcp_config({"journal": stdio_server("python", args=["-m", "s"])})
    text = serialize_mcp_config(config)
    assert json.loads(text) == config
    # Two-space indented and key-order-preserving for a human-diffable file.
    assert "\n  " in text
    assert text.index("mcpServers") < text.index("journal")
