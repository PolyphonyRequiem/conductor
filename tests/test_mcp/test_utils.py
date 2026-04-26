"""Tests for MCP server configuration utilities.

Tests cover:
- resolve_env_vars: environment variable resolution
- build_mcp_servers: config to dict conversion
- merge_mcp_server_configs: parent/child server merging
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conductor.mcp.utils import build_mcp_servers, merge_mcp_server_configs, resolve_env_vars


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_resolves_set_var(self) -> None:
        """Resolves a set environment variable."""
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            result = resolve_env_vars({"KEY": "${MY_VAR}"})
            assert result == {"KEY": "hello"}

    def test_resolves_default(self) -> None:
        """Uses default when variable is not set."""
        env = os.environ.copy()
        env.pop("UNSET_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            result = resolve_env_vars({"KEY": "${UNSET_VAR:-fallback}"})
            assert result == {"KEY": "fallback"}

    def test_empty_dict(self) -> None:
        """Empty input returns empty output."""
        assert resolve_env_vars({}) == {}

    def test_no_pattern_passthrough(self) -> None:
        """Values without patterns pass through unchanged."""
        result = resolve_env_vars({"KEY": "plain_value"})
        assert result == {"KEY": "plain_value"}


class TestMergeMcpServerConfigs:
    """Tests for merge_mcp_server_configs function."""

    def test_no_child_returns_parent(self) -> None:
        """No child servers returns parent as-is."""
        parent = {"twig": {"command": "twig"}}
        result = merge_mcp_server_configs(parent, None)
        assert result is parent

    def test_no_parent_returns_child(self) -> None:
        """No parent servers returns child as-is."""
        child = {"github": {"command": "github"}}
        result = merge_mcp_server_configs(None, child)
        assert result is child

    def test_both_none_returns_none(self) -> None:
        """Both None returns None."""
        result = merge_mcp_server_configs(None, None)
        assert result is None

    def test_disjoint_merge(self) -> None:
        """Non-overlapping servers are both included."""
        parent = {"twig": {"command": "twig"}}
        child = {"github": {"command": "github"}}
        result = merge_mcp_server_configs(parent, child)
        assert result is not None
        assert "twig" in result
        assert "github" in result

    def test_parent_wins_on_collision(self) -> None:
        """Parent definition wins when both declare same server name."""
        parent = {"twig": {"command": "parent-twig", "args": ["--parent"]}}
        child = {"twig": {"command": "child-twig", "args": ["--child"]}}
        result = merge_mcp_server_configs(parent, child)
        assert result is not None
        assert result["twig"]["command"] == "parent-twig"
        assert result["twig"]["args"] == ["--parent"]

    def test_collision_plus_unique(self) -> None:
        """Collision on one server, unique servers also included."""
        parent = {"twig": {"command": "parent-twig"}}
        child = {"twig": {"command": "child-twig"}, "github": {"command": "github"}}
        result = merge_mcp_server_configs(parent, child)
        assert result is not None
        assert result["twig"]["command"] == "parent-twig"
        assert result["github"]["command"] == "github"

    def test_empty_child_returns_parent(self) -> None:
        """Empty child dict returns parent."""
        parent = {"twig": {"command": "twig"}}
        result = merge_mcp_server_configs(parent, {})
        assert result is parent


class TestBuildMcpServers:
    """Tests for build_mcp_servers function."""

    @pytest.mark.asyncio
    async def test_no_servers_returns_none(self) -> None:
        """Config with no MCP servers returns None."""
        config = MagicMock()
        config.workflow.runtime.mcp_servers = {}
        result = await build_mcp_servers(config)
        assert result is None

    @pytest.mark.asyncio
    async def test_stdio_server(self) -> None:
        """Stdio server config is converted correctly."""
        server = MagicMock()
        server.type = "stdio"
        server.command = "my-mcp"
        server.args = ["--flag"]
        server.env = {}
        server.timeout = None
        server.tools = ["*"]

        config = MagicMock()
        config.workflow.runtime.mcp_servers = {"my-server": server}

        result = await build_mcp_servers(config)
        assert result is not None
        assert "my-server" in result
        assert result["my-server"]["type"] == "stdio"
        assert result["my-server"]["command"] == "my-mcp"
        assert result["my-server"]["args"] == ["--flag"]
        assert result["my-server"]["tools"] == ["*"]

    @pytest.mark.asyncio
    async def test_http_server(self) -> None:
        """HTTP server config is converted correctly."""
        server = MagicMock()
        server.type = "http"
        server.url = "https://example.com/mcp"
        server.headers = {"Authorization": "Bearer token"}
        server.timeout = 5000
        server.tools = ["search"]

        config = MagicMock()
        config.workflow.runtime.mcp_servers = {"remote": server}

        with patch(
            "conductor.mcp_auth.resolve_mcp_server_auth",
            new_callable=AsyncMock,
            side_effect=lambda name, cfg: cfg,
        ):
            result = await build_mcp_servers(config)

        assert result is not None
        assert "remote" in result
        assert result["remote"]["type"] == "http"
        assert result["remote"]["url"] == "https://example.com/mcp"
        assert result["remote"]["timeout"] == 5000

    @pytest.mark.asyncio
    async def test_env_vars_resolved(self) -> None:
        """Environment variables in server env are resolved."""
        server = MagicMock()
        server.type = "stdio"
        server.command = "my-mcp"
        server.args = []
        server.env = {"API_KEY": "${TEST_KEY:-default_key}"}
        server.timeout = None
        server.tools = ["*"]

        config = MagicMock()
        config.workflow.runtime.mcp_servers = {"my-server": server}

        result = await build_mcp_servers(config)
        assert result is not None
        assert result["my-server"]["env"]["API_KEY"] == "default_key"
