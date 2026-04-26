"""Tests for ProviderRegistry.merge_mcp_servers method.

Tests cover:
- No child MCP servers returns self (no new registry)
- Child MCP servers creates new registry
- Parent wins on collision
- Child registry uses child config
- Resume session IDs are propagated
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from conductor.providers.registry import ProviderRegistry


def _make_config(
    provider: str = "copilot",
    default_model: str | None = None,
    mcp_servers: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock WorkflowConfig."""
    config = MagicMock()
    config.workflow.runtime.provider = provider
    config.workflow.runtime.default_model = default_model
    config.workflow.runtime.temperature = None
    config.workflow.runtime.max_tokens = None
    config.workflow.runtime.timeout = None
    config.workflow.runtime.max_session_seconds = None
    config.workflow.runtime.max_agent_iterations = None
    config.workflow.runtime.mcp_servers = mcp_servers or {}
    return config


class TestRegistryMergeMcpServers:
    """Tests for ProviderRegistry.merge_mcp_servers."""

    def test_no_child_servers_returns_self(self) -> None:
        """When child has no MCP servers, returns the same registry."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})

        child_config = _make_config()
        result = registry.merge_mcp_servers(child_config, None)

        assert result is registry

    def test_empty_child_servers_returns_self(self) -> None:
        """Empty child servers dict returns same registry."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})

        child_config = _make_config()
        result = registry.merge_mcp_servers(child_config, {})

        assert result is registry

    def test_child_servers_creates_new_registry(self) -> None:
        """Child with new MCP servers creates a new registry."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})

        child_config = _make_config()
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result is not registry
        assert isinstance(result, ProviderRegistry)

    def test_merged_servers_include_both(self) -> None:
        """Merged registry includes both parent and child servers."""
        parent_config = _make_config()
        parent_servers = {"twig": {"cmd": "twig"}}
        registry = ProviderRegistry(parent_config, mcp_servers=parent_servers)

        child_config = _make_config()
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result._mcp_servers is not None
        assert "twig" in result._mcp_servers
        assert "github" in result._mcp_servers

    def test_parent_wins_collision(self) -> None:
        """Parent definition wins on server name collision."""
        parent_config = _make_config()
        parent_servers = {"twig": {"cmd": "parent-twig"}}
        registry = ProviderRegistry(parent_config, mcp_servers=parent_servers)

        child_config = _make_config()
        child_servers = {"twig": {"cmd": "child-twig"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result._mcp_servers is not None
        assert result._mcp_servers["twig"]["cmd"] == "parent-twig"

    def test_child_config_used_for_new_registry(self) -> None:
        """New registry is created from child config, not parent."""
        parent_config = _make_config(provider="copilot", default_model="gpt-4o")
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})

        child_config = _make_config(provider="claude", default_model="claude-3-5-sonnet")
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result._config is child_config
        assert result.default_provider_type == "claude"

    def test_resume_session_ids_propagated(self) -> None:
        """Resume session IDs are propagated to child registry."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})
        registry.set_resume_session_ids({"agent1": "session-123"})

        child_config = _make_config()
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result._resume_session_ids == {"agent1": "session-123"}

    @pytest.mark.asyncio
    async def test_child_registry_closeable(self) -> None:
        """Child registry can be closed independently."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers={"twig": {"cmd": "twig"}})

        child_config = _make_config()
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        # Should not raise
        await result.close()

    def test_no_parent_servers_child_only(self) -> None:
        """Parent with no MCP servers, child adds some."""
        parent_config = _make_config()
        registry = ProviderRegistry(parent_config, mcp_servers=None)

        child_config = _make_config()
        child_servers = {"github": {"cmd": "github"}}
        result = registry.merge_mcp_servers(child_config, child_servers)

        assert result is not registry
        assert result._mcp_servers is not None
        assert "github" in result._mcp_servers
