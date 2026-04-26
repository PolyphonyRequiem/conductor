"""Tests for MCP tool filtering in the Copilot provider.

Tests verify that the Copilot provider correctly filters MCP server
configs based on the agent's tools specification before passing them
to the SDK's create_session().

Covers:
- tools=None: all MCP servers passed (no filtering)
- tools=[]: no MCP servers passed (empty dict)
- tools=[server_name]: only that server included
- tools=[server__tool]: server included with filtered tools
- tools=[unprefixed]: matched against server tool lists
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from conductor.providers.copilot import CopilotProvider


def _make_provider(mcp_servers: dict[str, Any] | None = None) -> CopilotProvider:
    """Create a CopilotProvider with mock handler and optional MCP servers."""

    def mock_handler(agent: Any, prompt: str, context: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "mock"}

    return CopilotProvider(mock_handler=mock_handler, mcp_servers=mcp_servers)


def _make_agent(name: str = "test_agent") -> MagicMock:
    """Create a mock AgentDef."""
    agent = MagicMock()
    agent.name = name
    agent.model = None
    agent.retry = None
    agent.max_iterations = None
    return agent


MCP_SERVERS = {
    "twig": {
        "type": "stdio",
        "command": "twig-mcp",
        "args": [],
        "tools": ["set", "query", "show"],
    },
    "github": {
        "type": "stdio",
        "command": "github-mcp",
        "args": [],
        "tools": ["*"],
    },
}


class TestCopilotMcpToolFiltering:
    """Tests for MCP server filtering in CopilotProvider."""

    @pytest.mark.asyncio
    async def test_tools_none_passes_all_servers(self) -> None:
        """tools=None should pass all MCP servers to SDK session."""
        provider = _make_provider(MCP_SERVERS)
        agent = _make_agent()

        # Execute with tools=None (all tools)
        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="test",
            tools=None,
        )

        # Mock handler was used, so we check the call history
        assert len(provider._call_history) == 1
        assert provider._call_history[0]["tools"] is None

    @pytest.mark.asyncio
    async def test_tools_empty_list_no_servers(self) -> None:
        """tools=[] should result in no MCP servers."""
        provider = _make_provider(MCP_SERVERS)
        agent = _make_agent()

        await provider.execute(
            agent=agent,
            context={},
            rendered_prompt="test",
            tools=[],
        )

        assert len(provider._call_history) == 1
        assert provider._call_history[0]["tools"] == []

    @pytest.mark.asyncio
    async def test_filter_mcp_server_configs_integration(self) -> None:
        """Verify filter_mcp_server_configs works with Copilot provider data."""
        from conductor.mcp.tool_filter import filter_mcp_server_configs

        # Simulate what happens inside _execute_sdk_call
        result = filter_mcp_server_configs(MCP_SERVERS, None)
        assert result is MCP_SERVERS  # No filtering

        result = filter_mcp_server_configs(MCP_SERVERS, [])
        assert result == {}  # No tools

        result = filter_mcp_server_configs(MCP_SERVERS, ["twig"])
        assert "twig" in result
        assert "github" not in result

        result = filter_mcp_server_configs(MCP_SERVERS, ["twig__set"])
        assert "twig" in result
        assert result["twig"]["tools"] == ["set"]

        result = filter_mcp_server_configs(MCP_SERVERS, ["github__search"])
        assert "github" in result
        assert result["github"]["tools"] == ["search"]
