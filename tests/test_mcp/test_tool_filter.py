"""Tests for MCP tool name matching and filtering logic.

Tests cover:
- Prefixed exact match (server__tool format)
- Server name match (include all tools from that server)
- Unprefixed tool name match (match original name across servers)
- MCP server config filtering for Copilot provider
- MCP tool def filtering for Claude provider
- Edge cases: empty filters, None filters, no matches
"""

from __future__ import annotations

from conductor.mcp.tool_filter import (
    filter_mcp_server_configs,
    filter_mcp_tool_defs,
    matches_tool_filter,
)

# ---------------------------------------------------------------------------
# matches_tool_filter
# ---------------------------------------------------------------------------


class TestMatchesToolFilter:
    """Tests for the matches_tool_filter helper."""

    def test_prefixed_exact_match(self) -> None:
        """server__tool format should match exactly."""
        assert matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["twig__set"],
            all_server_names={"twig", "github"},
        )

    def test_prefixed_no_match(self) -> None:
        """Prefixed name that doesn't match should return False."""
        assert not matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["github__search"],
            all_server_names={"twig", "github"},
        )

    def test_server_name_match(self) -> None:
        """Server name should match all tools from that server."""
        assert matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["twig"],
            all_server_names={"twig", "github"},
        )

    def test_server_name_no_match_other_server(self) -> None:
        """Server name should not match tools from other servers."""
        assert not matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["github"],
            all_server_names={"twig", "github"},
        )

    def test_unprefixed_tool_name_match(self) -> None:
        """Unprefixed name should match against original_tool_name."""
        assert matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["set"],
            all_server_names={"twig", "github"},
        )

    def test_unprefixed_tool_name_no_match(self) -> None:
        """Unprefixed name that doesn't match should return False."""
        assert not matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["query"],
            all_server_names={"twig", "github"},
        )

    def test_server_name_takes_precedence_over_tool_name(self) -> None:
        """When a name matches a server name, it should be treated as server match."""
        # "twig" is a server name, so it matches ALL tools from that server
        # even if a different server has a tool named "twig"
        assert matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=["twig"],
            all_server_names={"twig"},
        )

    def test_multiple_filter_entries(self) -> None:
        """Should match if any filter entry matches."""
        assert matches_tool_filter(
            tool_name="github__search",
            server_name="github",
            original_tool_name="search",
            filter_names=["twig__set", "search"],
            all_server_names={"twig", "github"},
        )

    def test_empty_filter_no_match(self) -> None:
        """Empty filter list should not match anything."""
        assert not matches_tool_filter(
            tool_name="twig__set",
            server_name="twig",
            original_tool_name="set",
            filter_names=[],
            all_server_names={"twig"},
        )


# ---------------------------------------------------------------------------
# filter_mcp_tool_defs (Claude provider path)
# ---------------------------------------------------------------------------

FAKE_MCP_TOOLS = [
    {
        "name": "filesystem__read_file",
        "description": "Read a file",
        "input_schema": {"type": "object"},
        "server": "filesystem",
        "original_name": "read_file",
    },
    {
        "name": "filesystem__write_file",
        "description": "Write a file",
        "input_schema": {"type": "object"},
        "server": "filesystem",
        "original_name": "write_file",
    },
    {
        "name": "web_search__search",
        "description": "Search the web",
        "input_schema": {"type": "object"},
        "server": "web_search",
        "original_name": "search",
    },
]


class TestFilterMcpToolDefs:
    """Tests for filter_mcp_tool_defs (used by Claude provider)."""

    def test_none_filter_includes_all(self) -> None:
        """None filter includes all tools."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, None)
        assert len(result) == 3

    def test_empty_filter_includes_all(self) -> None:
        """Empty list filter includes all tools (no filtering)."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, [])
        assert len(result) == 3

    def test_prefixed_filter(self) -> None:
        """Prefixed name filters to exact match."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, ["filesystem__read_file"])
        assert len(result) == 1
        assert result[0]["name"] == "filesystem__read_file"

    def test_server_name_filter(self) -> None:
        """Server name includes all tools from that server."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, ["filesystem"])
        assert len(result) == 2
        names = {t["name"] for t in result}
        assert names == {"filesystem__read_file", "filesystem__write_file"}

    def test_unprefixed_tool_name_filter(self) -> None:
        """Unprefixed tool name matches original name across servers."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, ["search"])
        assert len(result) == 1
        assert result[0]["name"] == "web_search__search"

    def test_mixed_filter(self) -> None:
        """Mix of prefixed, server, and unprefixed names."""
        result = filter_mcp_tool_defs(
            FAKE_MCP_TOOLS,
            ["filesystem__read_file", "search"],
        )
        assert len(result) == 2
        names = {t["name"] for t in result}
        assert names == {"filesystem__read_file", "web_search__search"}

    def test_nonexistent_filter(self) -> None:
        """Filter with nonexistent name returns nothing."""
        result = filter_mcp_tool_defs(FAKE_MCP_TOOLS, ["nonexistent"])
        assert len(result) == 0

    def test_empty_tool_list(self) -> None:
        """Empty tool list with filter returns empty."""
        result = filter_mcp_tool_defs([], ["something"])
        assert result == []


# ---------------------------------------------------------------------------
# filter_mcp_server_configs (Copilot provider path)
# ---------------------------------------------------------------------------

FAKE_SERVER_CONFIGS = {
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
    "context7": {
        "type": "stdio",
        "command": "context7-mcp",
        "args": [],
        "tools": ["search", "resolve"],
    },
}


class TestFilterMcpServerConfigs:
    """Tests for filter_mcp_server_configs (used by Copilot provider)."""

    def test_none_filter_returns_all(self) -> None:
        """None filter returns all servers."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, None)
        assert result is FAKE_SERVER_CONFIGS

    def test_empty_filter_returns_empty(self) -> None:
        """Empty list returns empty dict (no tools)."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, [])
        assert result == {}

    def test_server_name_includes_whole_server(self) -> None:
        """Server name includes entire server config."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["twig"])
        assert "twig" in result
        assert result["twig"] is FAKE_SERVER_CONFIGS["twig"]
        assert len(result) == 1

    def test_prefixed_name_filters_server_tools(self) -> None:
        """server__tool format includes server with filtered tools."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["twig__set"])
        assert "twig" in result
        assert result["twig"]["tools"] == ["set"]
        assert len(result) == 1

    def test_prefixed_name_wildcard_server(self) -> None:
        """server__tool works even when server has tools: [*]."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["github__search"])
        assert "github" in result
        assert result["github"]["tools"] == ["search"]

    def test_unprefixed_tool_matches_across_servers(self) -> None:
        """Unprefixed tool name matches tools across servers."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["search"])
        # "search" matches context7's "search" tool and github (wildcard)
        assert "context7" in result
        assert "search" in result["context7"]["tools"]
        # github has ["*"] so "search" matches there too
        assert "github" in result

    def test_multiple_tools_from_same_server(self) -> None:
        """Multiple prefixed tools from same server are aggregated."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["twig__set", "twig__query"])
        assert "twig" in result
        assert set(result["twig"]["tools"]) == {"set", "query"}

    def test_no_matches_returns_empty(self) -> None:
        """Filter with no matches returns empty dict."""
        # Use servers without wildcard to test true "no match" scenario
        servers_no_wildcard = {
            "twig": FAKE_SERVER_CONFIGS["twig"],
            "context7": FAKE_SERVER_CONFIGS["context7"],
        }
        result = filter_mcp_server_configs(servers_no_wildcard, ["nonexistent"])
        assert result == {}

    def test_mixed_server_and_tool_names(self) -> None:
        """Mix of server names and tool names."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["twig", "github__search"])
        assert "twig" in result
        assert "github" in result
        # twig gets full config, github gets filtered
        assert result["twig"]["tools"] == ["set", "query", "show"]
        assert result["github"]["tools"] == ["search"]

    def test_empty_servers_with_filter(self) -> None:
        """Empty server dict with filter returns empty."""
        result = filter_mcp_server_configs({}, ["something"])
        assert result == {}

    def test_prefixed_tool_not_in_server_tools_list(self) -> None:
        """Prefixed name where tool isn't in server's tools list is excluded."""
        result = filter_mcp_server_configs(FAKE_SERVER_CONFIGS, ["twig__nonexistent"])
        # twig has explicit tools list, "nonexistent" not in it
        assert result == {}
