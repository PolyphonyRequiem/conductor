"""Shared tool name matching logic for MCP tool filtering.

Used by both the Copilot and Claude providers to filter MCP tools
based on agent-level tool specifications from workflow YAML.

Supports three matching modes:
1. Prefixed exact match: ``server__tool`` matches a specific tool from a specific server
2. Server name match: ``server_name`` (no ``__``) includes all tools from that server
3. Unprefixed tool name: matches against original tool names across all servers
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def matches_tool_filter(
    tool_name: str,
    server_name: str,
    original_tool_name: str,
    filter_names: list[str],
    all_server_names: set[str] | None = None,
) -> bool:
    """Check if a tool matches any entry in the filter list.

    Args:
        tool_name: The prefixed tool name (e.g., ``twig__set``).
        server_name: The MCP server name (e.g., ``twig``).
        original_tool_name: The unprefixed tool name (e.g., ``set``).
        filter_names: List of filter entries from agent's ``tools:`` config.
        all_server_names: Set of all known MCP server names, used to
            disambiguate unprefixed names from server names.

    Returns:
        True if the tool matches any filter entry.
    """
    known_servers = all_server_names or set()

    for name in filter_names:
        # Mode 1: Exact prefixed match (e.g., "twig__set")
        if "__" in name:
            if name == tool_name:
                return True
            continue

        # Mode 2: Server name match — include all tools from that server
        if name in known_servers:
            if name == server_name:
                return True
            continue

        # Mode 3: Unprefixed tool name — match against original name
        if name == original_tool_name:
            return True

    return False


def filter_mcp_tool_defs(
    tools: list[dict[str, Any]],
    tool_filter: list[str] | None,
    server_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter a list of MCP tool definitions by agent tool filter.

    Each tool dict must have ``name``, ``server``, and ``original_name`` keys
    (as produced by :class:`conductor.mcp.manager.MCPManager`).

    Args:
        tools: List of MCP tool definition dicts.
        tool_filter: Agent's tool filter list. ``None`` means all tools,
            empty list ``[]`` means all tools (no filtering).
        server_names: Set of all known server names for disambiguation.

    Returns:
        Filtered list of tool dicts.
    """
    # None or empty list = no filtering, include all
    if not tool_filter:
        return list(tools)

    known_servers = server_names or {t.get("server", "") for t in tools}

    return [
        tool
        for tool in tools
        if matches_tool_filter(
            tool_name=tool["name"],
            server_name=tool.get("server", ""),
            original_tool_name=tool.get("original_name", ""),
            filter_names=tool_filter,
            all_server_names=known_servers,
        )
    ]


def filter_mcp_server_configs(
    mcp_servers: dict[str, Any],
    tool_filter: list[str] | None,
) -> dict[str, Any]:
    """Filter MCP server configs based on agent tool filter.

    Used by the Copilot provider where MCP servers are passed as raw
    config dicts to the SDK (not individual tool defs).

    Matching logic:
    - ``tool_filter=None``: return all servers (no filtering)
    - ``tool_filter=[]``: return empty dict (no tools)
    - ``tool_filter=[list]``: match entries against server names and
      ``server__tool`` prefixed names

    Args:
        mcp_servers: Raw MCP server configuration dict.
        tool_filter: Agent's tool filter list.

    Returns:
        Filtered MCP server config dict.
    """
    if tool_filter is None:
        return mcp_servers

    if not tool_filter:
        return {}

    all_server_names = set(mcp_servers.keys())
    filtered: dict[str, Any] = {}

    for server_name, server_config in mcp_servers.items():
        server_tools = server_config.get("tools", ["*"])

        # Check if entire server is requested by name
        matched_as_server = False
        for name in tool_filter:
            if "__" not in name and name in all_server_names and name == server_name:
                filtered[server_name] = server_config
                matched_as_server = True
                break

        if matched_as_server:
            continue

        # Check for specific tool matches within this server
        matching_tools: list[str] = []
        for name in tool_filter:
            if "__" in name:
                # Prefixed format: server__tool
                prefix, suffix = name.split("__", 1)
                if prefix == server_name and (server_tools == ["*"] or suffix in server_tools):
                    matching_tools.append(suffix)
            else:
                # Unprefixed tool name: match against server's tool list
                if name not in all_server_names and (server_tools == ["*"] or name in server_tools):
                    matching_tools.append(name)

        if matching_tools:
            filtered_config = dict(server_config)
            filtered_config["tools"] = matching_tools
            filtered[server_name] = filtered_config

    return filtered
