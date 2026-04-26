"""MCP (Model Context Protocol) integration for Conductor.

This module provides MCP server management for providers that need to
spawn and communicate with MCP servers using the stdio transport.
"""

from conductor.mcp.manager import MCPManager
from conductor.mcp.tool_filter import filter_mcp_server_configs, filter_mcp_tool_defs
from conductor.mcp.utils import build_mcp_servers, merge_mcp_server_configs

__all__ = [
    "MCPManager",
    "build_mcp_servers",
    "filter_mcp_server_configs",
    "filter_mcp_tool_defs",
    "merge_mcp_server_configs",
]
