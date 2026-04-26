"""Shared MCP utility functions for building server configurations.

This module extracts common MCP server configuration building logic
from the CLI layer so it can be reused by the workflow engine for
sub-workflow MCP server merging.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Pattern for ${VAR} and ${VAR:-default} syntax
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def resolve_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Resolve ${VAR} and ${VAR:-default} patterns in env values.

    Resolves at runtime from the current process environment.

    Args:
        env: Dictionary of environment variable names to values,
             where values may contain ${VAR} patterns.

    Returns:
        New dictionary with all ${VAR} patterns resolved.
    """

    def replace_match(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_value = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value
        else:
            return ""

    resolved: dict[str, str] = {}
    for key, value in env.items():
        resolved[key] = _ENV_VAR_PATTERN.sub(replace_match, value)
    return resolved


async def build_mcp_servers(config: Any) -> dict[str, Any] | None:
    """Build MCP server configurations from a WorkflowConfig.

    Converts MCPServerDef Pydantic models into the raw dict format
    expected by providers (Copilot SDK session kwargs, Claude MCPManager).

    For http/sse servers, attempts to resolve authentication headers
    via the mcp_auth module.

    Args:
        config: A WorkflowConfig instance with workflow.runtime.mcp_servers.

    Returns:
        MCP server configurations dict, or None if none configured.
    """
    if not config.workflow.runtime.mcp_servers:
        return None

    from conductor.mcp_auth import resolve_mcp_server_auth

    mcp_servers: dict[str, Any] = {}
    for name, server in config.workflow.runtime.mcp_servers.items():
        if server.type in ("http", "sse"):
            server_config: dict[str, Any] = {
                "type": server.type,
                "url": server.url,
                "tools": server.tools,
            }
            if server.headers:
                server_config["headers"] = server.headers
            if server.timeout:
                server_config["timeout"] = server.timeout
            server_config = await resolve_mcp_server_auth(name, server_config)
        else:
            server_config = {
                "type": "stdio",
                "command": server.command,
                "args": server.args,
                "tools": server.tools,
            }
            if server.env:
                server_config["env"] = resolve_env_vars(server.env)
            if server.timeout:
                server_config["timeout"] = server.timeout
        mcp_servers[name] = server_config

    logger.debug(f"MCP servers configured: {list(mcp_servers.keys())}")
    return mcp_servers


def merge_mcp_server_configs(
    parent_servers: dict[str, Any] | None,
    child_servers: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge parent and child MCP server configurations.

    Child servers are added to the parent's set. On name collision,
    the parent's definition wins (parent is authoritative).

    Args:
        parent_servers: Parent workflow's MCP server configs.
        child_servers: Child sub-workflow's MCP server configs.

    Returns:
        Merged MCP server configs, or None if both are empty.
    """
    if not child_servers:
        return parent_servers
    if not parent_servers:
        return child_servers

    # Start with child, then overlay parent (parent wins on collision)
    merged = dict(child_servers)
    merged.update(parent_servers)
    return merged
