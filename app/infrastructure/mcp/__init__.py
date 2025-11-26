"""MCP (Model Context Protocol) client utilities.

This module provides utilities for connecting to MCP servers and
converting MCP tools to DSPy-compatible tools for use with ReAct agents.
"""

from app.infrastructure.mcp.client import (
    MCPServerConfig,
    create_mcp_tools_for_user,
    get_mcp_session,
)

__all__ = [
    "MCPServerConfig",
    "create_mcp_tools_for_user",
    "get_mcp_session",
]
