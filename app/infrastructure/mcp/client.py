"""MCP client for connecting to MCP servers and converting tools to DSPy format.

This module provides utilities for:
1. Connecting to MCP servers via stdio transport
2. Converting MCP tools to DSPy-compatible tools
3. Managing MCP server lifecycles

SECURITY: The user_id is passed to MCP servers via environment variables,
never as tool parameters. This prevents prompt injection attacks where
an AI could manipulate user_id to access other users' data.

Usage:
    ```python
    async with get_mcp_session(expense_config, user_id="user123") as session:
        tools = await session.list_tools()
        dspy_tools = [dspy.Tool.from_mcp_tool(session, t) for t in tools.tools]
    ```
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Base path for MCP servers relative to project root
MCP_SERVERS_BASE_PATH = Path(__file__).parent.parent.parent.parent / "mcp_servers"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        name: Human-readable name for the server
        server_dir: Directory name under mcp_servers/
        server_script: Path to the server script relative to server_dir
        env_vars: Additional environment variables to pass (user_id is handled separately)
        user_id_env_var: Name of env var for user_id (default: MCP_USER_ID)
    """

    name: str
    server_dir: str
    server_script: str = "src/{server_dir}/server.py"
    env_vars: dict[str, str] = field(default_factory=dict)
    user_id_env_var: str = "MCP_USER_ID"

    def __post_init__(self) -> None:
        """Format server_script with server_dir if needed."""
        if "{server_dir}" in self.server_script:
            self.server_script = self.server_script.format(server_dir=self.server_dir)

    def get_server_path(self) -> Path:
        """Get the full path to the server directory."""
        return MCP_SERVERS_BASE_PATH / self.server_dir

    def get_script_path(self) -> Path:
        """Get the full path to the server script."""
        return self.get_server_path() / self.server_script


# Pre-configured MCP servers
EXPENSE_MANAGER_CONFIG = MCPServerConfig(
    name="Expense Manager",
    server_dir="expense_manager",
    server_script="src/expense_manager/server.py",
)

NOTE_MANAGER_CONFIG = MCPServerConfig(
    name="Note Manager",
    server_dir="note_manager",
    server_script="src/note_manager/server.py",
)


@asynccontextmanager
async def get_mcp_session(
    config: MCPServerConfig,
    user_id: str,
) -> AsyncIterator[ClientSession]:
    """Create an MCP client session connected to a server.

    SECURITY: The user_id is passed via environment variable to the server,
    not as a parameter to tools. This ensures the AI cannot manipulate
    the user context.

    Args:
        config: MCP server configuration
        user_id: User ID to pass to the server via environment variable

    Yields:
        ClientSession connected to the MCP server

    Raises:
        FileNotFoundError: If server directory or script doesn't exist
        RuntimeError: If connection to server fails
    """
    # Validate paths exist
    server_path = config.get_server_path()
    script_path = config.get_script_path()

    if not server_path.exists():
        raise FileNotFoundError(f"MCP server directory not found: {server_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"MCP server script not found: {script_path}")

    # Build environment with user_id securely set
    env = {
        **os.environ,
        config.user_id_env_var: user_id,
        **config.env_vars,
    }

    # Configure server parameters for stdio transport
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--directory",
            str(server_path),
            "fastmcp",
            "run",
            str(script_path),
        ],
        env=env,
    )

    logger.debug(f"Connecting to MCP server: {config.name} for user: {user_id}")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.debug(f"Connected to MCP server: {config.name}")
                yield session
    except asyncio.CancelledError:
        # Re-raise cancellation without wrapping
        raise
    except Exception as e:
        logger.error(f"Failed to connect to MCP server {config.name}: {e}")
        raise RuntimeError(f"MCP server connection failed: {config.name}") from e


async def create_mcp_tools_for_user(
    config: MCPServerConfig,
    user_id: str,
    session: ClientSession,
) -> list[dspy.Tool]:
    """Convert MCP server tools to DSPy-compatible tools.

    This function fetches all available tools from an MCP server and
    converts them to DSPy Tool objects that can be used with ReAct agents.

    IMPORTANT: The tools returned are async and should be used with
    ReAct's acall() method.

    Args:
        config: MCP server configuration (for logging)
        user_id: User ID (for logging only - already set in session)
        session: Active MCP client session

    Returns:
        List of DSPy Tool objects ready for use with ReAct

    Example:
        ```python
        async with get_mcp_session(EXPENSE_MANAGER_CONFIG, user_id) as session:
            tools = await create_mcp_tools_for_user(
                EXPENSE_MANAGER_CONFIG, user_id, session
            )
            react = dspy.ReAct(signature, tools=tools)
            result = await react.acall(...)
        ```
    """
    tools_response = await session.list_tools()

    dspy_tools = []
    for tool in tools_response.tools:
        dspy_tool = dspy.Tool.from_mcp_tool(session, tool)
        dspy_tools.append(dspy_tool)
        logger.debug(f"Converted MCP tool: {tool.name}")

    logger.info(
        f"Created {len(dspy_tools)} DSPy tools from {config.name} for user {user_id}"
    )

    return dspy_tools


async def get_available_mcp_tools(
    configs: list[MCPServerConfig],
    user_id: str,
) -> list[dspy.Tool]:
    """Get all available tools from multiple MCP servers.

    This is a convenience function that connects to multiple MCP servers
    and aggregates their tools into a single list.

    NOTE: This creates temporary connections to each server. For production
    use with long-running agents, prefer managing sessions explicitly.

    Args:
        configs: List of MCP server configurations
        user_id: User ID to pass to all servers

    Returns:
        Combined list of DSPy tools from all servers
    """
    all_tools: list[dspy.Tool] = []

    for config in configs:
        try:
            async with get_mcp_session(config, user_id) as session:
                tools = await create_mcp_tools_for_user(config, user_id, session)
                all_tools.extend(tools)
        except Exception as e:
            logger.warning(f"Failed to get tools from {config.name}: {e}")
            # Continue with other servers

    return all_tools
