"""MCP Server Registry for hierarchical agent orchestration.

This module provides metadata about available MCP servers, enabling the
ChatBotAgent to discover and delegate tasks to specialized sub-agents
without loading all tools upfront.

The registry follows the Open/Closed principle - new servers can be
added without modifying the orchestration logic.
"""

from dataclasses import dataclass, field

from app.infrastructure.mcp.client import (
    EXPENSE_MANAGER_CONFIG,
    NOTE_MANAGER_CONFIG,
    MCPServerConfig,
)


@dataclass
class MCPServerMetadata:
    """Metadata for an MCP server used by the orchestrator.

    This provides descriptive information for the LLM to decide
    which server to delegate tasks to, without loading actual tools.

    Attributes:
        name: Unique identifier for the server (matches server_dir)
        config: The MCPServerConfig for spawning the server
        description: Human-readable description of the server's purpose
        capabilities: List of high-level capabilities (for LLM context)
        use_when: Guidance on when to use this server
    """

    name: str
    config: MCPServerConfig
    description: str
    capabilities: list[str] = field(default_factory=list)
    use_when: str = ""


# Registry of all available MCP servers with their metadata
MCP_SERVER_REGISTRY: dict[str, MCPServerMetadata] = {
    "expense_manager": MCPServerMetadata(
        name="expense_manager",
        config=EXPENSE_MANAGER_CONFIG,
        description="Manage personal expenses, track spending, view summaries and analyze trends",
        capabilities=[
            "create_expense",
            "list_expenses",
            "search_expenses",
            "update_expense",
            "delete_expense",
            "get_expense_summary",
            "get_monthly_trend",
            "get_top_categories",
            "list_categories",
        ],
        use_when="User wants to track expenses, log spending, view expense summaries, analyze spending patterns, or manage expense categories",
    ),
    "note_manager": MCPServerMetadata(
        name="note_manager",
        config=NOTE_MANAGER_CONFIG,
        description="Create and manage notes with AI-powered summarization and semantic search",
        capabilities=[
            "create_note",
            "get_note",
            "list_notes",
            "update_note",
            "delete_note",
            "semantic_search",
        ],
        use_when="User wants to save notes, find information from notes, create reminders, or manage personal knowledge",
    ),
}


def get_server_metadata(server_name: str) -> MCPServerMetadata | None:
    """Get metadata for a specific MCP server.

    Args:
        server_name: The name of the server to look up

    Returns:
        MCPServerMetadata if found, None otherwise
    """
    return MCP_SERVER_REGISTRY.get(server_name)


def get_all_server_names() -> list[str]:
    """Get list of all registered server names.

    Returns:
        List of server name strings
    """
    return list(MCP_SERVER_REGISTRY.keys())


def get_servers_summary() -> str:
    """Get a formatted summary of all available servers for LLM context.

    Returns:
        JSON-formatted string describing available servers
    """
    import json

    servers = []
    for name, metadata in MCP_SERVER_REGISTRY.items():
        servers.append({
            "name": name,
            "description": metadata.description,
#             "capabilities": metadata.capabilities,
            "use_when": metadata.use_when,
        })
    return json.dumps(servers, indent=2)
