"""Tools for the ChatBotAgent to use via ReAct.

This module defines the tools available to the ReAct agent,
following the DSPy Tool pattern with clear docstrings and type hints.

Includes:
- Memory tools: search, store, list memories
- Utility tools: get_current_time, calculate
- MCP Orchestration tools: get_available_mcp_servers, delegate_to_subagent
"""

import asyncio
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.infrastructure.memory.mem0_service import Mem0MemoryService

logger = logging.getLogger(__name__)


def create_memory_search_tool(memory_service: "Mem0MemoryService", user_id: str):
    """Factory to create a memory search tool bound to a specific user.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to search memories for.

    Returns:
        A callable tool function for searching memories.
    """

    def search_memories(query: str, limit: int = 5) -> str:
        """Search for relevant memories based on a semantic query.

        Use this tool when you need to recall information about the user,
        their preferences, past conversations, or any stored facts.

        Args:
            query: The search query describing what memories to find.
            limit: Maximum number of memories to retrieve (default: 5).

        Returns:
            A formatted string of relevant memories, or a message if none found.
        """
        try:
            result = memory_service.search(query=query, user_id=user_id, limit=limit)
            if not result.memories:
                return "No relevant memories found."

            memories = []
            for idx, mem in enumerate(result.memories, 1):
                category_str = f" [{mem.category}]" if mem.category else ""
                memories.append(f"{idx}.{category_str} {mem.content}")

            return "Relevant memories:\n" + "\n".join(memories)
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    return search_memories


def create_memory_store_tool(memory_service: "Mem0MemoryService", user_id: str):
    """Factory to create a memory store tool bound to a specific user.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to store memories for.

    Returns:
        A callable tool function for storing memories.
    """

    def store_memory(content: str, category: str | None = None) -> str:
        """Store a new memory about the user for future recall.

        Use this tool when you learn something important about the user
        that should be remembered, such as preferences, facts, or decisions.

        Args:
            content: The information to store as a memory.
            category: Optional category (e.g., 'preferences', 'context', 'goals').

        Returns:
            Confirmation message about the stored memory.
        """
        try:
            metadata = {}
            if category:
                metadata["category"] = category

            logger.info(f"Storing memory for user {user_id}: {content[:50]}...")

            memory_service.add(
                content=content,
                user_id=user_id,
                metadata=metadata if metadata else None,
            )

            logger.info(f"Memory stored successfully for user {user_id}")
            return f"Memory stored successfully: '{content}'"
        except Exception as e:
            logger.error(f"Failed to store memory for user {user_id}: {e}")
            return f"Error storing memory: {str(e)}"

    return store_memory


def create_memory_list_tool(memory_service: "Mem0MemoryService", user_id: str):
    """Factory to create a memory list tool bound to a specific user.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to list memories for.

    Returns:
        A callable tool function for listing all memories.
    """

    def list_all_memories(limit: int = 10) -> str:
        """Retrieve all stored memories for the current user.

        Use this tool when you need a complete overview of what you know
        about the user, not for specific queries.

        Args:
            limit: Maximum number of memories to retrieve (default: 10).

        Returns:
            A formatted list of all memories.
        """
        try:
            result = memory_service.get_all(user_id=user_id, limit=limit)
            if not result.memories:
                return "No memories stored for this user."

            memories = []
            for idx, mem in enumerate(result.memories, 1):
                category_str = f" [{mem.category}]" if mem.category else ""
                memories.append(f"{idx}.{category_str} {mem.content}")

            return "All stored memories:\n" + "\n".join(memories)
        except Exception as e:
            return f"Error listing memories: {str(e)}"

    return list_all_memories


def create_memory_search_by_category_tool(
    memory_service: "Mem0MemoryService", user_id: str
):
    """Factory to create a tool for searching memories within a specific category.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to search memories for.

    Returns:
        A callable tool function for category-filtered search.
    """

    def search_memories_by_category(
        query: str, category: str, limit: int = 5
    ) -> str:
        """Search for memories within a specific category.

        Use this tool when you need to find information in a specific
        memory category like 'preferences', 'context', or 'goals'.

        Args:
            query: The search query describing what memories to find.
            category: Category to filter by (e.g., 'preferences', 'context').
            limit: Maximum number of memories to retrieve (default: 5).

        Returns:
            A formatted string of relevant memories in the category.
        """
        try:
            result = memory_service.search_by_category(
                query=query, user_id=user_id, category=category, limit=limit
            )
            if not result.memories:
                return f"No memories found in category '{category}'."

            memories = []
            for idx, mem in enumerate(result.memories, 1):
                memories.append(f"{idx}. {mem.content}")

            return f"Memories in '{category}':\n" + "\n".join(memories)
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    return search_memories_by_category


def get_current_time() -> str:
    """Get the current date and time.

    Use this tool when the user asks about the current time, date,
    or when you need to provide time-sensitive information.

    Returns:
        The current date and time in a human-readable format.
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


def calculate(expression: str) -> str:
    """Perform a mathematical calculation.

    Use this tool when you need to perform arithmetic calculations
    that the user requests.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2', '15 * 7').

    Returns:
        The result of the calculation or an error message.
    """
    # Define safe operations
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters. Only numbers and +-*/() allowed."

    try:
        # Use eval with restricted builtins for basic math
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# Async tool versions for better performance


async def async_search_memories(
    memory_service: "Mem0MemoryService", user_id: str, query: str, limit: int = 5
) -> str:
    """Async version of memory search.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to search memories for.
        query: The search query.
        limit: Maximum number of results.

    Returns:
        Formatted string of relevant memories.
    """
    # Run synchronous mem0 operation in executor to avoid blocking
    loop = asyncio.get_event_loop()
    search_func = create_memory_search_tool(memory_service, user_id)
    return await loop.run_in_executor(None, search_func, query, limit)


async def async_store_memory(
    memory_service: "Mem0MemoryService",
    user_id: str,
    content: str,
    category: str | None = None,
) -> str:
    """Async version of memory storage.

    Args:
        memory_service: The MemoryService instance.
        user_id: The user ID to store memory for.
        content: The memory content to store.
        category: Optional category for the memory.

    Returns:
        Confirmation message.
    """
    loop = asyncio.get_event_loop()
    store_func = create_memory_store_tool(memory_service, user_id)
    return await loop.run_in_executor(None, store_func, content, category)


# =============================================================================
# MCP Orchestration Tools
# =============================================================================
# These tools enable the hierarchical agent architecture where the main
# ChatBotAgent delegates tasks to specialized MCPSubAgents.


def get_available_mcp_servers() -> str:
    """Get a list of available MCP servers and their capabilities.

    Use this tool to discover what specialized capabilities are available
    before delegating tasks. Each server handles a specific domain.

    Returns:
        A JSON-formatted list of available servers with their descriptions
        and capabilities.
    """
    from app.infrastructure.mcp.registry import get_servers_summary

    return get_servers_summary()


def create_delegate_to_subagent_tool(user_id: str) -> Callable:
    """Factory to create a delegate_to_subagent tool bound to a specific user.

    The delegate tool spawns an MCPSubAgent to handle specialized tasks
    using a specific MCP server's tools.

    Args:
        user_id: The user ID for the MCP operations.

    Returns:
        An async callable tool function for delegating tasks.
    """

    async def delegate_to_subagent(server_name: str, task: str, context: str = "") -> str:
        """Delegate a task to a specialized sub-agent for execution.

        Use this tool when you need to perform operations that require
        a specific MCP server's capabilities (e.g., expense tracking,
        note management).

        IMPORTANT: Before using this tool, call get_available_mcp_servers()
        to see what servers are available and their capabilities.

        Args:
            server_name: Name of the MCP server to use (e.g., 'expense_manager',
                        'note_manager').
            task: Clear, specific description of what needs to be done.
                  Be detailed - the sub-agent only sees this task description.
            context: Optional additional context from the conversation that
                    may help the sub-agent complete the task.

        Returns:
            The result from the sub-agent, or an error message if delegation failed.

        Examples:
            - delegate_to_subagent("expense_manager", "Create an expense for $25 lunch at Subway today")
            - delegate_to_subagent("note_manager", "Create a note titled 'Meeting Notes' with content about the Q4 planning discussion")
        """
        from app.infrastructure.llm.subagent import MCPSubAgent
        from app.infrastructure.mcp.registry import get_server_metadata

        # Validate server name
        metadata = get_server_metadata(server_name)
        if metadata is None:
            from app.infrastructure.mcp.registry import get_all_server_names

            available = ", ".join(get_all_server_names())
            return f"Error: Unknown server '{server_name}'. Available servers: {available}"

        # Create and execute sub-agent
        try:
            subagent = MCPSubAgent(
                mcp_config=metadata.config,
                user_id=user_id,
                max_iters=4,
            )

            result = await subagent.execute(task=task, context=context)

            if result.success:
                logger.info(
                    f"Delegation to {server_name} succeeded in {result.duration_ms:.0f}ms"
                )
                return result.result or "Task completed successfully."
            else:
                logger.warning(
                    f"Delegation to {server_name} failed: {result.error}"
                )
                return f"Error from {server_name}: {result.error}"

        except Exception as e:
            logger.error(f"Delegation to {server_name} raised exception: {e}")
            return f"Error delegating to {server_name}: {str(e)}"

    return delegate_to_subagent
