"""ChatBotAgent implementation using DSPy ReAct pattern.

This module provides the main ChatBotAgent class that integrates:
- DSPy ReAct for reasoning and tool usage
- MemoryService for conversation memory and context
- MCP (Model Context Protocol) servers for extensible tool capabilities
- Async-first design for scalability

Architecture follows SOLID principles:
- Single Responsibility: Agent handles conversation, tools handle specific operations
- Open/Closed: New tools can be added without modifying agent core (via MCP)
- Liskov Substitution: Agent implements standard dspy.Module interface
- Interface Segregation: Small, focused signatures per operation
- Dependency Inversion: Agent depends on abstractions (MemoryService, tools)

MCP Integration:
- MCP tools are converted to DSPy tools using dspy.Tool.from_mcp_tool()
- User ID is passed to MCP servers via environment variables (never as tool params)
- This prevents prompt injection attacks where AI could manipulate user context
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dspy

from app.infrastructure.llm.signatures import ChatBotSignature
from app.infrastructure.llm.tools import (
    calculate,
    create_memory_list_tool,
    create_memory_search_by_category_tool,
    create_memory_search_tool,
    create_memory_store_tool,
    get_current_time,
)
from app.infrastructure.mcp.client import (
    EXPENSE_MANAGER_CONFIG,
    MCPServerConfig,
    create_mcp_tools_for_user,
    get_mcp_session,
)

if TYPE_CHECKING:
    from app.infrastructure.memory.mem0_service import Mem0MemoryService

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Encapsulates conversation context for the agent."""

    user_id: str
    session_id: str | None = None
    conversation_history: str = ""
    memory_context: str = ""


@dataclass
class AgentResponse:
    """The response from the ChatBotAgent.

    Memory storage now happens via tool calls during ReAct execution,
    not via output fields.
    """

    response: str
    reasoning_trace: list[str] | None = None


class ChatBotAgent(dspy.Module):
    """A ReAct-based chatbot agent with memory and MCP integration.

    This agent uses DSPy's ReAct module to reason about user queries,
    decide when to use tools, and generate contextually relevant responses.

    MCP Integration:
        The agent can connect to MCP servers to extend its capabilities.
        MCP tools are dynamically loaded and converted to DSPy tools.
        User ID is passed securely via environment variables, never as
        tool parameters, preventing prompt injection attacks.

    Attributes:
        memory_service: The MemoryService for context retrieval and storage.
        max_iters: Maximum iterations for the ReAct reasoning loop.
        mcp_configs: List of MCP server configurations to connect to.
        enable_mcp: Whether MCP tools are enabled (default: True).
    """

    def __init__(
        self,
        memory_service: "Mem0MemoryService",
        max_iters: int = 6,
        include_reasoning_trace: bool = False,
        mcp_configs: list[MCPServerConfig] | None = None,
        enable_mcp: bool = True,
    ):
        """Initialize the ChatBotAgent.

        Args:
            memory_service: MemoryService for memory operations.
            max_iters: Maximum ReAct iterations (default: 6).
            include_reasoning_trace: Whether to include the reasoning trace in output.
            mcp_configs: List of MCP server configs (default: expense_manager).
            enable_mcp: Whether to enable MCP tools (default: True).
        """
        super().__init__()
        self.memory_service = memory_service
        self.max_iters = max_iters
        self.include_reasoning_trace = include_reasoning_trace
        self.enable_mcp = enable_mcp

        # Default MCP servers if none provided
        self.mcp_configs = mcp_configs if mcp_configs is not None else [
            EXPENSE_MANAGER_CONFIG,
        ]

        # Core tools available to all conversations
        self._base_tools = [
            get_current_time,
            calculate,
        ]

        # The ReAct module will be initialized per-request with user-specific tools
        # This is stored as a template signature
        self._signature = ChatBotSignature

    def _create_tools_for_user(self, user_id: str) -> list:
        """Create a tool set bound to a specific user (sync version).

        This creates only the synchronous tools (memory + base).
        For MCP tools, use _create_tools_for_user_async().

        Args:
            user_id: The user ID for memory operations.

        Returns:
            List of tools including base tools and user-specific memory tools.
        """
        return [
            *self._base_tools,
            create_memory_search_tool(self.memory_service, user_id),
            create_memory_store_tool(self.memory_service, user_id),
            create_memory_list_tool(self.memory_service, user_id),
            create_memory_search_by_category_tool(self.memory_service, user_id),
        ]

    @asynccontextmanager
    async def _mcp_tools_context(
        self, user_id: str
    ) -> AsyncIterator[list[dspy.Tool]]:
        """Context manager that provides MCP tools with active sessions.

        MCP tools require the session to remain open while they're being used.
        This context manager keeps all MCP sessions alive during tool usage.

        Args:
            user_id: The user ID for MCP operations.

        Yields:
            List of DSPy tools from all configured MCP servers.
        """
        mcp_tools: list[dspy.Tool] = []
        sessions = []
        exit_stacks = []

        if not self.enable_mcp or not self.mcp_configs:
            yield mcp_tools
            return

        try:
            # Open sessions for all MCP servers
            for config in self.mcp_configs:
                try:
                    # Create session context
                    session_cm = get_mcp_session(config, user_id)
                    session = await session_cm.__aenter__()
                    sessions.append(session)
                    exit_stacks.append(session_cm)

                    # Get tools from this session
                    tools = await create_mcp_tools_for_user(config, user_id, session)
                    mcp_tools.extend(tools)
                    logger.info(f"Added {len(tools)} tools from {config.name}")

                except Exception as e:
                    logger.warning(f"Failed to load MCP tools from {config.name}: {e}")
                    # Continue without this MCP server's tools

            yield mcp_tools

        finally:
            # Close all sessions
            for exit_stack in exit_stacks:
                try:
                    await exit_stack.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Error closing MCP session: {e}")

    def _format_conversation_history(
        self, history: list[dict[str, str]], max_messages: int = 10
    ) -> str:
        """Format conversation history for the signature.

        Args:
            history: List of message dicts with 'role' and 'content' keys.
            max_messages: Maximum number of recent messages to include.

        Returns:
            Formatted string representation of the conversation.
        """
        if not history:
            return "No previous conversation history."

        recent = history[-max_messages:]
        formatted = []
        for msg in recent:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _retrieve_memory_context(self, user_id: str, query: str) -> str:
        """Retrieve relevant memory context organized by category.

        Pulls from all configured categories to provide comprehensive
        context for the LLM, including user preferences, current context,
        and goals.

        Args:
            user_id: The user ID for memory lookup.
            query: The current user query for semantic search.

        Returns:
            Formatted string of relevant memories organized by category.
        """
        try:
            context_parts = []

            # Get configured categories from the memory service
            categories = self.memory_service.config.categories

            # Retrieve memories from each category
            for category in categories:
                result = self.memory_service.search_by_category(
                    query=query,
                    user_id=user_id,
                    category=category.name,
                    limit=2,
                )
                if result.memories:
                    # Format with category header
                    memories_text = "\n".join(
                        f"  - {mem.content}" for mem in result.memories
                    )
                    context_parts.append(f"[{category.name}]\n{memories_text}")

            if not context_parts:
                return "No relevant memories found for this user."

            return "User context from memory:\n" + "\n\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return f"Memory retrieval unavailable: {str(e)}"

    async def _retrieve_memory_context_async(self, user_id: str, query: str) -> str:
        """Async version of memory context retrieval.

        Args:
            user_id: The user ID for memory lookup.
            query: The current user query for semantic search.

        Returns:
            Formatted string of relevant memories.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._retrieve_memory_context, user_id, query
        )

    def forward(
        self,
        user_message: str,
        user_id: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        image: dspy.Image | None = None,
    ) -> AgentResponse:
        """Process a user message and generate a response synchronously.

        Args:
            user_message: The user's current message.
            user_id: Unique identifier for the user.
            conversation_history: Optional list of previous messages.
            session_id: Optional session identifier for tracking.
            image: Optional image for multimodal input.

        Returns:
            AgentResponse with the generated response and metadata.
        """
        # Format conversation history
        history_str = self._format_conversation_history(conversation_history or [])

        logger.debug("=" * 60)
        logger.debug("FORMATTED CONVERSATION HISTORY FOR LLM")
        logger.debug("=" * 60)
        logger.debug("\n%s", history_str)
        logger.debug("=" * 60)

        # Retrieve relevant memories
        memory_context = self._retrieve_memory_context(user_id, user_message)

        logger.debug("=" * 60)
        logger.debug("MEM0 CONTEXT FOR LLM (user_id=%s)", user_id)
        logger.debug("=" * 60)
        logger.debug("\n%s", memory_context)
        logger.debug("=" * 60)

        # Create user-specific tools
        tools = self._create_tools_for_user(user_id)

        # Create ReAct module with user-specific tools
        react = dspy.ReAct(
            signature=self._signature,
            tools=tools,
            max_iters=self.max_iters,
        )

        # Execute the ReAct loop
        # Memory storage happens via store_memory tool calls during execution
        try:
            # Build kwargs, only include image if provided
            call_kwargs = {
                "user_message": user_message,
                "conversation_history": history_str,
                "memory_context": memory_context,
                "user_id": user_id,
            }
            if image is not None:
                call_kwargs["image"] = image

            result = react(**call_kwargs)

            # Extract response - memory storage already happened via tool calls
            response = getattr(result, "response", str(result))

            return AgentResponse(
                response=response,
                reasoning_trace=self._extract_trace(result)
                if self.include_reasoning_trace
                else None,
            )

        except Exception as e:
            # Log the error for debugging
            logger.error(f"ReAct execution failed: {e}")
            error_msg = (
                "I apologize, but I encountered an error processing your request. "
                "Please try again."
            )
            return AgentResponse(response=error_msg)

    async def aforward(
        self,
        user_message: str,
        user_id: str,
        conversation_history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        image: dspy.Image | None = None,
    ) -> AgentResponse:
        """Process a user message and generate a response asynchronously.

        This is the preferred method for production use, enabling
        non-blocking operations and better scalability.

        MCP Integration:
            When enable_mcp=True, this method connects to configured MCP servers
            and adds their tools to the ReAct agent. The user_id is passed to
            MCP servers via environment variables (never as tool parameters)
            to prevent prompt injection attacks.

            IMPORTANT: MCP sessions are kept open during the entire ReAct
            execution because MCP tools hold references to their sessions.

        Args:
            user_message: The user's current message.
            user_id: Unique identifier for the user.
            conversation_history: Optional list of previous messages.
            session_id: Optional session identifier for tracking.
            image: Optional image for multimodal input.

        Returns:
            AgentResponse with the generated response and metadata.
        """
        # Format conversation history (CPU-bound, fast)
        history_str = self._format_conversation_history(conversation_history or [])

        # Retrieve relevant memories asynchronously
        memory_context = await self._retrieve_memory_context_async(
            user_id, user_message
        )

        # Create base tools (memory + utility)
        tools = self._create_tools_for_user(user_id)

        # Use context manager to keep MCP sessions open during ReAct execution
        async with self._mcp_tools_context(user_id) as mcp_tools:
            # Combine base tools with MCP tools
            all_tools = tools + mcp_tools

            logger.debug(f"Agent has {len(all_tools)} tools available for user {user_id}")
            logger.debug(f"Tools: {[getattr(t, 'name', str(t)) for t in all_tools]}")

            # Create ReAct module with all tools
            react = dspy.ReAct(
                signature=self._signature,
                tools=all_tools,
                max_iters=self.max_iters,
            )

            # Execute the ReAct loop asynchronously
            # Memory storage happens via store_memory tool calls during execution
            # Expense operations happen via MCP expense_manager tools
            try:
                # Build kwargs, only include image if provided
                call_kwargs = {
                    "user_message": user_message,
                    "conversation_history": history_str,
                    "memory_context": memory_context,
                    "user_id": user_id,
                }
                if image is not None:
                    call_kwargs["image"] = image

                result = await react.acall(**call_kwargs)

                # Extract response - memory storage already happened via tool calls
                response = getattr(result, "response", str(result))

                return AgentResponse(
                    response=response,
                    reasoning_trace=self._extract_trace(result)
                    if self.include_reasoning_trace
                    else None,
                )

            except Exception as e:
                # Log the error for debugging
                logger.error(f"ReAct async execution failed: {e}", exc_info=True)
                error_msg = (
                    "I apologize, but I encountered an error processing your request. "
                    "Please try again."
                )
                return AgentResponse(response=error_msg)

    def _extract_trace(self, result) -> list[str] | None:
        """Extract reasoning trace from ReAct result.

        Args:
            result: The dspy.Prediction result from ReAct.

        Returns:
            List of reasoning steps or None if not available.
        """
        try:
            if hasattr(result, "trajectory"):
                return [str(step) for step in result.trajectory]
            return None
        except Exception:
            return None


class SimpleChatBotAgent(dspy.Module):
    """A simpler chatbot agent without ReAct for basic use cases.

    This agent uses ChainOfThought for simpler reasoning without
    tool usage. Suitable for straightforward Q&A scenarios.

    NOTE: This agent cannot store memories as it doesn't use tools.
    Use ChatBotAgent for full memory support.
    """

    def __init__(self, memory_service: "Mem0MemoryService"):
        """Initialize the simple agent.

        Args:
            memory_service: MemoryService for context retrieval.
        """
        super().__init__()
        self.memory_service = memory_service
        self.cot = dspy.ChainOfThought(ChatBotSignature)

    def forward(
        self,
        user_message: str,
        user_id: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Process message with chain-of-thought reasoning.

        Args:
            user_message: The user's message.
            user_id: User identifier.
            conversation_history: Previous messages.

        Returns:
            AgentResponse with the response.
        """
        # Format history
        history_str = ""
        if conversation_history:
            history_str = "\n".join(
                f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}"
                for m in conversation_history[-10:]
            )
        else:
            history_str = "No previous conversation."

        # Get memory context
        try:
            result = self.memory_service.search(
                query=user_message, user_id=user_id, limit=3
            )
            if result.memories:
                memory_str = self.memory_service.format_context(
                    result.memories, include_category=True
                )
            else:
                memory_str = "No relevant memories."
        except Exception as e:
            logger.warning(f"Memory search failed: {e}")
            memory_str = "Memory unavailable."

        # Generate response
        result = self.cot(
            user_message=user_message,
            conversation_history=history_str,
            memory_context=memory_str,
            user_id=user_id,
        )

        return AgentResponse(response=getattr(result, "response", str(result)))

    async def aforward(
        self,
        user_message: str,
        user_id: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Async version of forward.

        Args:
            user_message: The user's message.
            user_id: User identifier.
            conversation_history: Previous messages.

        Returns:
            AgentResponse with the response.
        """
        # Format history
        history_str = ""
        if conversation_history:
            history_str = "\n".join(
                f"{m.get('role', 'unknown').capitalize()}: {m.get('content', '')}"
                for m in conversation_history[-10:]
            )
        else:
            history_str = "No previous conversation."

        # Get memory context async
        loop = asyncio.get_event_loop()

        def get_memory():
            try:
                result = self.memory_service.search(
                    query=user_message, user_id=user_id, limit=3
                )
                if result.memories:
                    return self.memory_service.format_context(
                        result.memories, include_category=True
                    )
                return "No relevant memories."
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")
                return "Memory unavailable."

        memory_str = await loop.run_in_executor(None, get_memory)

        # Generate response async
        result = await self.cot.acall(
            user_message=user_message,
            conversation_history=history_str,
            memory_context=memory_str,
            user_id=user_id,
        )

        return AgentResponse(response=getattr(result, "response", str(result)))
