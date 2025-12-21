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

from app.infrastructure.llm.dspy_config import get_tracer
from app.infrastructure.llm.signatures import ChatBotSignature
from app.infrastructure.llm.tools import (
    calculate,
    create_delegate_to_subagent_tool,
    create_memory_list_tool,
    create_memory_search_by_category_tool,
    create_memory_search_tool,
    create_memory_store_tool,
    get_available_mcp_servers,
    get_current_time,
)
from app.infrastructure.mcp.client import (
    EXPENSE_MANAGER_CONFIG,
    NOTE_MANAGER_CONFIG,
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
        The agent supports two modes for MCP integration:

        1. Hierarchical Mode (use_hierarchical_mcp=True, default):
           - Agent has only 8 tools (base + memory + meta-tools)
           - Uses get_available_mcp_servers() to discover capabilities
           - Uses delegate_to_subagent() to spawn specialized MCPSubAgents
           - Reduces token usage and improves tool selection accuracy

        2. Flat Mode (use_hierarchical_mcp=False):
           - Agent loads all MCP tools directly (30+ tools)
           - Legacy behavior, kept for comparison/fallback

        User ID is passed securely via environment variables, never as
        tool parameters, preventing prompt injection attacks.

    Attributes:
        memory_service: The MemoryService for context retrieval and storage.
        max_iters: Maximum iterations for the ReAct reasoning loop.
        mcp_configs: List of MCP server configurations (used in flat mode).
        enable_mcp: Whether MCP tools are enabled (default: True).
        use_hierarchical_mcp: Use hierarchical delegation (default: True).
    """

    def __init__(
        self,
        memory_service: "Mem0MemoryService",
        max_iters: int = 6,
        include_reasoning_trace: bool = False,
        mcp_configs: list[MCPServerConfig] | None = None,
        enable_mcp: bool = True,
        use_hierarchical_mcp: bool = True,
    ):
        """Initialize the ChatBotAgent.

        Args:
            memory_service: MemoryService for memory operations.
            max_iters: Maximum ReAct iterations (default: 6).
            include_reasoning_trace: Whether to include the reasoning trace in output.
            mcp_configs: List of MCP server configs (used in flat mode).
            enable_mcp: Whether to enable MCP tools (default: True).
            use_hierarchical_mcp: Use hierarchical delegation instead of loading
                all MCP tools directly (default: True). When True, the agent uses
                meta-tools (get_available_mcp_servers, delegate_to_subagent) to
                spawn specialized sub-agents on demand.
        """
        super().__init__()
        self.memory_service = memory_service
        self.max_iters = max_iters
        self.include_reasoning_trace = include_reasoning_trace
        self.enable_mcp = enable_mcp
        self.use_hierarchical_mcp = use_hierarchical_mcp

        # Default MCP servers if none provided (used in flat mode)
        self.mcp_configs = mcp_configs if mcp_configs is not None else [
            EXPENSE_MANAGER_CONFIG,
            NOTE_MANAGER_CONFIG,
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
        """Create a tool set bound to a specific user.

        In hierarchical mode, this includes meta-tools for MCP orchestration.
        In flat mode, this creates only base + memory tools (MCP tools added separately).

        Args:
            user_id: The user ID for memory operations.

        Returns:
            List of tools including base tools, memory tools, and optionally meta-tools.
        """
        tools = [
            *self._base_tools,
            create_memory_search_tool(self.memory_service, user_id),
            create_memory_store_tool(self.memory_service, user_id),
            create_memory_list_tool(self.memory_service, user_id),
            create_memory_search_by_category_tool(self.memory_service, user_id),
        ]

        # In hierarchical mode, add meta-tools for MCP orchestration
        if self.use_hierarchical_mcp and self.enable_mcp:
            tools.append(get_available_mcp_servers)
            tools.append(create_delegate_to_subagent_tool(user_id))

        return tools

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
        active_context_managers: list = []

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
                    # Only track context managers that successfully entered
                    active_context_managers.append(session_cm)

                    # Get tools from this session
                    tools = await create_mcp_tools_for_user(config, user_id, session)
                    mcp_tools.extend(tools)
                    logger.info(f"Added {len(tools)} tools from {config.name}")

                except Exception as e:
                    logger.warning(f"Failed to load MCP tools from {config.name}: {e}")
                    # Continue without this MCP server's tools

            yield mcp_tools

        finally:
            # Close all successfully opened sessions (in reverse order)
            for cm in reversed(active_context_managers):
                try:
                    await cm.__aexit__(None, None, None)
                except asyncio.CancelledError:
                    # Ignore cancellation errors during cleanup
                    pass
                except Exception as e:
                    logger.debug(f"Error closing MCP session: {e}")

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

    def _retrieve_memory_by_category(
        self, user_id: str, query: str, category: str, limit: int = 3
    ) -> str:
        """Retrieve memories from a specific category.

        Args:
            user_id: The user ID for memory lookup.
            query: The current user query for semantic search.
            category: The category name to retrieve from.
            limit: Maximum number of memories to retrieve.

        Returns:
            Formatted string of relevant memories from the category.
        """
        try:
            result = self.memory_service.search_by_category(
                query=query,
                user_id=user_id,
                category=category,
                limit=limit,
            )
            if result.memories:
                memories_text = "\n".join(
                    f"- {mem.content}" for mem in result.memories
                )
                return memories_text

            return f"No {category} information available."

        except Exception as e:
            logger.warning(f"Memory retrieval failed for {category}: {e}")
            return f"Memory retrieval unavailable for {category}."

    def _retrieve_memory_contexts(
        self, user_id: str, query: str
    ) -> dict[str, str]:
        """Retrieve relevant memory contexts separated by category type.

        Retrieves memories from each category type (preferences, context,
        goals, procedural) and returns them as separate strings for
        distinct LLM input fields.

        Args:
            user_id: The user ID for memory lookup.
            query: The current user query for semantic search.

        Returns:
            Dict mapping category names to formatted memory strings.
        """
        return {
            "preferences": self._retrieve_memory_by_category(
                user_id, query, "preferences", limit=3
            ),
            "context": self._retrieve_memory_by_category(
                user_id, query, "context", limit=3
            ),
            "goals": self._retrieve_memory_by_category(
                user_id, query, "goals", limit=2
            ),
            "procedural": self._retrieve_memory_by_category(
                user_id, query, "procedural", limit=3
            ),
        }

    async def _retrieve_memory_contexts_async(
        self, user_id: str, query: str
    ) -> dict[str, str]:
        """Async version of memory contexts retrieval.

        Args:
            user_id: The user ID for memory lookup.
            query: The current user query for semantic search.

        Returns:
            Dict mapping category names to formatted memory strings.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._retrieve_memory_contexts, user_id, query
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

        # Retrieve relevant memories separated by category
        memory_contexts = self._retrieve_memory_contexts(user_id, user_message)

        logger.debug("=" * 60)
        logger.debug("MEM0 CONTEXTS FOR LLM (user_id=%s)", user_id)
        logger.debug("=" * 60)
        for category, context in memory_contexts.items():
            logger.debug("[%s]\n%s\n", category.upper(), context)
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
                "user_preferences": memory_contexts["preferences"],
                "user_context": memory_contexts["context"],
                "user_goals": memory_contexts["goals"],
                "procedural_knowledge": memory_contexts["procedural"],
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
            The agent supports two modes:

            1. Hierarchical Mode (use_hierarchical_mcp=True, default):
               - Agent uses meta-tools (get_available_mcp_servers, delegate_to_subagent)
               - MCP connections are established on-demand by sub-agents
               - Only 8 tools in context (better accuracy, lower token usage)

            2. Flat Mode (use_hierarchical_mcp=False):
               - All MCP tools loaded directly (30+ tools)
               - MCP sessions kept open during entire ReAct execution
               - Legacy behavior for comparison

            User ID is passed securely via environment variables, never as
            tool parameters, preventing prompt injection attacks.

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

        # Retrieve relevant memories asynchronously separated by category
        memory_contexts = await self._retrieve_memory_contexts_async(
            user_id, user_message
        )

        # Create tools for user (includes meta-tools in hierarchical mode)
        tools = self._create_tools_for_user(user_id)

        # Choose execution path based on mode
        if self.use_hierarchical_mcp:
            # Hierarchical mode: use meta-tools, no direct MCP loading
            return await self._execute_hierarchical(
                tools=tools,
                user_message=user_message,
                history_str=history_str,
                memory_contexts=memory_contexts,
                user_id=user_id,
                image=image,
            )
        else:
            # Flat mode: load all MCP tools directly (legacy behavior)
            return await self._execute_flat(
                tools=tools,
                user_message=user_message,
                history_str=history_str,
                memory_contexts=memory_contexts,
                user_id=user_id,
                image=image,
            )

    async def _execute_hierarchical(
        self,
        tools: list,
        user_message: str,
        history_str: str,
        memory_contexts: dict[str, str],
        user_id: str,
        image: dspy.Image | None = None,
    ) -> AgentResponse:
        """Execute ReAct with hierarchical MCP orchestration.

        In this mode, the agent has meta-tools for discovering and delegating
        to MCP sub-agents. No MCP tools are loaded directly.

        Args:
            tools: Pre-built tool list (base + memory + meta-tools)
            user_message: The user's message
            history_str: Formatted conversation history
            memory_contexts: Retrieved memory contexts by category
            user_id: User identifier
            image: Optional image input

        Returns:
            AgentResponse with the generated response
        """
        logger.debug(
            f"Hierarchical mode: {len(tools)} tools for user {user_id}"
        )
        logger.debug(f"Tools: {[getattr(t, '__name__', str(t)) for t in tools]}")

        # Create ReAct module with meta-tools
        react = dspy.ReAct(
            signature=self._signature,
            tools=tools,
            max_iters=self.max_iters,
        )

        # Get tracer for custom spans
        tracer = get_tracer()

        try:
            call_kwargs = {
                "user_message": user_message,
                "conversation_history": history_str,
                "user_preferences": memory_contexts["preferences"],
                "user_context": memory_contexts["context"],
                "user_goals": memory_contexts["goals"],
                "procedural_knowledge": memory_contexts["procedural"],
                "user_id": user_id,
            }
            if image is not None:
                call_kwargs["image"] = image

            # Execute with tracing span if available
            if tracer:
                with tracer.start_as_current_span(
                    "ChatBotAgent.execute",
                    attributes={
                        "agent.type": "main",
                        "agent.mode": "hierarchical",
                        "agent.user_id": user_id,
                        "agent.tools_count": len(tools),
                        "agent.max_iters": self.max_iters,
                        "input.message_length": len(user_message),
                        "input.has_image": image is not None,
                    },
                ) as span:
                    result = await react.acall(**call_kwargs)
                    response = getattr(result, "response", str(result))

                    # Add output attributes
                    span.set_attribute("output.response_length", len(response))
                    if self.include_reasoning_trace and hasattr(result, "trajectory"):
                        span.set_attribute(
                            "output.iterations", len(result.trajectory)
                        )
            else:
                result = await react.acall(**call_kwargs)
                response = getattr(result, "response", str(result))

            return AgentResponse(
                response=response,
                reasoning_trace=self._extract_trace(result)
                if self.include_reasoning_trace
                else None,
            )

        except Exception as e:
            logger.error(f"Hierarchical ReAct execution failed: {e}", exc_info=True)
            return AgentResponse(
                response="I apologize, but I encountered an error processing your request. Please try again."
            )

    async def _execute_flat(
        self,
        tools: list,
        user_message: str,
        history_str: str,
        memory_contexts: dict[str, str],
        user_id: str,
        image: dspy.Image | None = None,
    ) -> AgentResponse:
        """Execute ReAct with all MCP tools loaded directly (flat mode).

        Legacy behavior where all MCP tools are loaded upfront.
        MCP sessions are kept open during the entire ReAct execution.

        Args:
            tools: Pre-built tool list (base + memory, no meta-tools)
            user_message: The user's message
            history_str: Formatted conversation history
            memory_contexts: Retrieved memory contexts by category
            user_id: User identifier
            image: Optional image input

        Returns:
            AgentResponse with the generated response
        """
        # Use context manager to keep MCP sessions open during ReAct execution
        async with self._mcp_tools_context(user_id) as mcp_tools:
            # Combine base tools with MCP tools
            all_tools = tools + mcp_tools

            logger.debug(f"Flat mode: {len(all_tools)} tools for user {user_id}")
            logger.debug(f"Tools: {[getattr(t, 'name', str(t)) for t in all_tools]}")

            # Create ReAct module with all tools
            react = dspy.ReAct(
                signature=self._signature,
                tools=all_tools,
                max_iters=self.max_iters,
            )

            # Get tracer for custom spans
            tracer = get_tracer()

            try:
                call_kwargs = {
                    "user_message": user_message,
                    "conversation_history": history_str,
                    "user_preferences": memory_contexts["preferences"],
                    "user_context": memory_contexts["context"],
                    "user_goals": memory_contexts["goals"],
                    "procedural_knowledge": memory_contexts["procedural"],
                    "user_id": user_id,
                }
                if image is not None:
                    call_kwargs["image"] = image

                # Execute with tracing span if available
                if tracer:
                    with tracer.start_as_current_span(
                        "ChatBotAgent.execute",
                        attributes={
                            "agent.type": "main",
                            "agent.mode": "flat",
                            "agent.user_id": user_id,
                            "agent.tools_count": len(all_tools),
                            "agent.mcp_tools_count": len(mcp_tools),
                            "agent.max_iters": self.max_iters,
                            "input.message_length": len(user_message),
                            "input.has_image": image is not None,
                        },
                    ) as span:
                        result = await react.acall(**call_kwargs)
                        response = getattr(result, "response", str(result))

                        # Add output attributes
                        span.set_attribute("output.response_length", len(response))
                        if self.include_reasoning_trace and hasattr(result, "trajectory"):
                            span.set_attribute(
                                "output.iterations", len(result.trajectory)
                            )
                else:
                    result = await react.acall(**call_kwargs)
                    response = getattr(result, "response", str(result))

                return AgentResponse(
                    response=response,
                    reasoning_trace=self._extract_trace(result)
                    if self.include_reasoning_trace
                    else None,
                )

            except Exception as e:
                logger.error(f"Flat ReAct execution failed: {e}", exc_info=True)
                return AgentResponse(
                    response="I apologize, but I encountered an error processing your request. Please try again."
                )

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

        # Get memory contexts by category
        def get_category_memories(category: str, limit: int = 3) -> str:
            try:
                result = self.memory_service.search_by_category(
                    query=user_message, user_id=user_id, category=category, limit=limit
                )
                if result.memories:
                    return "\n".join(f"- {m.content}" for m in result.memories)
                return f"No {category} information available."
            except Exception as e:
                logger.warning(f"Memory search failed for {category}: {e}")
                return f"Memory unavailable for {category}."

        preferences = get_category_memories("preferences", 3)
        context = get_category_memories("context", 3)
        goals = get_category_memories("goals", 2)
        procedural = get_category_memories("procedural", 3)

        # Generate response
        result = self.cot(
            user_message=user_message,
            conversation_history=history_str,
            user_preferences=preferences,
            user_context=context,
            user_goals=goals,
            procedural_knowledge=procedural,
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

        # Get memory contexts async
        loop = asyncio.get_event_loop()

        def get_category_memories(category: str, limit: int = 3) -> str:
            try:
                result = self.memory_service.search_by_category(
                    query=user_message, user_id=user_id, category=category, limit=limit
                )
                if result.memories:
                    return "\n".join(f"- {m.content}" for m in result.memories)
                return f"No {category} information available."
            except Exception as e:
                logger.warning(f"Memory search failed for {category}: {e}")
                return f"Memory unavailable for {category}."

        preferences = await loop.run_in_executor(
            None, get_category_memories, "preferences", 3
        )
        context = await loop.run_in_executor(None, get_category_memories, "context", 3)
        goals = await loop.run_in_executor(None, get_category_memories, "goals", 2)
        procedural = await loop.run_in_executor(
            None, get_category_memories, "procedural", 3
        )

        # Generate response async
        result = await self.cot.acall(
            user_message=user_message,
            conversation_history=history_str,
            user_preferences=preferences,
            user_context=context,
            user_goals=goals,
            procedural_knowledge=procedural,
            user_id=user_id,
        )

        return AgentResponse(response=getattr(result, "response", str(result)))
