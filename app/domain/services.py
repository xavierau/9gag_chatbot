"""Domain service protocols/interfaces.

This module defines the abstract interfaces for services used by the domain.
Implementations are provided in the infrastructure layer, following
the Dependency Inversion Principle.
"""

from datetime import datetime, timedelta
from typing import Any, Protocol

from app.domain.memory import Memory, MemoryConfig, MemoryFilter, MemorySearchResult
from app.domain.session import ConversationMessage, Session, SessionHistoryResult


class MemoryService(Protocol):
    """Protocol for memory storage and retrieval services.

    This abstraction allows swapping between different memory backends
    (mem0, custom implementations, mocks for testing) without changing
    the domain or application layer code.
    """

    @property
    def config(self) -> MemoryConfig:
        """Get the current memory configuration."""
        ...

    def add(
        self,
        content: str | list[dict[str, str]],
        user_id: str,
        *,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Memory]:
        """Add a memory for a user.

        Args:
            content: The content to store. Can be a string or list of messages.
            user_id: The user identifier.
            agent_id: Optional agent identifier for agent-specific memories.
            run_id: Optional run/session identifier for isolation.
            metadata: Optional metadata to attach.

        Returns:
            List of created Memory objects.
        """
        ...

    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int | None = None,
        filters: MemoryFilter | None = None,
    ) -> MemorySearchResult:
        """Search for relevant memories.

        Args:
            query: The search query for semantic matching.
            user_id: The user identifier.
            limit: Maximum number of results (uses config default if None).
            filters: Optional filter criteria.

        Returns:
            MemorySearchResult with matching memories.
        """
        ...

    def get_all(
        self,
        user_id: str,
        *,
        limit: int | None = None,
        filters: MemoryFilter | None = None,
    ) -> MemorySearchResult:
        """Get all memories for a user.

        Args:
            user_id: The user identifier.
            limit: Maximum number of results.
            filters: Optional filter criteria.

        Returns:
            MemorySearchResult with all matching memories.
        """
        ...

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        ...

    def update(self, memory_id: str, content: str) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: The memory identifier.
            content: The new content.

        Returns:
            The updated Memory if found, None otherwise.
        """
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def delete_all(self, user_id: str) -> int:
        """Delete all memories for a user.

        Args:
            user_id: The user identifier.

        Returns:
            Number of memories deleted.
        """
        ...

    def format_context(
        self,
        memories: list[Memory],
        *,
        include_category: bool = False,
        max_items: int | None = None,
    ) -> str:
        """Format memories as context string for LLM prompts.

        Args:
            memories: List of memories to format.
            include_category: Whether to include category in output.
            max_items: Maximum items to include.

        Returns:
            Formatted string suitable for LLM context.
        """
        ...

    def add_with_expiration(
        self,
        content: str | list[dict[str, str]],
        user_id: str,
        expires_in: timedelta,
        **kwargs: Any,
    ) -> list[Memory]:
        """Add a memory with automatic expiration.

        Convenience method that calculates expiration date from timedelta.

        Args:
            content: The content to store.
            user_id: The user identifier.
            expires_in: Duration until expiration.
            **kwargs: Additional arguments passed to add().

        Returns:
            List of created Memory objects.
        """
        ...


class SessionMemoryService(Protocol):
    """Protocol for session-based conversation memory service.

    This abstraction handles turn-by-turn conversation history storage
    and retrieval, separate from long-term semantic memory (mem0).
    Sessions persist conversations as an audit trail and provide
    context for the DSPy agent.
    """

    async def create_session(
        self,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new conversation session.

        Args:
            user_id: The user identifier.
            metadata: Optional metadata (e.g., client info, tags).

        Returns:
            The created Session.
        """
        ...

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            The Session if found, None otherwise.
        """
        ...

    async def get_or_create_session(
        self,
        session_id: str | None,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Get an existing session or create a new one.

        If session_id is provided and exists, returns that session.
        Otherwise, creates a new session for the user.

        Args:
            session_id: Optional session identifier to look up.
            user_id: The user identifier (used for new sessions).
            metadata: Optional metadata for new sessions.

        Returns:
            The existing or newly created Session.
        """
        ...

    async def list_user_sessions(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Session]:
        """List all sessions for a user.

        Args:
            user_id: The user identifier.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip (for pagination).

        Returns:
            List of Sessions, ordered by created_at descending.
        """
        ...

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationMessage:
        """Add a message to a session.

        Args:
            session_id: The session identifier.
            role: The message role ("user" or "assistant").
            content: The message content.
            metadata: Optional metadata (e.g., token count).

        Returns:
            The created ConversationMessage.

        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        ...

    async def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> SessionHistoryResult:
        """Get conversation history for a session.

        Args:
            session_id: The session identifier.
            limit: Maximum messages to return (uses config default if None).

        Returns:
            SessionHistoryResult with messages in chronological order.
        """
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...
