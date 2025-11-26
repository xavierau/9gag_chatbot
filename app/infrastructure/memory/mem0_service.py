"""Mem0 implementation of the MemoryService protocol.

This module provides the concrete implementation of MemoryService
using mem0 as the backend. It handles all mem0-specific details
while exposing a clean domain interface.
"""

import contextlib
import logging
from datetime import datetime, timedelta
from typing import Any

from mem0 import Memory as Mem0Memory

from app.domain.memory import (
    Memory,
    MemoryConfig,
    MemoryFilter,
    MemorySearchResult,
    create_default_memory_config,
)

logger = logging.getLogger(__name__)


class Mem0MemoryService:
    """Memory service implementation using mem0.

    This class wraps the raw mem0 client and provides a clean interface
    that follows the MemoryService protocol. It handles:
    - Configuration with custom categories and instructions
    - Memory CRUD operations
    - Search with filtering
    - Context formatting for LLM prompts
    """

    def __init__(
        self,
        mem0_client: Mem0Memory,
        config: MemoryConfig | None = None,
    ):
        """Initialize the Mem0MemoryService.

        Args:
            mem0_client: The underlying mem0 Memory client.
            config: Optional memory configuration. Uses default if not provided.
        """
        self._client = mem0_client
        self._config = config or create_default_memory_config()

    @property
    def config(self) -> MemoryConfig:
        """Get the current memory configuration."""
        return self._config

    @property
    def client(self) -> Mem0Memory:
        """Get the underlying mem0 client for direct access if needed."""
        return self._client

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
        try:
            kwargs: dict[str, Any] = {"user_id": user_id}

            # Use agent_id from config if not explicitly provided
            effective_agent_id = agent_id or self._config.agent_id
            if effective_agent_id:
                kwargs["agent_id"] = effective_agent_id

            if run_id:
                kwargs["run_id"] = run_id

            # Merge config metadata with provided metadata
            # Categories can be stored as metadata since custom_categories is not
            # supported in mem0ai v1.0.1 add() method
            effective_metadata = metadata.copy() if metadata else {}
            if self._config.categories:
                # Store category info in metadata for reference
                effective_metadata["configured_categories"] = [
                    cat.name for cat in self._config.categories
                ]
            if effective_metadata:
                kwargs["metadata"] = effective_metadata

            # Note: expiration_date and custom_instructions are not supported
            # in mem0ai v1.0.1 add() method - these features may be available
            # in the cloud API or future versions

            # Handle content format
            kwargs["messages"] = content

            result = self._client.add(**kwargs)

            return self._parse_add_result(result, user_id)

        except Exception as e:
            logger.error(f"Failed to add memory for user {user_id}: {e}")
            return []

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
        try:
            effective_limit = limit or self._config.default_limit

            kwargs: dict[str, Any] = {
                "query": query,
                "user_id": user_id,
                "limit": effective_limit,
            }

            if filters:
                mem0_filter = filters.to_mem0_filter()
                if mem0_filter:
                    kwargs["filters"] = mem0_filter

            result = self._client.search(**kwargs)
            return self._parse_search_result(result)

        except Exception as e:
            logger.error(f"Failed to search memories for user {user_id}: {e}")
            return MemorySearchResult(memories=[], total=0)

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
        try:
            kwargs: dict[str, Any] = {"user_id": user_id}

            if limit:
                kwargs["limit"] = limit

            if filters:
                mem0_filter = filters.to_mem0_filter()
                if mem0_filter:
                    kwargs["filters"] = mem0_filter

            result = self._client.get_all(**kwargs)
            return self._parse_get_all_result(result)

        except Exception as e:
            logger.error(f"Failed to get all memories for user {user_id}: {e}")
            return MemorySearchResult(memories=[], total=0)

    def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: The memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        try:
            result = self._client.get(memory_id)
            if result:
                return self._parse_single_memory(result)
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    def update(self, memory_id: str, content: str) -> Memory | None:
        """Update an existing memory.

        Args:
            memory_id: The memory identifier.
            content: The new content.

        Returns:
            The updated Memory if found, None otherwise.
        """
        try:
            result = self._client.update(memory_id, content)
            if result:
                return self._parse_single_memory(result)
            return None
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return None

    def delete(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: The memory identifier.

        Returns:
            True if deleted, False if not found.
        """
        try:
            self._client.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def delete_all(self, user_id: str) -> int:
        """Delete all memories for a user.

        Args:
            user_id: The user identifier.

        Returns:
            Number of memories deleted.
        """
        try:
            # Get count before deletion
            all_memories = self.get_all(user_id)
            count = len(all_memories.memories)

            self._client.delete_all(user_id=user_id)
            return count
        except Exception as e:
            logger.error(f"Failed to delete all memories for user {user_id}: {e}")
            return 0

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
        if not memories:
            return "No relevant memories found for this user."

        items = memories[:max_items] if max_items else memories
        formatted = []

        for mem in items:
            if include_category and mem.category:
                formatted.append(f"- [{mem.category}] {mem.content}")
            else:
                formatted.append(f"- {mem.content}")

        return "User context from memory:\n" + "\n".join(formatted)

    def add_with_expiration(
        self,
        content: str | list[dict[str, str]],
        user_id: str,
        expires_in: timedelta,
        **kwargs: Any,
    ) -> list[Memory]:
        """Add a memory with automatic expiration.

        Note: expiration_date is not supported in mem0ai v1.0.1 open-source version.
        This method stores the expiration info in metadata for reference but
        does not actually expire the memory automatically.

        Args:
            content: The content to store.
            user_id: The user identifier.
            expires_in: Duration until expiration.
            **kwargs: Additional arguments passed to add().

        Returns:
            List of created Memory objects.
        """
        expiration_date = datetime.now() + expires_in
        # Store expiration in metadata since the API doesn't support it directly
        metadata = kwargs.get("metadata", {}) or {}
        metadata["expiration_date"] = expiration_date.isoformat()
        kwargs["metadata"] = metadata
        return self.add(
            content=content,
            user_id=user_id,
            **kwargs,
        )

    def search_by_category(
        self,
        query: str,
        user_id: str,
        category: str,
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        """Convenience method to search within a specific category.

        Args:
            query: The search query.
            user_id: The user identifier.
            category: The category to filter by.
            limit: Maximum results.

        Returns:
            MemorySearchResult with matching memories.
        """
        filters = MemoryFilter(category=category)
        return self.search(query, user_id, limit=limit, filters=filters)

    def search_by_categories(
        self,
        query: str,
        user_id: str,
        categories: list[str],
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        """Convenience method to search across multiple categories.

        Args:
            query: The search query.
            user_id: The user identifier.
            categories: List of categories to include.
            limit: Maximum results.

        Returns:
            MemorySearchResult with matching memories.
        """
        filters = MemoryFilter(categories_in=categories)
        return self.search(query, user_id, limit=limit, filters=filters)

    # Private helper methods

    def _parse_add_result(self, result: Any, user_id: str) -> list[Memory]:
        """Parse mem0 add result into Memory objects."""
        memories = []

        if not result:
            return memories

        # Handle different result formats from mem0
        results_list = result.get("results", []) if isinstance(result, dict) else []

        for item in results_list:
            memory = self._parse_single_memory(item, default_user_id=user_id)
            if memory:
                memories.append(memory)

        return memories

    def _parse_search_result(self, result: Any) -> MemorySearchResult:
        """Parse mem0 search result into MemorySearchResult."""
        if not result:
            return MemorySearchResult(memories=[], total=0)

        results_list = result.get("results", []) if isinstance(result, dict) else []
        memories = []

        for item in results_list:
            memory = self._parse_single_memory(item)
            if memory:
                memories.append(memory)

        return MemorySearchResult(
            memories=memories,
            total=len(memories),
        )

    def _parse_get_all_result(self, result: Any) -> MemorySearchResult:
        """Parse mem0 get_all result into MemorySearchResult."""
        if not result:
            return MemorySearchResult(memories=[], total=0)

        # get_all may return a list directly or a dict with "results"
        if isinstance(result, list):
            results_list = result
        elif isinstance(result, dict):
            results_list = result.get("results", result.get("memories", []))
        else:
            results_list = []

        memories = []
        for item in results_list:
            memory = self._parse_single_memory(item)
            if memory:
                memories.append(memory)

        return MemorySearchResult(
            memories=memories,
            total=len(memories),
        )

    def _parse_single_memory(
        self, item: dict[str, Any], default_user_id: str = ""
    ) -> Memory | None:
        """Parse a single memory item from mem0 format."""
        if not item:
            return None

        memory_id = item.get("id", item.get("memory_id", ""))
        content = item.get("memory", item.get("text", item.get("content", "")))
        user_id = item.get("user_id", default_user_id)

        if not memory_id or not content:
            return None

        # Parse created_at if present
        created_at = None
        if "created_at" in item:
            with contextlib.suppress(ValueError, TypeError):
                created_at = datetime.fromisoformat(item["created_at"])

        return Memory(
            id=memory_id,
            content=content,
            user_id=user_id,
            category=item.get("category"),
            metadata=item.get("metadata", {}),
            created_at=created_at,
        )


def create_mem0_service(config: MemoryConfig | None = None) -> Mem0MemoryService:
    """Factory function to create a configured Mem0MemoryService.

    Reuses the existing mem0 client factory to avoid config duplication.

    Args:
        config: Optional memory configuration.

    Returns:
        Configured Mem0MemoryService instance.
    """
    from app.infrastructure.memory.mem0_client import get_mem0_client

    mem0_client = get_mem0_client()
    return Mem0MemoryService(mem0_client, config or create_default_memory_config())


# Singleton instance
_mem0_service: Mem0MemoryService | None = None


def get_mem0_service() -> Mem0MemoryService:
    """Get the singleton Mem0MemoryService instance."""
    global _mem0_service
    if _mem0_service is None:
        _mem0_service = create_mem0_service()
    return _mem0_service
