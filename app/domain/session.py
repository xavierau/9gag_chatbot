"""Domain models for session-based conversation memory.

This module defines the core domain models for conversation sessions,
messages, and history retrieval. These models are independent of any
specific storage implementation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ConversationMessage:
    """A single message in a conversation session.

    Attributes:
        id: Unique identifier for the message.
        session_id: The session this message belongs to.
        role: The role of the message sender ("user" or "assistant").
        content: The message content/text.
        created_at: When the message was created.
        metadata: Additional metadata (e.g., token count, model used).
    """

    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """Convert to format expected by DSPy agent conversation_history."""
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """A conversation session container.

    Sessions group related conversation messages together and track
    metadata about the conversation lifecycle.

    Attributes:
        id: Unique identifier for the session.
        user_id: The user this session belongs to.
        created_at: When the session was created.
        updated_at: When the session was last updated.
        metadata: Additional metadata (e.g., client info, tags).
        message_count: Number of messages in the session.
    """

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    message_count: int = 0


@dataclass
class SessionHistoryResult:
    """Result from retrieving session conversation history.

    Attributes:
        messages: List of conversation messages (chronological order).
        total: Total number of messages in the session.
        session: The session object (optional).
    """

    messages: list[ConversationMessage]
    total: int = 0
    session: Session | None = None

    def to_agent_format(self) -> list[dict[str, str]]:
        """Convert to format expected by DSPy agent conversation_history.

        Returns:
            List of dicts with 'role' and 'content' keys.
        """
        return [msg.to_dict() for msg in self.messages]
