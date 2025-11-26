"""PostgreSQL-based session memory service implementation.

This module provides the concrete implementation of SessionMemoryService
using PostgreSQL for persistent storage of conversation sessions and messages.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.session import ConversationMessage, Session, SessionHistoryResult
from app.infrastructure.database.models import ConversationMessageORM, SessionORM

logger = logging.getLogger(__name__)


class PgSessionMemoryService:
    """PostgreSQL-based implementation of SessionMemoryService.

    Provides persistent storage for conversation sessions and messages,
    supporting turn-by-turn conversation history retrieval for the
    DSPy agent and audit trail functionality.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialize the service with a database session.

        Args:
            db: AsyncSession for database operations.
        """
        self.db = db
        self.message_limit = settings.session_message_limit

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
        session_orm = SessionORM(
            id=str(uuid.uuid4()),
            user_id=user_id,
            metadata_=metadata or {},
        )
        self.db.add(session_orm)
        await self.db.flush()
        await self.db.refresh(session_orm)

        logger.debug(f"Created session {session_orm.id} for user {user_id}")
        return self._to_domain(session_orm)

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            The Session if found, None otherwise.
        """
        result = await self.db.execute(
            select(SessionORM).where(SessionORM.id == session_id)
        )
        session_orm = result.scalar_one_or_none()

        if session_orm is None:
            return None

        return self._to_domain(session_orm)

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
        if session_id:
            session = await self.get_session(session_id)
            if session:
                return session

        return await self.create_session(user_id, metadata)

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
        result = await self.db.execute(
            select(SessionORM)
            .where(SessionORM.user_id == user_id)
            .order_by(desc(SessionORM.created_at))
            .limit(limit)
            .offset(offset)
        )
        return [self._to_domain(s) for s in result.scalars().all()]

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
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'.")

        # Create the message
        msg_orm = ConversationMessageORM(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            metadata_=metadata or {},
        )
        self.db.add(msg_orm)

        # Update session message count and updated_at
        result = await self.db.execute(
            select(SessionORM).where(SessionORM.id == session_id)
        )
        session_orm = result.scalar_one_or_none()
        if session_orm:
            session_orm.message_count += 1
            session_orm.updated_at = datetime.now(timezone.utc)

        await self.db.flush()
        await self.db.refresh(msg_orm)

        logger.debug(
            f"Added {role} message to session {session_id}: "
            f"{content[:50]}{'...' if len(content) > 50 else ''}"
        )
        return self._msg_to_domain(msg_orm)

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
        limit = limit or self.message_limit

        # Get session
        session = await self.get_session(session_id)

        # Get most recent messages (descending), then reverse for chronological order
        result = await self.db.execute(
            select(ConversationMessageORM)
            .where(ConversationMessageORM.session_id == session_id)
            .order_by(desc(ConversationMessageORM.created_at))
            .limit(limit)
        )
        messages = [self._msg_to_domain(m) for m in result.scalars().all()]
        messages.reverse()  # Chronological order

        # Get total count
        total_result = await self.db.execute(
            select(func.count())
            .select_from(ConversationMessageORM)
            .where(ConversationMessageORM.session_id == session_id)
        )
        total = total_result.scalar() or 0

        return SessionHistoryResult(
            messages=messages,
            total=total,
            session=session,
        )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session identifier.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.db.execute(
            select(SessionORM).where(SessionORM.id == session_id)
        )
        session_orm = result.scalar_one_or_none()

        if session_orm is None:
            return False

        await self.db.delete(session_orm)
        await self.db.flush()

        logger.info(f"Deleted session {session_id}")
        return True

    def _to_domain(self, orm: SessionORM) -> Session:
        """Convert ORM model to domain model.

        Args:
            orm: SessionORM instance.

        Returns:
            Session domain model.
        """
        return Session(
            id=orm.id,
            user_id=orm.user_id,
            created_at=orm.created_at,
            updated_at=orm.updated_at,
            metadata=orm.metadata_,
            message_count=orm.message_count,
        )

    def _msg_to_domain(self, orm: ConversationMessageORM) -> ConversationMessage:
        """Convert ORM model to domain model.

        Args:
            orm: ConversationMessageORM instance.

        Returns:
            ConversationMessage domain model.
        """
        return ConversationMessage(
            id=orm.id,
            session_id=orm.session_id,
            role=orm.role,
            content=orm.content,
            created_at=orm.created_at,
            metadata=orm.metadata_,
        )
