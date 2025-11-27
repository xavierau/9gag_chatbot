"""Repository layer for note data access.

This module provides the data access layer for notes.
It uses SQLAlchemy for database operations and follows the Repository pattern.

All methods require a user_id parameter which is obtained from the
NoteContext (set via MCP_USER_ID environment variable).

ORM models are imported from the main app to maintain a single source of truth
and avoid DRY violations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models from the main app to avoid duplication
from app.infrastructure.database.models import NoteORM


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class NoteRepository:
    """Repository for note CRUD operations."""

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        """Initialize repository with database session and user context.

        Args:
            db: Async database session
            user_id: User identifier (from MCP_USER_ID env var)
        """
        self.db = db
        self.user_id = user_id

    async def create(
        self,
        title: str,
        content: str,
    ) -> NoteORM:
        """Create a new note.

        Args:
            title: Note title
            content: Note content in Markdown

        Returns:
            The created note ORM object
        """
        note = NoteORM(
            user_id=self.user_id,
            title=title,
            content=content,
        )
        self.db.add(note)
        await self.db.flush()
        await self.db.refresh(note)
        return note

    async def get_by_id(self, note_id: str) -> NoteORM | None:
        """Get a note by ID.

        Args:
            note_id: The note ID

        Returns:
            The note if found and belongs to user, None otherwise
        """
        result = await self.db.execute(
            select(NoteORM).where(
                and_(
                    NoteORM.id == note_id,
                    NoteORM.user_id == self.user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[NoteORM]:
        """List all notes for the user.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of notes ordered by updated_at (most recent first)
        """
        query = (
            select(NoteORM)
            .where(NoteORM.user_id == self.user_id)
            .order_by(NoteORM.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update(
        self,
        note_id: str,
        **updates: Any,
    ) -> NoteORM | None:
        """Update a note.

        Args:
            note_id: The note ID
            **updates: Fields to update (title, content)

        Returns:
            The updated note if found, None otherwise
        """
        # Filter out None values to allow partial updates
        valid_updates = {k: v for k, v in updates.items() if v is not None}

        if not valid_updates:
            # No updates to make, just return the existing note
            return await self.get_by_id(note_id)

        # Add updated_at timestamp
        valid_updates["updated_at"] = utc_now()

        # Clear AI fields when content changes (will be regenerated)
        if "content" in valid_updates:
            valid_updates["embedding"] = None
            valid_updates["summary"] = None
            valid_updates["key_points"] = None

        await self.db.execute(
            update(NoteORM)
            .where(
                and_(
                    NoteORM.id == note_id,
                    NoteORM.user_id == self.user_id,
                )
            )
            .values(**valid_updates)
        )
        await self.db.flush()

        return await self.get_by_id(note_id)

    async def delete(self, note_id: str) -> bool:
        """Delete a note.

        Args:
            note_id: The note ID

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(NoteORM).where(
                and_(
                    NoteORM.id == note_id,
                    NoteORM.user_id == self.user_id,
                )
            )
        )
        await self.db.flush()
        return result.rowcount > 0

    async def update_ai_fields(
        self,
        note_id: str,
        summary: str,
        key_points: list[str],
        embedding: list[float],
    ) -> NoteORM | None:
        """Update AI-generated fields for a note.

        Args:
            note_id: The note ID
            summary: AI-generated summary
            key_points: AI-extracted key points
            embedding: 768-dimensional embedding vector

        Returns:
            The updated note if found, None otherwise
        """
        await self.db.execute(
            update(NoteORM)
            .where(
                and_(
                    NoteORM.id == note_id,
                    NoteORM.user_id == self.user_id,
                )
            )
            .values(
                summary=summary,
                key_points=key_points,
                embedding=embedding,
            )
        )
        await self.db.flush()

        return await self.get_by_id(note_id)

    async def semantic_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
    ) -> list[tuple[NoteORM, float]]:
        """Find similar notes using cosine similarity.

        Args:
            query_embedding: 768-dimensional embedding vector for the query
            limit: Maximum number of results

        Returns:
            List of tuples (note, similarity_score) ordered by similarity (highest first)
        """
        # Use pgvector's cosine distance operator (<=>)
        # Cosine distance = 1 - cosine similarity, so we sort ascending
        # and convert back to similarity score
        query = (
            select(
                NoteORM,
                (1 - NoteORM.embedding.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(
                and_(
                    NoteORM.user_id == self.user_id,
                    NoteORM.embedding.isnot(None),
                )
            )
            .order_by(NoteORM.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        result = await self.db.execute(query)
        rows = result.all()

        return [(row[0], float(row[1])) for row in rows]
