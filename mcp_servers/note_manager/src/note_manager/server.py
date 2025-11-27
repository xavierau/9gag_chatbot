"""FastMCP server for note management.

This module defines the MCP server with tools for note CRUD operations
with integrated AI capabilities:
- Automatic summarization and key point extraction on create/update
- Automatic embedding generation for semantic search
- Semantic search integrated into list_notes via query parameter

SECURITY: The user_id is obtained from MCP_USER_ID environment variable
at server startup and is NEVER exposed as a tool parameter. This ensures
that the AI cannot access or modify notes for other users.

Usage:
    Set MCP_USER_ID environment variable before starting the server:

    MCP_USER_ID=user123 uv run fastmcp run src/note_manager/server.py
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastmcp import Context, FastMCP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from note_manager.ai_service import generate_embedding, generate_query_embedding, generate_summary
from note_manager.config import settings
from note_manager.context import NoteContext
from note_manager.models import NoteCreate, NoteResponse, NoteSearchResult, NoteUpdate
from note_manager.repository import NoteRepository

if TYPE_CHECKING:
    pass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[NoteContext]:
    """Manage application lifecycle with database connection.

    Creates a database engine and session factory on startup, yields the
    NoteContext for use in tools, and cleans up on shutdown.

    Raises:
        ValueError: If MCP_USER_ID environment variable is not set
    """
    # Validate user_id is set
    if not settings.mcp_user_id:
        raise ValueError(
            "MCP_USER_ID environment variable must be set. "
            "This identifies the user for note operations."
        )

    # Create database engine and session factory
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    try:
        yield NoteContext(session_factory=session_factory, user_id=settings.mcp_user_id)
    finally:
        await engine.dispose()


# Create the FastMCP server with lifespan
mcp = FastMCP(
    "note-manager",
    instructions="MCP server for note management with CRUD operations, AI summarization, and semantic search",
    lifespan=app_lifespan,
)


def _get_context(ctx: Context) -> NoteContext:
    """Extract NoteContext from FastMCP Context.

    Args:
        ctx: FastMCP Context with lifespan context

    Returns:
        NoteContext with session factory and user_id
    """
    return ctx.request_context.lifespan_context


# =============================================================================
# Note CRUD Tools
# =============================================================================


@mcp.tool()
async def create_note(
    ctx: Context,
    title: str,
    content: str,
) -> dict:
    """Create a new note with automatic AI summarization and semantic indexing.

    Creates a note with the specified title and content. The note is
    associated with the current user automatically. AI automatically
    generates a summary, key points, and embeddings for semantic search.

    Args:
        title: Note title (max 500 characters)
        content: Note content in Markdown format

    Returns:
        The created note with AI-generated summary and key points.
    """
    # Validate using Pydantic model
    try:
        note_data = NoteCreate(
            title=title,
            content=content,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    note_ctx = _get_context(ctx)

    async with note_ctx.get_session() as session:
        repo = NoteRepository(session, note_ctx.user_id)

        # Create the note first
        note = await repo.create(
            title=note_data.title,
            content=note_data.content,
        )

        # Generate AI fields in parallel
        try:
            summary_task = generate_summary(note_data.title, note_data.content)
            embedding_task = generate_embedding(note_data.title, note_data.content)

            summary_result, embedding = await asyncio.gather(summary_task, embedding_task)

            # Update with AI-generated fields
            note = await repo.update_ai_fields(
                note_id=note.id,
                summary=summary_result.summary,
                key_points=summary_result.key_points,
                embedding=embedding,
            )
        except Exception as e:
            logger.warning(f"AI processing failed for note {note.id}: {e}")
            # Note is still created, just without AI fields

        return NoteResponse.model_validate(note).model_dump(mode="json")


@mcp.tool()
async def get_note(
    ctx: Context,
    note_id: str,
) -> dict:
    """Get a specific note by its ID.

    Args:
        note_id: The unique identifier of the note

    Returns:
        The note details if found, or an error if not found.
    """
    note_ctx = _get_context(ctx)

    async with note_ctx.get_session() as session:
        repo = NoteRepository(session, note_ctx.user_id)
        note = await repo.get_by_id(note_id)

        if note is None:
            return {"error": f"Note not found: {note_id}"}

        return NoteResponse.model_validate(note).model_dump(mode="json")


@mcp.tool()
async def list_notes(
    ctx: Context,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List notes for the current user, with optional semantic search.

    When a query is provided, performs semantic search to find notes
    that are semantically similar to the query, even if they don't
    contain the exact search terms. Results are ranked by relevance.

    When no query is provided, returns notes ordered by last updated
    time (most recent first).

    Args:
        query: Optional natural language search query for semantic search
        limit: Maximum number of notes to return (default: 50)
        offset: Number of notes to skip (default: 0, not used for semantic search)

    Returns:
        A list of notes. For semantic search, includes similarity scores.
    """
    note_ctx = _get_context(ctx)

    async with note_ctx.get_session() as session:
        repo = NoteRepository(session, note_ctx.user_id)

        if query:
            # Semantic search mode
            try:
                query_embedding = await generate_query_embedding(query)
                results = await repo.semantic_search(query_embedding, limit=limit)

                if not results:
                    return {
                        "notes": [],
                        "count": 0,
                        "query": query,
                        "search_mode": "semantic",
                    }

                return {
                    "notes": [
                        NoteSearchResult(
                            id=note.id,
                            user_id=note.user_id,
                            title=note.title,
                            content=note.content,
                            summary=note.summary,
                            key_points=note.key_points,
                            similarity_score=round(score, 4),
                            created_at=note.created_at,
                            updated_at=note.updated_at,
                        ).model_dump(mode="json")
                        for note, score in results
                    ],
                    "count": len(results),
                    "query": query,
                    "search_mode": "semantic",
                }
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                return {"error": f"Semantic search failed: {str(e)}"}

        # Standard list mode
        notes = await repo.list(limit=limit, offset=offset)

        return {
            "notes": [
                NoteResponse.model_validate(note).model_dump(mode="json")
                for note in notes
            ],
            "count": len(notes),
            "limit": limit,
            "offset": offset,
            "search_mode": "chronological",
        }


@mcp.tool()
async def update_note(
    ctx: Context,
    note_id: str,
    title: str | None = None,
    content: str | None = None,
) -> dict:
    """Update an existing note with automatic AI re-processing.

    Updates only the fields that are provided. At least one field must be
    provided for an update. When content is updated, AI automatically
    regenerates the summary, key points, and embeddings.

    Args:
        note_id: The unique identifier of the note to update
        title: New title (optional)
        content: New content in Markdown format (optional)

    Returns:
        The updated note with refreshed AI-generated fields.
    """
    if title is None and content is None:
        return {"error": "At least one of 'title' or 'content' must be provided"}

    # Validate using Pydantic model
    try:
        update_data = NoteUpdate(
            title=title,
            content=content,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    # Build updates dict from non-None values
    updates = update_data.model_dump(exclude_none=True)

    note_ctx = _get_context(ctx)

    async with note_ctx.get_session() as session:
        repo = NoteRepository(session, note_ctx.user_id)

        note = await repo.update(note_id, **updates)

        if note is None:
            return {"error": f"Note not found: {note_id}"}

        # Regenerate AI fields if content changed (or title changed without content)
        # Note: repo.update already clears AI fields when content changes
        if content is not None or (title is not None and note.summary is None):
            try:
                summary_task = generate_summary(note.title, note.content)
                embedding_task = generate_embedding(note.title, note.content)

                summary_result, embedding = await asyncio.gather(summary_task, embedding_task)

                note = await repo.update_ai_fields(
                    note_id=note.id,
                    summary=summary_result.summary,
                    key_points=summary_result.key_points,
                    embedding=embedding,
                )
            except Exception as e:
                logger.warning(f"AI processing failed for note {note.id}: {e}")
                # Note is still updated, just without refreshed AI fields

        return NoteResponse.model_validate(note).model_dump(mode="json")


@mcp.tool()
async def delete_note(
    ctx: Context,
    note_id: str,
) -> dict:
    """Delete a note by its ID.

    Permanently removes the note. This action cannot be undone.

    Args:
        note_id: The unique identifier of the note to delete

    Returns:
        Success status or an error if the note is not found.
    """
    note_ctx = _get_context(ctx)

    async with note_ctx.get_session() as session:
        repo = NoteRepository(session, note_ctx.user_id)
        deleted = await repo.delete(note_id)

        if not deleted:
            return {"error": f"Note not found: {note_id}"}

        return {"success": True, "message": f"Note {note_id} deleted successfully"}


if __name__ == "__main__":
    mcp.run()
