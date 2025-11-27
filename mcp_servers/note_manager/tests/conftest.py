"""Pytest configuration and fixtures for note manager tests."""

import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import StaticPool, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add paths for imports
# 1. note_manager src for the MCP server code
note_manager_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(note_manager_src))

# 2. Main chatbot project root for app module
chatbot_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chatbot_root))

from note_manager.context import NoteContext

# Import ORM models and Base from the main app
from app.infrastructure.database.models import NoteORM

# Re-export async_sessionmaker for type hints
__all__ = ["async_sessionmaker"]


# Set test environment variables before importing settings
os.environ["MCP_USER_ID"] = "test-user-123"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def test_user_id() -> str:
    """Return the test user ID."""
    return "test-user-123"


@pytest.fixture
async def async_engine():
    """Create an async in-memory SQLite engine for testing.

    Note: SQLite doesn't support pgvector, so we create a simplified table
    without the embedding column for basic CRUD tests.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create a simplified notes table for SQLite (without Vector column)
    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS notes (
                id VARCHAR(50) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                key_points TEXT,
                embedding TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_notes_user_id ON notes (user_id)"))

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def session_factory(async_engine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory for testing."""
    return async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest.fixture
def note_context(session_factory: async_sessionmaker[AsyncSession], test_user_id: str) -> NoteContext:
    """Create a NoteContext for testing."""
    return NoteContext(session_factory=session_factory, user_id=test_user_id)


@pytest.fixture
def mock_context(note_context: NoteContext) -> MagicMock:
    """Create a mock FastMCP Context with NoteContext in lifespan_context."""
    mock_ctx = MagicMock()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = note_context
    return mock_ctx


@pytest.fixture
async def sample_note(db_session: AsyncSession, test_user_id: str) -> NoteORM:
    """Create a sample note for testing."""
    from datetime import datetime, timezone

    note = NoteORM(
        user_id=test_user_id,
        title="Test Note",
        content="This is a test note with some content.",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(note)
    await db_session.flush()
    await db_session.refresh(note)
    return note


@pytest.fixture
async def sample_notes(
    db_session: AsyncSession, test_user_id: str
) -> list[NoteORM]:
    """Create multiple sample notes for testing."""
    from datetime import datetime, timezone, timedelta

    base_time = datetime.now(timezone.utc)

    notes = [
        NoteORM(
            user_id=test_user_id,
            title="Meeting Notes",
            content="# Meeting Notes\n\nDiscussed project timeline and deliverables.",
            created_at=base_time - timedelta(days=3),
            updated_at=base_time - timedelta(days=3),
        ),
        NoteORM(
            user_id=test_user_id,
            title="Python Tips",
            content="## Python Tips\n\n- Use list comprehensions\n- Prefer f-strings",
            created_at=base_time - timedelta(days=2),
            updated_at=base_time - timedelta(days=2),
        ),
        NoteORM(
            user_id=test_user_id,
            title="Shopping List",
            content="- Milk\n- Bread\n- Eggs",
            created_at=base_time - timedelta(days=1),
            updated_at=base_time - timedelta(days=1),
        ),
    ]

    for note in notes:
        db_session.add(note)

    await db_session.flush()

    for note in notes:
        await db_session.refresh(note)

    return notes
