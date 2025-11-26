"""Unit tests for PgSessionMemoryService.

These tests use a real test database to verify the session service
implementation works correctly with PostgreSQL.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.infrastructure.memory.session_service import PgSessionMemoryService


@pytest.fixture(scope="function")
async def db_session():
    """Provide a database session for tests.

    Each test gets a fresh session. Tests are responsible for cleanup
    if they create persistent data.
    """
    engine = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
    )
    async_session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def session_service(db_session: AsyncSession) -> PgSessionMemoryService:
    """Create a session service instance for testing."""
    return PgSessionMemoryService(db_session)


class TestCreateSession:
    """Tests for create_session method."""

    @pytest.mark.asyncio
    async def test_create_session_basic(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test creating a basic session."""
        session = await session_service.create_session(user_id="test_user")

        assert session.id is not None
        assert len(session.id) == 36  # UUID format
        assert session.user_id == "test_user"
        assert session.message_count == 0
        assert session.metadata == {}
        assert session.created_at is not None
        assert session.updated_at is not None

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test creating a session with metadata."""
        metadata = {"client": "web", "version": "1.0"}
        session = await session_service.create_session(
            user_id="test_user", metadata=metadata
        )

        assert session.metadata == metadata

    @pytest.mark.asyncio
    async def test_create_multiple_sessions(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test creating multiple sessions for the same user."""
        session1 = await session_service.create_session(user_id="test_user")
        session2 = await session_service.create_session(user_id="test_user")

        assert session1.id != session2.id
        assert session1.user_id == session2.user_id


class TestGetSession:
    """Tests for get_session method."""

    @pytest.mark.asyncio
    async def test_get_existing_session(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test retrieving an existing session."""
        created = await session_service.create_session(user_id="test_user")
        retrieved = await session_service.get_session(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.user_id == created.user_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test retrieving a non-existent session returns None."""
        retrieved = await session_service.get_session("nonexistent-id")

        assert retrieved is None


class TestGetOrCreateSession:
    """Tests for get_or_create_session method."""

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that existing session is returned when session_id is provided."""
        created = await session_service.create_session(user_id="test_user")
        retrieved = await session_service.get_or_create_session(
            session_id=created.id, user_id="test_user"
        )

        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_when_not_found(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that new session is created when session_id is not found."""
        new_session = await session_service.get_or_create_session(
            session_id="nonexistent-id", user_id="test_user"
        )

        assert new_session is not None
        assert new_session.id != "nonexistent-id"
        assert new_session.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_when_none(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that new session is created when session_id is None."""
        new_session = await session_service.get_or_create_session(
            session_id=None, user_id="test_user"
        )

        assert new_session is not None
        assert new_session.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_get_or_create_with_metadata(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that metadata is passed to new session."""
        metadata = {"source": "api"}
        new_session = await session_service.get_or_create_session(
            session_id=None, user_id="test_user", metadata=metadata
        )

        assert new_session.metadata == metadata


class TestListUserSessions:
    """Tests for list_user_sessions method."""

    @pytest.mark.asyncio
    async def test_list_empty(self, session_service: PgSessionMemoryService) -> None:
        """Test listing sessions for a user with no sessions."""
        sessions = await session_service.list_user_sessions(user_id="no_sessions_user")

        assert sessions == []

    @pytest.mark.asyncio
    async def test_list_user_sessions(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test listing sessions for a user."""
        # Create sessions
        await session_service.create_session(user_id="list_test_user")
        await session_service.create_session(user_id="list_test_user")
        await session_service.create_session(user_id="other_user")

        sessions = await session_service.list_user_sessions(user_id="list_test_user")

        assert len(sessions) == 2
        for session in sessions:
            assert session.user_id == "list_test_user"

    @pytest.mark.asyncio
    async def test_list_with_limit(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test listing sessions with limit."""
        # Create sessions
        for _ in range(5):
            await session_service.create_session(user_id="limit_test_user")

        sessions = await session_service.list_user_sessions(
            user_id="limit_test_user", limit=3
        )

        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_list_with_offset(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test listing sessions with offset."""
        # Create sessions
        for _ in range(5):
            await session_service.create_session(user_id="offset_test_user")

        all_sessions = await session_service.list_user_sessions(
            user_id="offset_test_user"
        )
        offset_sessions = await session_service.list_user_sessions(
            user_id="offset_test_user", offset=2
        )

        assert len(offset_sessions) == 3
        assert offset_sessions[0].id == all_sessions[2].id


class TestAddMessage:
    """Tests for add_message method."""

    @pytest.mark.asyncio
    async def test_add_user_message(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test adding a user message."""
        session = await session_service.create_session(user_id="test_user")
        msg = await session_service.add_message(
            session_id=session.id, role="user", content="Hello, world!"
        )

        assert msg.id is not None
        assert msg.session_id == session.id
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.created_at is not None
        assert msg.metadata == {}

    @pytest.mark.asyncio
    async def test_add_assistant_message(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test adding an assistant message."""
        session = await session_service.create_session(user_id="test_user")
        msg = await session_service.add_message(
            session_id=session.id, role="assistant", content="Hi there!"
        )

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_add_message_with_metadata(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test adding a message with metadata."""
        session = await session_service.create_session(user_id="test_user")
        metadata = {"tokens": 10, "model": "test"}
        msg = await session_service.add_message(
            session_id=session.id,
            role="user",
            content="Test message",
            metadata=metadata,
        )

        assert msg.metadata == metadata

    @pytest.mark.asyncio
    async def test_add_message_updates_session_count(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that adding a message updates session message count."""
        session = await session_service.create_session(user_id="test_user")
        assert session.message_count == 0

        await session_service.add_message(
            session_id=session.id, role="user", content="Message 1"
        )
        await session_service.add_message(
            session_id=session.id, role="assistant", content="Response 1"
        )

        updated_session = await session_service.get_session(session.id)
        assert updated_session is not None
        assert updated_session.message_count == 2

    @pytest.mark.asyncio
    async def test_add_message_invalid_role(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that invalid role raises ValueError."""
        session = await session_service.create_session(user_id="test_user")

        with pytest.raises(ValueError, match="Invalid role"):
            await session_service.add_message(
                session_id=session.id, role="invalid", content="Test"
            )


class TestGetHistory:
    """Tests for get_history method."""

    @pytest.mark.asyncio
    async def test_get_empty_history(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test getting history for a session with no messages."""
        session = await session_service.create_session(user_id="test_user")
        result = await session_service.get_history(session.id)

        assert result.messages == []
        assert result.total == 0
        assert result.session is not None
        assert result.session.id == session.id

    @pytest.mark.asyncio
    async def test_get_history_with_messages(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test getting history with messages."""
        session = await session_service.create_session(user_id="test_user")
        await session_service.add_message(
            session_id=session.id, role="user", content="Hello"
        )
        await session_service.add_message(
            session_id=session.id, role="assistant", content="Hi there!"
        )

        result = await session_service.get_history(session.id)

        assert len(result.messages) == 2
        assert result.total == 2
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_get_history_chronological_order(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test that history is returned in chronological order."""
        session = await session_service.create_session(user_id="test_user")
        for i in range(5):
            await session_service.add_message(
                session_id=session.id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        result = await session_service.get_history(session.id)

        for i, msg in enumerate(result.messages):
            assert msg.content == f"Message {i}"

    @pytest.mark.asyncio
    async def test_get_history_with_limit(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test getting history with a limit."""
        session = await session_service.create_session(user_id="test_user")
        for i in range(10):
            await session_service.add_message(
                session_id=session.id, role="user", content=f"Message {i}"
            )

        result = await session_service.get_history(session.id, limit=5)

        assert len(result.messages) == 5
        assert result.total == 10
        # Should get the most recent 5 messages (5-9), in chronological order
        assert result.messages[0].content == "Message 5"
        assert result.messages[4].content == "Message 9"

    @pytest.mark.asyncio
    async def test_get_history_to_agent_format(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test converting history to agent format."""
        session = await session_service.create_session(user_id="test_user")
        await session_service.add_message(
            session_id=session.id, role="user", content="What is 2+2?"
        )
        await session_service.add_message(
            session_id=session.id, role="assistant", content="2+2 equals 4."
        )

        result = await session_service.get_history(session.id)
        agent_format = result.to_agent_format()

        assert agent_format == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ]


class TestDeleteSession:
    """Tests for delete_session method."""

    @pytest.mark.asyncio
    async def test_delete_existing_session(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test deleting an existing session."""
        session = await session_service.create_session(user_id="test_user")
        await session_service.add_message(
            session_id=session.id, role="user", content="Test"
        )

        result = await session_service.delete_session(session.id)

        assert result is True
        assert await session_service.get_session(session.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(
        self, session_service: PgSessionMemoryService
    ) -> None:
        """Test deleting a non-existent session returns False."""
        result = await session_service.delete_session("nonexistent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_cascades_to_messages(
        self, session_service: PgSessionMemoryService, db_session: AsyncSession
    ) -> None:
        """Test that deleting a session also deletes its messages."""
        session = await session_service.create_session(user_id="test_user")
        await session_service.add_message(
            session_id=session.id, role="user", content="Test 1"
        )
        await session_service.add_message(
            session_id=session.id, role="assistant", content="Test 2"
        )

        await session_service.delete_session(session.id)

        # Verify messages are also deleted
        result = await db_session.execute(
            text(
                "SELECT COUNT(*) FROM conversation_messages WHERE session_id = :sid"
            ).bindparams(sid=session.id)
        )
        count = result.scalar()
        assert count == 0
