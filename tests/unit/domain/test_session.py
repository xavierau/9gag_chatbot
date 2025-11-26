"""Unit tests for session domain models."""

from datetime import datetime, timezone

import pytest

from app.domain.session import ConversationMessage, Session, SessionHistoryResult


class TestConversationMessage:
    """Tests for ConversationMessage domain model."""

    def test_create_message(self) -> None:
        """Test creating a conversation message with all fields."""
        now = datetime.now(timezone.utc)
        msg = ConversationMessage(
            id="msg-123",
            session_id="session-456",
            role="user",
            content="Hello, world!",
            created_at=now,
            metadata={"token_count": 5},
        )

        assert msg.id == "msg-123"
        assert msg.session_id == "session-456"
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.created_at == now
        assert msg.metadata == {"token_count": 5}

    def test_create_message_default_metadata(self) -> None:
        """Test creating a message with default empty metadata."""
        now = datetime.now(timezone.utc)
        msg = ConversationMessage(
            id="msg-123",
            session_id="session-456",
            role="assistant",
            content="Hi there!",
            created_at=now,
        )

        assert msg.metadata == {}

    def test_to_dict_user_message(self) -> None:
        """Test converting user message to agent format."""
        msg = ConversationMessage(
            id="msg-123",
            session_id="session-456",
            role="user",
            content="What's the weather?",
            created_at=datetime.now(timezone.utc),
        )

        result = msg.to_dict()

        assert result == {"role": "user", "content": "What's the weather?"}

    def test_to_dict_assistant_message(self) -> None:
        """Test converting assistant message to agent format."""
        msg = ConversationMessage(
            id="msg-123",
            session_id="session-456",
            role="assistant",
            content="I can help with that!",
            created_at=datetime.now(timezone.utc),
        )

        result = msg.to_dict()

        assert result == {"role": "assistant", "content": "I can help with that!"}

    def test_message_is_immutable(self) -> None:
        """Test that ConversationMessage is frozen (immutable)."""
        msg = ConversationMessage(
            id="msg-123",
            session_id="session-456",
            role="user",
            content="Hello",
            created_at=datetime.now(timezone.utc),
        )

        with pytest.raises(AttributeError):
            msg.content = "Modified"  # type: ignore


class TestSession:
    """Tests for Session domain model."""

    def test_create_session(self) -> None:
        """Test creating a session with all fields."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="session-123",
            user_id="user-456",
            created_at=now,
            updated_at=now,
            metadata={"client": "web"},
            message_count=5,
        )

        assert session.id == "session-123"
        assert session.user_id == "user-456"
        assert session.created_at == now
        assert session.updated_at == now
        assert session.metadata == {"client": "web"}
        assert session.message_count == 5

    def test_create_session_defaults(self) -> None:
        """Test creating a session with default values."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="session-123",
            user_id="user-456",
            created_at=now,
            updated_at=now,
        )

        assert session.metadata == {}
        assert session.message_count == 0

    def test_session_is_mutable(self) -> None:
        """Test that Session is mutable (can update message_count)."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="session-123",
            user_id="user-456",
            created_at=now,
            updated_at=now,
        )

        session.message_count = 10
        assert session.message_count == 10


class TestSessionHistoryResult:
    """Tests for SessionHistoryResult domain model."""

    def test_create_empty_result(self) -> None:
        """Test creating an empty history result."""
        result = SessionHistoryResult(messages=[])

        assert result.messages == []
        assert result.total == 0
        assert result.session is None

    def test_create_result_with_messages(self) -> None:
        """Test creating a history result with messages."""
        now = datetime.now(timezone.utc)
        messages = [
            ConversationMessage(
                id="msg-1",
                session_id="session-123",
                role="user",
                content="Hello",
                created_at=now,
            ),
            ConversationMessage(
                id="msg-2",
                session_id="session-123",
                role="assistant",
                content="Hi there!",
                created_at=now,
            ),
        ]

        result = SessionHistoryResult(messages=messages, total=2)

        assert len(result.messages) == 2
        assert result.total == 2

    def test_create_result_with_session(self) -> None:
        """Test creating a history result with session object."""
        now = datetime.now(timezone.utc)
        session = Session(
            id="session-123",
            user_id="user-456",
            created_at=now,
            updated_at=now,
        )

        result = SessionHistoryResult(messages=[], total=0, session=session)

        assert result.session is not None
        assert result.session.id == "session-123"

    def test_to_agent_format_empty(self) -> None:
        """Test converting empty history to agent format."""
        result = SessionHistoryResult(messages=[])

        agent_format = result.to_agent_format()

        assert agent_format == []

    def test_to_agent_format_with_messages(self) -> None:
        """Test converting history with messages to agent format."""
        now = datetime.now(timezone.utc)
        messages = [
            ConversationMessage(
                id="msg-1",
                session_id="session-123",
                role="user",
                content="What is 2+2?",
                created_at=now,
            ),
            ConversationMessage(
                id="msg-2",
                session_id="session-123",
                role="assistant",
                content="2+2 equals 4.",
                created_at=now,
            ),
            ConversationMessage(
                id="msg-3",
                session_id="session-123",
                role="user",
                content="Thanks!",
                created_at=now,
            ),
        ]

        result = SessionHistoryResult(messages=messages, total=3)
        agent_format = result.to_agent_format()

        assert agent_format == [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "Thanks!"},
        ]

    def test_to_agent_format_preserves_order(self) -> None:
        """Test that to_agent_format preserves message order."""
        now = datetime.now(timezone.utc)
        messages = [
            ConversationMessage(
                id=f"msg-{i}",
                session_id="session-123",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                created_at=now,
            )
            for i in range(5)
        ]

        result = SessionHistoryResult(messages=messages, total=5)
        agent_format = result.to_agent_format()

        for i, msg_dict in enumerate(agent_format):
            assert msg_dict["content"] == f"Message {i}"
