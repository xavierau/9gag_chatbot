"""Tests for NoteRepository."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from note_manager.repository import NoteRepository
from app.infrastructure.database.models import NoteORM


class TestNoteRepository:
    """Tests for NoteRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_note(self, db_session: AsyncSession, test_user_id: str):
        """Test creating a new note."""
        repo = NoteRepository(db_session, test_user_id)

        note = await repo.create(
            title="My First Note",
            content="This is the content of my first note.",
        )

        assert note.id is not None
        assert note.user_id == test_user_id
        assert note.title == "My First Note"
        assert note.content == "This is the content of my first note."
        assert note.created_at is not None
        assert note.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_note_by_id(self, db_session: AsyncSession, sample_note: NoteORM, test_user_id: str):
        """Test getting a note by ID."""
        repo = NoteRepository(db_session, test_user_id)

        note = await repo.get_by_id(sample_note.id)

        assert note is not None
        assert note.id == sample_note.id
        assert note.title == sample_note.title

    @pytest.mark.asyncio
    async def test_get_note_by_id_not_found(self, db_session: AsyncSession, test_user_id: str):
        """Test getting a non-existent note returns None."""
        repo = NoteRepository(db_session, test_user_id)

        note = await repo.get_by_id("non-existent-id")

        assert note is None

    @pytest.mark.asyncio
    async def test_get_note_by_id_wrong_user(self, db_session: AsyncSession, sample_note: NoteORM):
        """Test that a note cannot be accessed by a different user."""
        repo = NoteRepository(db_session, "different-user")

        note = await repo.get_by_id(sample_note.id)

        assert note is None

    @pytest.mark.asyncio
    async def test_list_notes(self, db_session: AsyncSession, sample_notes: list[NoteORM], test_user_id: str):
        """Test listing all notes for a user."""
        repo = NoteRepository(db_session, test_user_id)

        notes = await repo.list()

        assert len(notes) == 3
        # Should be ordered by updated_at descending
        assert notes[0].title == "Shopping List"  # Most recent

    @pytest.mark.asyncio
    async def test_list_notes_pagination(self, db_session: AsyncSession, sample_notes: list[NoteORM], test_user_id: str):
        """Test listing notes with pagination."""
        repo = NoteRepository(db_session, test_user_id)

        notes = await repo.list(limit=2, offset=0)
        assert len(notes) == 2

        notes = await repo.list(limit=2, offset=2)
        assert len(notes) == 1

    @pytest.mark.asyncio
    async def test_list_notes_empty(self, db_session: AsyncSession, test_user_id: str):
        """Test listing notes when there are none."""
        repo = NoteRepository(db_session, test_user_id)

        notes = await repo.list()

        assert len(notes) == 0

    @pytest.mark.asyncio
    async def test_update_note_title(self, db_session: AsyncSession, sample_note: NoteORM, test_user_id: str):
        """Test updating a note's title."""
        repo = NoteRepository(db_session, test_user_id)

        updated_note = await repo.update(sample_note.id, title="Updated Title")

        assert updated_note is not None
        assert updated_note.title == "Updated Title"
        assert updated_note.content == sample_note.content

    @pytest.mark.asyncio
    async def test_update_note_content(self, db_session: AsyncSession, sample_note: NoteORM, test_user_id: str):
        """Test updating a note's content."""
        repo = NoteRepository(db_session, test_user_id)

        updated_note = await repo.update(sample_note.id, content="New content here.")

        assert updated_note is not None
        assert updated_note.content == "New content here."
        assert updated_note.title == sample_note.title

    @pytest.mark.asyncio
    async def test_update_note_not_found(self, db_session: AsyncSession, test_user_id: str):
        """Test updating a non-existent note returns None."""
        repo = NoteRepository(db_session, test_user_id)

        updated_note = await repo.update("non-existent-id", title="New Title")

        assert updated_note is None

    @pytest.mark.asyncio
    async def test_update_note_wrong_user(self, db_session: AsyncSession, sample_note: NoteORM):
        """Test that a note cannot be updated by a different user."""
        repo = NoteRepository(db_session, "different-user")

        updated_note = await repo.update(sample_note.id, title="Hacked Title")

        assert updated_note is None

    @pytest.mark.asyncio
    async def test_delete_note(self, db_session: AsyncSession, sample_note: NoteORM, test_user_id: str):
        """Test deleting a note."""
        repo = NoteRepository(db_session, test_user_id)

        result = await repo.delete(sample_note.id)

        assert result is True

        # Verify note is deleted
        note = await repo.get_by_id(sample_note.id)
        assert note is None

    @pytest.mark.asyncio
    async def test_delete_note_not_found(self, db_session: AsyncSession, test_user_id: str):
        """Test deleting a non-existent note returns False."""
        repo = NoteRepository(db_session, test_user_id)

        result = await repo.delete("non-existent-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_note_wrong_user(self, db_session: AsyncSession, sample_note: NoteORM):
        """Test that a note cannot be deleted by a different user."""
        repo = NoteRepository(db_session, "different-user")

        result = await repo.delete(sample_note.id)

        assert result is False


class TestNoteContextIntegration:
    """Tests for NoteContext integration."""

    @pytest.mark.asyncio
    async def test_context_creates_session(self, note_context, test_user_id: str):
        """Test that NoteContext creates a valid session."""
        async with note_context.get_session() as session:
            repo = NoteRepository(session, test_user_id)
            note = await repo.create(title="Context Test", content="Testing context.")

            assert note.id is not None

    @pytest.mark.asyncio
    async def test_context_user_id_validation(self, session_factory):
        """Test that NoteContext requires a user_id."""
        from note_manager.context import NoteContext

        with pytest.raises(ValueError, match="user_id must be provided"):
            NoteContext(session_factory=session_factory, user_id="")
