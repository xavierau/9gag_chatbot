"""SQLAlchemy ORM models for the Chatbot application.

This module defines the database models for:
- Conversation sessions and messages
- Expense tracking and categories
- Note management with semantic search

Uses PostgreSQL with JSONB for metadata storage and pgvector for embeddings.
"""

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base


def generate_uuid() -> str:
    """Generate a UUID string for primary keys."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class SessionORM(Base):
    """SQLAlchemy model for conversation sessions.

    A session groups related conversation messages together and tracks
    metadata about the conversation lifecycle.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationship to messages
    messages: Mapped[list["ConversationMessageORM"]] = relationship(
        "ConversationMessageORM",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationMessageORM.created_at",
    )

    __table_args__ = (
        Index("ix_sessions_user_id_created_at", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<SessionORM(id={self.id!r}, user_id={self.user_id!r})>"


class ConversationMessageORM(Base):
    """SQLAlchemy model for conversation messages.

    A message represents a single turn in a conversation, either from
    the user or the assistant.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, index=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    # Relationship to session
    session: Mapped["SessionORM"] = relationship(
        "SessionORM", back_populates="messages"
    )

    __table_args__ = (
        Index(
            "ix_conversation_messages_session_id_created_at",
            "session_id",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationMessageORM(id={self.id!r}, "
            f"session_id={self.session_id!r}, role={self.role!r})>"
        )


# =============================================================================
# Expense Management Models
# =============================================================================


class ExpenseORM(Base):
    """SQLAlchemy model for expense records.

    Tracks individual expenses with amount, category, date, and optional
    metadata like merchant name and payment method.
    """

    __tablename__ = "expenses"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="HKD")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category_name: Mapped[str] = mapped_column(String(100), nullable=False)
    expense_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    merchant_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payment_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_expenses_user_id_expense_date", "user_id", "expense_date"),
        Index("ix_expenses_user_id_category_name", "user_id", "category_name"),
    )

    def __repr__(self) -> str:
        return f"<ExpenseORM(id={self.id!r}, amount={self.amount!r}, category={self.category_name!r})>"


class ExpenseCategoryORM(Base):
    """SQLAlchemy model for custom expense categories.

    Users can create custom categories in addition to the predefined ones.
    Categories are soft-deleted by setting is_active=False.
    """

    __tablename__ = "expense_categories"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        Index(
            "ix_expense_categories_user_id_name",
            "user_id",
            "name",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<ExpenseCategoryORM(id={self.id!r}, name={self.name!r})>"


# =============================================================================
# Note Management Models
# =============================================================================


class NoteORM(Base):
    """SQLAlchemy model for notes.

    Stores user notes with content, AI-generated summaries, and embeddings for semantic search.
    Embeddings are 768-dimensional vectors from Gemini embedding model.
    Summary and key_points are automatically generated when notes are created or updated.
    """

    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_points: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(768), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        Index("ix_notes_user_id_created_at", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<NoteORM(id={self.id!r}, title={self.title!r})>"


# =============================================================================
# Google OAuth Models
# =============================================================================


class GoogleOAuthTokenORM(Base):
    """SQLAlchemy model for Google OAuth tokens.

    Stores encrypted OAuth tokens for Google Calendar and Gmail access.
    One token per user_id (WhatsApp phone number). Linking a new account
    replaces the existing tokens.

    Security:
    - access_token and refresh_token are encrypted at rest using Fernet
    - Tokens are decrypted only when needed for API calls
    - Encryption key stored in GOOGLE_OAUTH_ENCRYPTION_KEY env var
    """

    __tablename__ = "google_oauth_tokens"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    user_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    # Encrypted tokens (Fernet)
    access_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    # Token metadata (not sensitive)
    token_expiry: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    scopes: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    # Google account info (for display purposes)
    google_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    google_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    # Revocation tracking
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<GoogleOAuthTokenORM(id={self.id!r}, user_id={self.user_id!r}, google_email={self.google_email!r})>"


class OAuthStateORM(Base):
    """Temporary storage for OAuth state tokens (CSRF protection).

    These records are short-lived and cleaned up after OAuth completion
    or expiration (10 minutes).
    """

    __tablename__ = "oauth_states"

    id: Mapped[str] = mapped_column(
        String(50), primary_key=True, default=generate_uuid
    )
    state: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    whatsapp_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    def __repr__(self) -> str:
        return f"<OAuthStateORM(id={self.id!r}, user_id={self.user_id!r})>"
