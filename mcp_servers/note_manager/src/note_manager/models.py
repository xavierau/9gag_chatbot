"""Pydantic models for Note Manager MCP Server.

These models handle request validation and response serialization
for note operations.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class NoteCreate(BaseModel):
    """Schema for creating a new note."""

    title: str = Field(..., min_length=1, max_length=500, description="Note title")
    content: str = Field(..., min_length=1, description="Note content in Markdown")


class NoteUpdate(BaseModel):
    """Schema for updating an existing note."""

    title: str | None = Field(None, min_length=1, max_length=500, description="New title")
    content: str | None = Field(None, min_length=1, description="New content in Markdown")


class NoteResponse(BaseModel):
    """Schema for note response.

    Includes AI-generated summary and key points which are
    automatically created when notes are created or updated.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    content: str
    summary: str | None = None
    key_points: list[str] | None = None
    created_at: datetime
    updated_at: datetime


class NoteSearchResult(BaseModel):
    """Schema for search result with similarity score."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str
    content: str
    summary: str | None = None
    key_points: list[str] | None = None
    similarity_score: float
    created_at: datetime
    updated_at: datetime
