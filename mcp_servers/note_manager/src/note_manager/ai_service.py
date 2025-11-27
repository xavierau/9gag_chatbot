"""AI service for note summarization and embedding generation.

This module provides async functions for:
- Generating summaries and key points using DSPy
- Creating embeddings using Gemini's embedding model

These are internal utilities used by note CRUD operations, not exposed as MCP tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import dspy
import google.generativeai as genai

from note_manager.config import settings

logger = logging.getLogger(__name__)


@dataclass
class NoteSummary:
    """Result of note summarization."""

    summary: str
    key_points: list[str]


class NoteSummarizerSignature(dspy.Signature):
    """Summarize a note and extract key points."""

    title: str = dspy.InputField(desc="The title of the note")
    content: str = dspy.InputField(desc="The content of the note in Markdown")
    summary: str = dspy.OutputField(desc="A concise summary of the note (2-3 sentences)")
    key_points: list[str] = dspy.OutputField(desc="A list of 3-5 key points from the note")


async def generate_summary(title: str, content: str) -> NoteSummary:
    """Generate a summary and key points for a note.

    Args:
        title: The note title
        content: The note content in Markdown

    Returns:
        NoteSummary with summary text and key points
    """
    lm = dspy.LM("gemini/gemini-2.5-flash", api_key=settings.google_api_key)

    with dspy.settings.context(lm=lm):
        summarizer = dspy.ChainOfThought(NoteSummarizerSignature)
        result = summarizer(title=title, content=content)

    return NoteSummary(
        summary=result.summary,
        key_points=result.key_points,
    )


async def generate_embedding(title: str, content: str) -> list[float]:
    """Generate an embedding vector for a note.

    Combines title and content for better semantic representation.

    Args:
        title: The note title
        content: The note content

    Returns:
        768-dimensional embedding vector
    """
    genai.configure(api_key=settings.google_api_key)

    text_to_embed = f"{title}\n\n{content}"

    embedding_result = genai.embed_content(
        model="models/text-embedding-004",
        content=text_to_embed,
        task_type="retrieval_document",
    )

    return embedding_result["embedding"]


async def generate_query_embedding(query: str) -> list[float]:
    """Generate an embedding vector for a search query.

    Args:
        query: The search query text

    Returns:
        768-dimensional embedding vector optimized for retrieval queries
    """
    genai.configure(api_key=settings.google_api_key)

    embedding_result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )

    return embedding_result["embedding"]
