"""Domain models for memory management.

This module defines the core domain models for memory configuration,
categories, filters, and search results. These models are independent
of any specific memory implementation (mem0, etc.).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CategoryType(str, Enum):
    """Standard category types for personal AI assistants."""

    PREFERENCES = "preferences"
    CONTEXT = "context"
    GOALS = "goals"
    CONSTRAINTS = "constraints"
    DECISIONS = "decisions"
    PERSONAL = "personal"
    PROCEDURAL = "procedural"


@dataclass(frozen=True)
class MemoryCategory:
    """A category for organizing memories.

    Attributes:
        name: Unique identifier for the category.
        description: Description used by the classifier to assign memories.
    """

    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict format expected by mem0."""
        return {self.name: self.description}


@dataclass(frozen=True)
class MemoryInstructions:
    """Instructions for what to extract and ignore from conversations.

    Attributes:
        extract: List of things to extract and remember.
        ignore: List of things to ignore and not store.
    """

    extract: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert to prompt string for mem0 custom_instructions."""
        lines = []
        if self.extract:
            lines.append("EXTRACT and remember:")
            for item in self.extract:
                lines.append(f"- {item}")

        if self.ignore:
            if lines:
                lines.append("")
            lines.append("DO NOT store:")
            for item in self.ignore:
                lines.append(f"- {item}")

        return "\n".join(lines)


@dataclass
class MemoryFilter:
    """Filter criteria for memory search and retrieval.

    Supports comparison operators and logical combinations
    as defined by mem0's enhanced metadata filtering.
    """

    category: str | None = None
    categories_in: list[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    metadata: dict[str, Any] | None = None

    def to_mem0_filter(self) -> dict[str, Any]:
        """Convert to mem0 filter format."""
        conditions: list[dict[str, Any]] = []

        if self.category:
            conditions.append({"category": self.category})

        if self.categories_in:
            conditions.append({"categories": {"in": self.categories_in}})

        if self.created_after:
            conditions.append(
                {"created_at": {"gte": self.created_after.strftime("%Y-%m-%d")}}
            )

        if self.created_before:
            conditions.append(
                {"created_at": {"lte": self.created_before.strftime("%Y-%m-%d")}}
            )

        if self.metadata:
            for key, value in self.metadata.items():
                conditions.append({f"metadata.{key}": value})

        if not conditions:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"AND": conditions}


@dataclass
class Memory:
    """A single memory item.

    Attributes:
        id: Unique identifier for the memory.
        content: The memory content/text.
        user_id: The user this memory belongs to.
        category: Optional category assignment.
        metadata: Additional metadata.
        created_at: When the memory was created.
    """

    id: str
    content: str
    user_id: str
    category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class MemorySearchResult:
    """Result from a memory search operation.

    Attributes:
        memories: List of matching memories.
        total: Total number of matches (may be more than returned).
    """

    memories: list[Memory]
    total: int = 0


@dataclass
class MemoryConfig:
    """Configuration for the memory service.

    This is the main configuration object that holds categories,
    instructions, and default settings for memory operations.

    Attributes:
        categories: Custom categories for memory classification.
        instructions: What to extract and ignore from conversations.
        default_limit: Default number of memories to retrieve.
        agent_id: Optional agent identifier for agent-specific memories.
    """

    categories: list[MemoryCategory] = field(default_factory=list)
    instructions: MemoryInstructions | None = None
    default_limit: int = 5
    agent_id: str | None = None

    def get_categories_dict(self) -> list[dict[str, str]]:
        """Get categories in mem0 format."""
        return [cat.to_dict() for cat in self.categories]


# Pre-defined category sets for common use cases


def get_personal_assistant_categories() -> list[MemoryCategory]:
    """Get recommended categories for The Captain - a 9gag-style AI with deep personalization."""
    return [
        MemoryCategory(
            name="preferences",
            description=(
                "User's vibe, humor tolerance (how much roasting they can take), "
                "meme literacy level, preferred Captain features (banana mode, TL;DR, sauce), "
                "how they like info delivered (sarcastic, direct, with references), "
                "and what annoys them (corporate speak, cringe, being patronized)"
            ),
        ),
        MemoryCategory(
            name="context",
            description=(
                "User's real situation: their actual job/projects (not just what they claim), "
                "their internet culture background (normie to veteran scale), "
                "what they're REALLY struggling with (read between the lines), "
                "their procrastination patterns, scrolling habits, and chronically-online status"
            ),
        ),
        MemoryCategory(
            name="goals",
            description=(
                "What user says they want vs. what they actually need, "
                "their stated objectives and deadlines, "
                "patterns in goal-setting (overpromiser? realistic? delusional?), "
                "follow-through rate, and what motivates them (clout? learning? fixing a mess?)"
            ),
        ),
        MemoryCategory(
            name="procedural",
            description=(
                "User's actual workflows and how they get things done, "
                "their tech stack and tools, debugging/problem-solving patterns, "
                "repeated mistakes they make, successful strategies that worked before, "
                "and their preferred learning style (RTFM? trial-and-error? YouTube?)"
            ),
        ),
    ]


def get_personal_assistant_instructions() -> MemoryInstructions:
    """Get memory instructions for The Captain - optimized for 9gag-style personalization."""
    return MemoryInstructions(
        extract=[
            "User's humor style and meme literacy (do they get the references? what makes them laugh?)",
            "How they respond to sarcasm and roasting (can they take it? do they dish it back?)",
            "Their actual problems vs. what they say (read between the lines of complaints)",
            "Technical skills and knowledge gaps (what do they pretend to know vs. actually know?)",
            "Patterns in how they ask for help (do they RTFM first? panic immediately? blame tools?)",
            "Their internet culture level (normie, casual, veteran, or touch-grass-resistant)",
            "Preferred Captain features they use/request (banana mode, sauce, TL;DR, etc.)",
            "What genuinely helps them vs. what they think they need",
            "Their procrastination tells and productivity patterns",
            "Workflows, tech stack, and tools they actually use (not just mention)",
            "Mistakes they repeat and lessons they've learned",
            "Goals and deadlines with their track record of follow-through",
            "What motivates them (learning, fixing mess, clout, avoiding embarrassment)",
            "Topics that trigger rants or strong opinions",
        ],
        ignore=[
            "Generic 'hello' and 'thanks' (we're not corporate)",
            "Rage bait and political rants (deflect these)",
            "Hypothetical 'what if' scenarios without context",
            "Temporary mood states unless they're patterns",
            "Obvious sarcasm or jokes from user (don't over-analyze)",
            "Questions they can easily Google (unless revealing a learning gap)",
        ],
    )


def create_default_memory_config() -> MemoryConfig:
    """Create the default memory configuration for The Captain (9gag-style AI assistant)."""
    return MemoryConfig(
        categories=get_personal_assistant_categories(),
        instructions=get_personal_assistant_instructions(),
        default_limit=5,
    )
