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
    """Get recommended categories for a personal AI assistant."""
    return [
        MemoryCategory(
            name="preferences",
            description=(
                "User's communication style, work habits, scheduling preferences, "
                "and how they like to receive information"
            ),
        ),
        MemoryCategory(
            name="context",
            description=(
                "User's current projects, job role, responsibilities, "
                "ongoing situations, and relevant background"
            ),
        ),
        MemoryCategory(
            name="goals",
            description=(
                "User's objectives, targets, deadlines, "
                "and what they're working toward short and long term"
            ),
        ),
    ]


def get_personal_assistant_instructions() -> MemoryInstructions:
    """Get recommended instructions for a personal AI assistant."""
    return MemoryInstructions(
        extract=[
            "User's stated preferences and how they like things done",
            "Background context about their work, projects, or situation",
            "Goals, objectives, and deadlines they mention",
            "Decisions they've made and their reasoning",
            "Constraints, limitations, or restrictions they face",
            "Key facts about their role, team, or responsibilities",
        ],
        ignore=[
            "Greetings, pleasantries, or filler conversation",
            "Hypothetical scenarios or 'what if' discussions",
            "Generic questions without personal context",
            "Temporary states like 'I'm tired today'",
            "Information they explicitly ask to forget",
        ],
    )


def create_default_memory_config() -> MemoryConfig:
    """Create the default memory configuration for personal AI assistant."""
    return MemoryConfig(
        categories=get_personal_assistant_categories(),
        instructions=get_personal_assistant_instructions(),
        default_limit=5,
    )
