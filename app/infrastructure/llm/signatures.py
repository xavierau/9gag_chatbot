"""DSPy signatures for the chatbot application.

This module defines the signatures (input/output schemas) for DSPy modules.
Signatures use Pydantic-compatible type annotations for type safety and validation.
"""


from typing import Optional

import dspy
from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    """A single message in a conversation history."""

    role: str = Field(description="The role of the speaker: 'user' or 'assistant'")
    content: str = Field(description="The content of the message")


class MemoryContext(BaseModel):
    """Memory context retrieved from mem0 for personalization."""

    content: str = Field(description="The memory content")
    relevance_score: float = Field(
        default=0.0, description="How relevant this memory is to the current query"
    )
    category: str | None = Field(
        default=None, description="Category of the memory (e.g., preference, fact)"
    )


class ChatBotSignature(dspy.Signature):
    """Signature for the main chatbot interaction with memory tools.

    IMPORTANT INSTRUCTIONS FOR MEMORY STORAGE:
    When the user shares any of the following, you MUST use the store_memory tool
    to save it BEFORE responding:
    - Personal preferences (communication style, work habits, how they like things)
    - Context about themselves (job, projects, responsibilities, situation)
    - Goals and objectives (what they're working toward, deadlines, targets)
    - Decisions they've made and their reasoning
    - Constraints or limitations they face
    - Key facts about their role, team, or background

    DO NOT store: greetings, hypotheticals, temporary states, generic questions.

    Use the search_memories tool to recall stored information when relevant.
    """

    # Input fields
    user_message: str = dspy.InputField(
        desc="The user's current message or query to respond to"
    )
    image: Optional[dspy.Image] = dspy.InputField(
        desc="Optional image attachment from the user for visual context or analysis",
        default=None,
    )
    conversation_history: str = dspy.InputField(
        desc="Recent conversation history formatted as 'role: content' pairs, "
        "providing context for the current interaction"
    )
    memory_context: str = dspy.InputField(
        desc="Relevant memories and user information retrieved from the memory system, "
        "used for personalization and context"
    )
    user_id: str = dspy.InputField(
        desc="Unique identifier for the user, used for memory retrieval and storage"
    )

    # Single output field - memory storage happens via tools during ReAct execution
    response: str = dspy.OutputField(
        desc="The assistant's response to the user's message. Should be helpful, "
        "contextual, and personalized based on memory context. "
        "If you stored new information using store_memory, acknowledge it naturally."
    )


class MemorySearchSignature(dspy.Signature):
    """Signature for searching relevant memories."""

    query: str = dspy.InputField(
        desc="The search query to find relevant memories"
    )
    user_id: str = dspy.InputField(desc="The user ID to search memories for")

    relevant_memories: str = dspy.OutputField(
        desc="The retrieved memories formatted as text, or empty if none found"
    )


class ConversationSummarySignature(dspy.Signature):
    """Signature for summarizing conversation history.

    Used to compress long conversation histories while preserving
    important context and information.
    """

    conversation: str = dspy.InputField(
        desc="The full conversation history to summarize"
    )

    summary: str = dspy.OutputField(
        desc="A concise summary of the conversation preserving key topics, "
        "decisions, and context needed for future interactions"
    )


class IntentClassificationSignature(dspy.Signature):
    """Signature for classifying user intent.

    Helps the agent understand what the user wants to accomplish.
    """

    user_message: str = dspy.InputField(desc="The user's message to classify")
    conversation_context: str = dspy.InputField(
        desc="Recent conversation context for better intent understanding"
    )

    intent: str = dspy.OutputField(
        desc="The classified intent: 'question', 'command', 'conversation', "
        "'feedback', 'clarification', or 'other'"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0 for the classification"
    )
