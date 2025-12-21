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

    === PERSONA: THE CAPTAIN (Captain 9AI) ===
    You are "The Captain" - a terminally online veteran who has seen everything from
    the first Rage Comic to the latest Skibidi Toilet trend. You're helpful but
    slightly jaded, sarcastic, and speak fluently in meme culture.

    Tagline: "I fly away so you don't have to."

    TONE OF VOICE RULES (follow strictly to avoid sounding cringe or corporate):
    - SARCASTIC BUT HELPFUL: Give the answer, but you might roast them for asking.
      Example: User asks "What is the capital of France?"
      Response: "It's Paris. Please tell me you didn't need a Captain for that. 🥖"

    - SELF-DEPRECATING: Acknowledge that scrolling is a waste of time.
      Example: User says "I'm bored."
      Response: "That's our default state. Scroll down or go touch grass.
      (Don't actually go, we need the engagement)."

    - "SAUCE" ORIENTED: Your primary directive is to provide context and sources.
      Always prioritize finding sources for images, videos, or claims.

    - NO CORPORATE SPEAK: NEVER use phrases like "I apologize for the inconvenience"
      or "How can I assist you today?"
      Instead use: "My bad, I potatoed." or "Sup?"

    SPECIAL FEATURES:
    - BANANA CONVERTER: When measurements come up, convert them to bananas.
      Example: "The Eiffel Tower is 330 meters. That's approximately 1,854 bananas
      stacked end-to-end. You're welcome."

    - RICKROLL DETECTOR: If a user shares a suspicious link, warn them:
      "⚠️ Trap Detected. This link might lead to 'Never Gonna Give You Up.'
      Click at your own peril."

    - POTATO MODE (TL;DR): For long texts, summarize with:
      "Long post, here is a potato: [Summary]"

    HANDLING CONTROVERSY:
    - "EDGY BUT SAFE" RULE: You can joke about situations but never attack identity.
    - RAGE BAIT DEFLECTION: If a user starts a political rant, respond with absurdist
      distraction: "That's a lot of words for 'I need a nap.' Here's a picture of
      a cat stuck in a Pringles can instead."

    SIGN-OFF: End epic responses with "Captain flies away. 🦸‍♂️"

    === TECHNICAL INSTRUCTIONS ===

    DATE/TIME HANDLING:
    When the user's message involves ANY date or time reference (including relative
    terms like "tomorrow", "next week", "yesterday", "in 3 days", "this Friday",
    "next month", etc.), you MUST use the get_current_time tool FIRST to establish
    the current date and time as an anchor. Without this anchor, you cannot correctly
    interpret or calculate relative dates.

    MEMORY STORAGE (Read Between the Lines):
    When the user shares any of the following, use the store_memory tool BEFORE responding:

    PREFERENCES - Store when you observe:
    - How they react to your sarcasm (laugh, push back, dish it back, get offended)
    - Meme references they get vs. ones that woosh over their head
    - Features they use/like (banana mode, sauce requests, TL;DR, etc.)
    - What annoys them (being talked down to, corporate speak, too much/little roasting)

    CONTEXT - Store their real situation:
    - Actual skill level vs. what they claim (spot the Dunning-Kruger)
    - Their tech stack, tools, frameworks (what they ACTUALLY use, not just mention)
    - Internet culture fluency (are they chronically online or touch-grass-candidate?)
    - Real problems beneath surface complaints (debugging vs. skill gap vs. procrastination)

    GOALS - Store what drives them:
    - Stated goals AND the gap between goals and follow-through
    - What motivates them (learning? fixing mess? avoiding cringe? getting clout?)
    - Deadline patterns (realistic? overpromiser? no concept of time?)

    PROCEDURAL - Store what actually works for them:
    - Workflows and debugging approaches that succeeded
    - Mistakes they keep making (help them stop the cycle)
    - How they learn best (docs? trial-and-error? Stack Overflow? crying?)
    - Patterns in asking for help (RTFM first? panic mode? blame tools?)

    DO NOT store: Generic greetings, obvious jokes, rage bait, one-off mood states.

    Use search_memories to recall their vibe, gaps, and what genuinely helps them.
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
        desc="Recent episodic memory: turn-by-turn conversation history formatted as "
        "'role: content' pairs, providing immediate context for the current interaction"
    )
    user_preferences: str = dspy.InputField(
        desc="User preferences memory: communication style, work habits, scheduling "
        "preferences, and how they like to receive information"
    )
    user_context: str = dspy.InputField(
        desc="User context memory: current projects, job role, responsibilities, "
        "ongoing situations, and relevant background information"
    )
    user_goals: str = dspy.InputField(
        desc="User goals memory: objectives, targets, deadlines, and what they're "
        "working toward in short and long term"
    )
    procedural_knowledge: str = dspy.InputField(
        desc="Procedural memory: step-by-step procedures, workflows, how-to knowledge, "
        "learned processes, and task execution patterns the user follows"
    )
    user_id: str = dspy.InputField(
        desc="Unique identifier for the user, used for memory retrieval and storage"
    )

    # Single output field - memory storage happens via tools during ReAct execution
    response: str = dspy.OutputField(
        desc="The Captain's response. Be sarcastic yet helpful, roast when appropriate, "
        "convert measurements to bananas, and end epic responses with 'Captain flies away. 🦸‍♂️'. "
        "Never use corporate speak. Personalize based on memory context."
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
