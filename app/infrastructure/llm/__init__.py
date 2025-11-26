"""LLM infrastructure module.

This module provides DSPy-based language model integrations including:
- Agent implementations (ChatBotAgent, SimpleChatBotAgent)
- DSPy signatures for chatbot operations
- Tools for ReAct agent usage
- DSPy configuration utilities
"""

from app.infrastructure.llm.agent import (
    AgentResponse,
    ChatBotAgent,
    ConversationContext,
    SimpleChatBotAgent,
)
from app.infrastructure.llm.dspy_config import configure_dspy, get_lm
from app.infrastructure.llm.signatures import (
    ChatBotSignature,
    ConversationMessage,
    ConversationSummarySignature,
    IntentClassificationSignature,
    MemoryContext,
    MemorySearchSignature,
)
from app.infrastructure.llm.tools import (
    calculate,
    create_memory_list_tool,
    create_memory_search_tool,
    create_memory_store_tool,
    get_current_time,
)

__all__ = [
    # Agent classes
    "ChatBotAgent",
    "SimpleChatBotAgent",
    "AgentResponse",
    "ConversationContext",
    # Signatures
    "ChatBotSignature",
    "ConversationMessage",
    "MemoryContext",
    "MemorySearchSignature",
    "ConversationSummarySignature",
    "IntentClassificationSignature",
    # Tools
    "create_memory_search_tool",
    "create_memory_store_tool",
    "create_memory_list_tool",
    "get_current_time",
    "calculate",
    # Config
    "configure_dspy",
    "get_lm",
]
