from collections.abc import AsyncGenerator
from typing import Annotated

import dspy
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.services import MemoryService, SessionMemoryService
from app.infrastructure.database.session import get_db_session
from app.infrastructure.llm.agent import ChatBotAgent, SimpleChatBotAgent
from app.infrastructure.llm.dspy_config import get_lm
from app.infrastructure.memory.mem0_service import (
    Mem0MemoryService,
    get_mem0_service,
)
from app.infrastructure.memory.session_service import PgSessionMemoryService

DbSession = Annotated[AsyncSession, Depends(get_db_session)]


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async for session in get_db_session():
        yield session


def get_memory_service() -> Mem0MemoryService:
    """Get the memory service with custom categories and instructions."""
    return get_mem0_service()


def get_language_model() -> dspy.LM:
    return get_lm()


def get_chatbot_agent(
    memory_service: Mem0MemoryService = Depends(get_memory_service),
) -> ChatBotAgent:
    """Create a ChatBotAgent instance with the configured memory service.

    The agent uses ReAct pattern for reasoning and tool usage.
    DSPy must be configured before using this dependency.

    Args:
        memory_service: The MemoryService from dependency injection.

    Returns:
        A configured ChatBotAgent instance.
    """
    # Ensure DSPy is configured
    get_lm()
    return ChatBotAgent(memory_service=memory_service, max_iters=6)


def get_simple_chatbot_agent(
    memory_service: Mem0MemoryService = Depends(get_memory_service),
) -> SimpleChatBotAgent:
    """Create a SimpleChatBotAgent instance for basic Q&A scenarios.

    This agent uses ChainOfThought without tool usage.
    Suitable for simpler interactions that don't require tools.

    Args:
        memory_service: The MemoryService from dependency injection.

    Returns:
        A configured SimpleChatBotAgent instance.
    """
    # Ensure DSPy is configured
    get_lm()
    return SimpleChatBotAgent(memory_service=memory_service)


def get_session_memory_service(
    db: AsyncSession = Depends(get_db),
) -> PgSessionMemoryService:
    """Create a session memory service instance.

    Args:
        db: AsyncSession from dependency injection.

    Returns:
        A PgSessionMemoryService instance.
    """
    return PgSessionMemoryService(db)


MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
SessionMemoryServiceDep = Annotated[
    SessionMemoryService, Depends(get_session_memory_service)
]
LanguageModel = Annotated[dspy.LM, Depends(get_language_model)]
ChatBotAgentDep = Annotated[ChatBotAgent, Depends(get_chatbot_agent)]
SimpleChatBotAgentDep = Annotated[SimpleChatBotAgent, Depends(get_simple_chatbot_agent)]
