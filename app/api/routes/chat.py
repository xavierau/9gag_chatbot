import io
import logging
import uuid

import dspy
import httpx
from fastapi import APIRouter
from PIL import Image as PILImage

from app.core.dependencies import ChatBotAgentDep, SessionMemoryServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: ChatBotAgentDep,
    session_service: SessionMemoryServiceDep,
) -> ChatResponse:
    """Process a chat message using the DSPy ChatBotAgent.

    Maintains conversation history using session memory for turn-by-turn context.
    Also serves as an audit trail for all conversations.

    Args:
        request: The chat request containing user message and metadata.
        agent: The ChatBotAgent injected via dependency.
        session_service: The SessionMemoryService for conversation history.

    Returns:
        ChatResponse with the agent's response and session_id.
    """
    user_id = request.user_id or str(uuid.uuid4())

    # Get or create session
    session = await session_service.get_or_create_session(
        session_id=request.session_id,
        user_id=user_id,
    )

    # Add user message to session history
    await session_service.add_message(
        session_id=session.id,
        role="user",
        content=request.message,
    )

    # Get conversation history for agent context
    history_result = await session_service.get_history(session.id)
    conversation_history = history_result.to_agent_format()

    # Log session history for debugging
    logger.debug("=" * 60)
    logger.debug("SESSION HISTORY (session_id=%s, user_id=%s)", session.id, user_id)
    logger.debug("=" * 60)
    logger.debug("Total messages in session: %d", history_result.total)
    for i, msg in enumerate(conversation_history):
        logger.debug("[%d] %s: %s", i + 1, msg["role"].upper(), msg["content"][:200])
    logger.debug("=" * 60)

    # Convert image_url to dspy.Image if provided
    image: dspy.Image | None = None
    if request.image_url:
        headers = request.get_image_headers()
        if headers:
            # Download image with custom headers
            async with httpx.AsyncClient() as client:
                response = await client.get(request.image_url, headers=headers)
                response.raise_for_status()
                # Load image bytes directly into PIL
                pil_image = PILImage.open(io.BytesIO(response.content))
                image = dspy.Image.from_PIL(pil_image)
                logger.debug("Image downloaded with headers from: %s", request.image_url)
        else:
            image = dspy.Image.from_url(request.image_url)
            logger.debug("Image loaded from URL: %s", request.image_url)

    # Call agent with conversation history
    result = await agent.aforward(
        user_message=request.message,
        user_id=user_id,
        conversation_history=conversation_history,
        session_id=session.id,
        image=image,
    )

    # Add assistant response to session history
    await session_service.add_message(
        session_id=session.id,
        role="assistant",
        content=result.response,
    )

    return ChatResponse(
        response=result.response,
        user_id=user_id,
        session_id=session.id,
    )
