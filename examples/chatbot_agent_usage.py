"""Example usage of the ChatBotAgent.

This file demonstrates how to use the ChatBotAgent with ReAct pattern
for multi-turn conversations with memory integration.
"""

import asyncio

import dspy
from mem0 import Memory

from app.infrastructure.llm.agent import AgentResponse, ChatBotAgent


def setup_dspy():
    """Configure DSPy with Google Gemini."""
    lm = dspy.LM(
        "gemini/gemini-2.5-flash",
        api_key="your-google-api-key",  # Use environment variable in production
    )
    dspy.settings.configure(lm=lm)
    return lm


def setup_memory() -> Memory:
    """Configure mem0 with PostgreSQL backend.

    In production, use environment variables for configuration.
    """
    config = {
        "llm": {
            "provider": "google",
            "config": {
                "model": "gemini-2.5-flash",
                "api_key": "your-google-api-key",
            },
        },
        "embedder": {
            "provider": "google",
            "config": {
                "model": "gemini-embedding-001",
                "api_key": "your-google-api-key",
                "embedding_dims": 1536,
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "dbname": "chatbot",
                "collection_name": "chatbot_memories",
                "embedding_model_dims": 1536,
            },
        },
    }
    return Memory.from_config(config)


# Example 1: Basic synchronous usage
def example_sync_usage():
    """Demonstrate synchronous agent usage."""
    print("=== Synchronous Usage Example ===\n")

    # Setup
    setup_dspy()
    memory = setup_memory()
    agent = ChatBotAgent(memory_client=memory, max_iters=5)

    # Single turn conversation
    response: AgentResponse = agent.forward(
        user_message="Hi! I'm a Python developer and I love building AI applications.",
        user_id="user_123",
        conversation_history=[],
    )

    print(f"Response: {response.response}")
    print(f"Should store memory: {response.should_store_memory}")
    if response.memory_to_store:
        print(f"Memory to store: {response.memory_to_store}")


# Example 2: Async usage (preferred for production)
async def example_async_usage():
    """Demonstrate asynchronous agent usage."""
    print("\n=== Asynchronous Usage Example ===\n")

    # Setup
    setup_dspy()
    memory = setup_memory()
    agent = ChatBotAgent(memory_client=memory, max_iters=5)

    # Multi-turn conversation
    conversation_history = []

    # Turn 1
    response1 = await agent.aforward(
        user_message="What's 25 multiplied by 16?",
        user_id="user_123",
        conversation_history=conversation_history,
    )
    print(f"Turn 1 - User: What's 25 multiplied by 16?")
    print(f"Turn 1 - Agent: {response1.response}\n")

    # Add to history
    conversation_history.append({"role": "user", "content": "What's 25 multiplied by 16?"})
    conversation_history.append({"role": "assistant", "content": response1.response})

    # Turn 2
    response2 = await agent.aforward(
        user_message="Now divide that result by 5",
        user_id="user_123",
        conversation_history=conversation_history,
    )
    print(f"Turn 2 - User: Now divide that result by 5")
    print(f"Turn 2 - Agent: {response2.response}\n")


# Example 3: Concurrent processing of multiple users
async def example_concurrent_users():
    """Demonstrate processing multiple users concurrently."""
    print("\n=== Concurrent Users Example ===\n")

    setup_dspy()
    memory = setup_memory()
    agent = ChatBotAgent(memory_client=memory, max_iters=5)

    # Process multiple users concurrently
    users = [
        ("user_1", "What time is it?"),
        ("user_2", "Calculate 100 divided by 4"),
        ("user_3", "Hello! How are you?"),
    ]

    # Create tasks for concurrent execution
    tasks = [
        agent.aforward(user_message=msg, user_id=uid, conversation_history=[])
        for uid, msg in users
    ]

    # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks)

    for (uid, msg), response in zip(users, responses):
        print(f"User {uid}: {msg}")
        print(f"Agent: {response.response}\n")


# Example 4: Using the agent with reasoning trace
async def example_with_reasoning_trace():
    """Demonstrate agent with reasoning trace enabled."""
    print("\n=== Reasoning Trace Example ===\n")

    setup_dspy()
    memory = setup_memory()

    # Enable reasoning trace
    agent = ChatBotAgent(
        memory_client=memory,
        max_iters=5,
        include_reasoning_trace=True,
    )

    response = await agent.aforward(
        user_message="Search my memories for any preferences I've mentioned",
        user_id="user_123",
        conversation_history=[],
    )

    print(f"Response: {response.response}")
    if response.reasoning_trace:
        print("\nReasoning trace:")
        for step in response.reasoning_trace:
            print(f"  - {step}")


# Example 5: FastAPI integration
"""
# In your FastAPI route handler:

from fastapi import APIRouter, Depends
from app.core.dependencies import ChatBotAgentDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: ChatBotAgentDep,
):
    # Use the async method for non-blocking operation
    result = await agent.aforward(
        user_message=request.message,
        user_id=request.user_id or "anonymous",
        conversation_history=[],  # Load from session if needed
        session_id=request.session_id,
    )

    return ChatResponse(
        response=result.response,
        user_id=request.user_id,
        session_id=request.session_id,
    )
"""


if __name__ == "__main__":
    # Note: These examples require proper environment setup
    # - Google API key configured
    # - PostgreSQL with pgvector running
    # - mem0 configured

    print("ChatBotAgent Usage Examples")
    print("=" * 50)
    print("\nNote: Uncomment and run the examples after setting up:")
    print("- Google API key (GOOGLE_API_KEY)")
    print("- PostgreSQL with pgvector extension")
    print("- mem0 configuration")

    # Uncomment to run examples:
    # example_sync_usage()
    # asyncio.run(example_async_usage())
    # asyncio.run(example_concurrent_users())
    # asyncio.run(example_with_reasoning_trace())
