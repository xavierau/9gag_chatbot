import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/",
        json={"message": "Hello", "user_id": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Echo: Hello"
    assert data["user_id"] == "test_user"


@pytest.mark.asyncio
async def test_chat_endpoint_minimal(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/",
        json={"message": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Echo: Hello"
