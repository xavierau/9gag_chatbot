from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    image_url: str | None = None
    user_id: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    user_id: str | None = None
    session_id: str | None = None
