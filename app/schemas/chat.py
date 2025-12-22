from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    image_url: str | None = None
    access_token: str | None = None
    image_headers: dict[str, str] | None = None
    mime_type: str | None = None
    user_id: str | None = None
    session_id: str | None = None

    def get_image_headers(self) -> dict[str, str] | None:
        """Get headers for fetching the image URL.

        Returns custom image_headers if provided, otherwise constructs
        Authorization header from access_token if available.

        Returns:
            Headers dict or None if no authentication is needed.
        """
        if self.image_headers:
            return self.image_headers
        if self.access_token:
            return {"Authorization": f"Bearer {self.access_token}"}
        return None


class ChatResponse(BaseModel):
    response: str
    user_id: str | None = None
    session_id: str | None = None


class LatestSessionResponse(BaseModel):
    """Response model for getting the latest session."""

    user_id: str
    session_id: str | None = None
    has_session: bool
