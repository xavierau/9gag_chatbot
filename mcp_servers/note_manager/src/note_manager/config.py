"""Configuration settings for the Note Manager MCP Server.

Settings are loaded from environment variables with sensible defaults.
The MCP_USER_ID is required for security - it identifies which user's
notes the server operates on.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Note Manager configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database configuration
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/database"

    # User identification (CRITICAL: Must be set at server startup)
    # This is the user_id for note operations - NEVER exposed as tool parameter
    mcp_user_id: str = ""

    # Google API key for Gemini (summarization and embeddings)
    google_api_key: str = ""

    # Pagination defaults
    default_page_limit: int = 50
    max_page_limit: int = 200

    # Debug mode
    debug: bool = False


settings = Settings()
