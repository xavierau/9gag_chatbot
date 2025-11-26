"""Configuration settings for the Expense Manager MCP Server.

Settings are loaded from environment variables with sensible defaults.
The MCP_USER_ID is required for security - it identifies which user's
expenses the server operates on.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Expense Manager configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database configuration
    # NOTE: No default credentials - must be set via environment variable or .env file
    # Example: DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/database"

    # User identification (CRITICAL: Must be set at server startup)
    # This is the user_id for expense operations - NEVER exposed as tool parameter
    mcp_user_id: str = ""

    # Default currency for new expenses
    default_currency: str = "HKD"

    # Pagination defaults
    default_page_limit: int = 50
    max_page_limit: int = 200

    # Debug mode
    debug: bool = False


settings = Settings()
