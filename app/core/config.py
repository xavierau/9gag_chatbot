from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Chatbot"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot"

    # Google Gemini
    google_api_key: str = ""

    # LLM Configuration
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # mem0 Configuration
    mem0_collection_name: str = "chatbot_memories"
    mem0_default_limit: int = 5
    mem0_agent_id: str | None = None

    # Memory Categories (JSON string for env var support)
    # Format: '[{"name": "desc"}, ...]' or empty for defaults
    mem0_custom_categories: str = ""

    # Memory Instructions (can be overridden via env)
    mem0_extract_instructions: str = ""
    mem0_ignore_instructions: str = ""

    # Session Memory Configuration
    session_message_limit: int = 10  # Max messages for agent context

    # Google OAuth Configuration
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://localhost:8000/api/v1/oauth/google/callback"
    google_oauth_encryption_key: str = ""  # Fernet key for token encryption

    # Internal API Security (for MCP servers)
    internal_api_secret: str = ""

    # WhatsApp API Configuration
    whatsapp_api_url: str = ""
    whatsapp_api_token: str = ""

    # Langfuse Observability Configuration
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"  # EU region default


settings = Settings()
