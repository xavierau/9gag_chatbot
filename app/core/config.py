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


settings = Settings()
