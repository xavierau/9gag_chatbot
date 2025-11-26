import os

from mem0 import Memory

from app.core.config import settings


def create_mem0_client() -> Memory:
    """Create and configure a mem0 Memory client.

    Uses Gemini for LLM and embeddings, with pgvector for vector storage.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    # Set GEMINI_API_KEY from settings if not already in environment
    if settings.google_api_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.google_api_key

    # Parse database URL to extract components for pgvector
    # Expected format: postgresql+asyncpg://user:password@host:port/dbname
    db_url = settings.database_url
    # Remove the async driver prefix for pgvector (it uses psycopg2)
    db_url_sync = db_url.replace("postgresql+asyncpg://", "postgresql://")

    config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": settings.llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/text-embedding-004",
                "embedding_dims": 768,
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": db_url_sync,
                "collection_name": settings.mem0_collection_name,
                "embedding_model_dims": 768,
            },
        },
    }
    return Memory.from_config(config)


_mem0_client: Memory | None = None


def get_mem0_client() -> Memory:
    global _mem0_client
    if _mem0_client is None:
        _mem0_client = create_mem0_client()
    return _mem0_client
