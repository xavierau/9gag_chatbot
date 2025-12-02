import logging
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.routes import chat, health, internal, oauth
from app.core.config import settings
from app.infrastructure.llm.dspy_config import configure_dspy

# Configure logging to show in terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set specific loggers to DEBUG for detailed context visibility
logging.getLogger("app.api.routes.chat").setLevel(logging.DEBUG)
logging.getLogger("app.infrastructure.llm.agent").setLevel(logging.DEBUG)
logging.getLogger("app.infrastructure.mcp").setLevel(logging.DEBUG)
logging.getLogger("app.infrastructure.mcp.client").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: configure DSPy
    configure_dspy()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router, prefix="/api/v1")
app.include_router(oauth.router, prefix="/api/v1/oauth", tags=["oauth"])
app.include_router(internal.router, prefix="/api/v1/internal", tags=["internal"])
