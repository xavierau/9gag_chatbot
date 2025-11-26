"""Expense context for MCP server lifespan management.

The ExpenseContext holds the session factory and user_id for the
current server instance. It is created during server startup and
made available to all tools via FastMCP's lifespan context.

Each tool call creates a new session from the factory to ensure:
- Clean session state for each request
- No stale data accumulation
- Proper transaction boundaries
- Memory-efficient operation over long server lifetimes

SECURITY: user_id is set from MCP_USER_ID environment variable
at server startup and is NEVER exposed as a tool parameter.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass
class ExpenseContext:
    """Context for expense operations within the MCP server.

    Attributes:
        session_factory: Async session factory for creating new sessions per request
        user_id: User identifier (from MCP_USER_ID env var)
    """

    session_factory: async_sessionmaker[AsyncSession]
    user_id: str

    def __post_init__(self) -> None:
        """Validate context after initialization."""
        if not self.user_id:
            raise ValueError("user_id must be provided (set MCP_USER_ID env var)")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a new database session for a single operation.

        This context manager ensures:
        - A fresh session for each tool call
        - Automatic commit on success
        - Automatic rollback on failure
        - Proper session cleanup

        Yields:
            AsyncSession: A new database session

        Example:
            async with ctx.get_session() as session:
                repo = ExpenseRepository(session, ctx.user_id)
                expense = await repo.create(...)
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
