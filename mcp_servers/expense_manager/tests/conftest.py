"""Pytest configuration and fixtures for expense manager tests."""

import os
from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from expense_manager.context import ExpenseContext

# Import ORM models and Base from the main app
from app.infrastructure.database.base import Base
from app.infrastructure.database.models import ExpenseCategoryORM, ExpenseORM

# Re-export async_sessionmaker for type hints
__all__ = ["async_sessionmaker"]


# Set test environment variables before importing settings
os.environ["MCP_USER_ID"] = "test-user-123"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def test_user_id() -> str:
    """Return the test user ID."""
    return "test-user-123"


@pytest.fixture
async def async_engine():
    """Create an async in-memory SQLite engine for testing.

    Only creates tables for expense-related models (ExpenseORM and ExpenseCategoryORM)
    to avoid issues with PostgreSQL-specific types (like JSONB) used by other models.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Only create tables for the models we need (expense-related)
    # This avoids issues with PostgreSQL-specific types like JSONB
    tables_to_create = [
        ExpenseORM.__table__,
        ExpenseCategoryORM.__table__,
    ]

    async with engine.begin() as conn:
        for table in tables_to_create:
            await conn.run_sync(table.create, checkfirst=True)

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async database session for testing."""
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def session_factory(async_engine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory for testing."""
    return async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest.fixture
def expense_context(session_factory: async_sessionmaker[AsyncSession], test_user_id: str) -> ExpenseContext:
    """Create an ExpenseContext for testing."""
    return ExpenseContext(session_factory=session_factory, user_id=test_user_id)


@pytest.fixture
def mock_context(expense_context: ExpenseContext) -> MagicMock:
    """Create a mock FastMCP Context with ExpenseContext in lifespan_context."""
    mock_ctx = MagicMock()
    mock_ctx.request_context = MagicMock()
    mock_ctx.request_context.lifespan_context = expense_context
    return mock_ctx


@pytest.fixture
async def sample_expense(db_session: AsyncSession, test_user_id: str) -> ExpenseORM:
    """Create a sample expense for testing."""
    expense = ExpenseORM(
        user_id=test_user_id,
        amount=Decimal("100.50"),
        currency="HKD",
        description="Test lunch",
        category_name="Food & Dining",
        expense_date=date(2024, 1, 15),
        merchant_name="Test Restaurant",
        payment_method="Credit Card",
        notes="Business lunch",
    )
    db_session.add(expense)
    await db_session.flush()
    await db_session.refresh(expense)
    return expense


@pytest.fixture
async def sample_expenses(
    db_session: AsyncSession, test_user_id: str
) -> list[ExpenseORM]:
    """Create multiple sample expenses for testing."""
    expenses = [
        ExpenseORM(
            user_id=test_user_id,
            amount=Decimal("50.00"),
            currency="HKD",
            description="Grocery shopping",
            category_name="Food & Dining",
            expense_date=date(2024, 1, 10),
        ),
        ExpenseORM(
            user_id=test_user_id,
            amount=Decimal("200.00"),
            currency="HKD",
            description="MTR monthly pass",
            category_name="Transportation",
            expense_date=date(2024, 1, 12),
        ),
        ExpenseORM(
            user_id=test_user_id,
            amount=Decimal("350.00"),
            currency="HKD",
            description="Electricity bill",
            category_name="Utilities",
            expense_date=date(2024, 1, 15),
        ),
        ExpenseORM(
            user_id=test_user_id,
            amount=Decimal("150.00"),
            currency="HKD",
            description="Dinner with friends",
            category_name="Food & Dining",
            expense_date=date(2024, 1, 18),
        ),
        ExpenseORM(
            user_id=test_user_id,
            amount=Decimal("80.00"),
            currency="HKD",
            description="Movie tickets",
            category_name="Entertainment",
            expense_date=date(2024, 1, 20),
        ),
    ]

    for expense in expenses:
        db_session.add(expense)

    await db_session.flush()

    for expense in expenses:
        await db_session.refresh(expense)

    return expenses


@pytest.fixture
async def sample_category(
    db_session: AsyncSession, test_user_id: str
) -> ExpenseCategoryORM:
    """Create a sample custom category for testing."""
    category = ExpenseCategoryORM(
        user_id=test_user_id,
        name="Pet Supplies",
        description="Expenses for pets",
    )
    db_session.add(category)
    await db_session.flush()
    await db_session.refresh(category)
    return category
