"""FastMCP server for expense management.

This module defines the MCP server with tools for expense tracking,
category management, and spending analytics.

SECURITY: The user_id is obtained from MCP_USER_ID environment variable
at server startup and is NEVER exposed as a tool parameter. This ensures
that the AI cannot access or modify expenses for other users.

Usage:
    Set MCP_USER_ID environment variable before starting the server:

    MCP_USER_ID=user123 uv run fastmcp run src/expense_manager/server.py
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal

from fastmcp import Context, FastMCP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from expense_manager.config import settings
from expense_manager.constants import PredefinedCategory
from expense_manager.context import ExpenseContext
from expense_manager.models import (
    CategoryCreate,
    CategoryResponse,
    CategorySummary,
    ExpenseCreate,
    ExpenseListFilter,
    ExpenseResponse,
    ExpenseSummary,
    ExpenseUpdate,
    GroupedExpenseSummary,
    MonthlyTrend,
    MonthlyTrendResponse,
    TopCategoriesResponse,
    TopCategory,
)
from expense_manager.repository import CategoryRepository, ExpenseRepository


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[ExpenseContext]:
    """Manage application lifecycle with database connection.

    Creates a database engine and session factory on startup, yields the
    ExpenseContext for use in tools, and cleans up on shutdown.

    The session factory is stored in the context, allowing each tool call
    to create its own session. This ensures:
    - No stale data accumulation over server lifetime
    - Clean transaction boundaries per request
    - Memory-efficient operation

    Raises:
        ValueError: If MCP_USER_ID environment variable is not set
    """
    # Validate user_id is set
    if not settings.mcp_user_id:
        raise ValueError(
            "MCP_USER_ID environment variable must be set. "
            "This identifies the user for expense operations."
        )

    # Create database engine and session factory
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )

    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    try:
        # Yield context with session factory (not a single session)
        yield ExpenseContext(session_factory=session_factory, user_id=settings.mcp_user_id)
    finally:
        await engine.dispose()


# Create the FastMCP server with lifespan
mcp = FastMCP(
    "expense-manager",
    instructions="MCP server for expense tracking, categorization, and analytics",
    lifespan=app_lifespan,
)


def _get_context(ctx: Context) -> ExpenseContext:
    """Extract ExpenseContext from FastMCP Context.

    Args:
        ctx: FastMCP Context with lifespan context

    Returns:
        ExpenseContext with session factory and user_id
    """
    return ctx.request_context.lifespan_context


# Decimal precision constant for monetary amounts
DECIMAL_PLACES = Decimal("0.01")


def _to_decimal(amount: float) -> Decimal:
    """Convert float to Decimal with proper precision for monetary amounts.

    This function ensures consistent decimal precision by:
    1. Converting float to string first (avoids binary float representation issues)
    2. Applying quantize with ROUND_HALF_UP for 2 decimal places

    Args:
        amount: Float amount to convert

    Returns:
        Decimal with exactly 2 decimal places
    """
    return Decimal(str(amount)).quantize(DECIMAL_PLACES)


# =============================================================================
# Expense CRUD Tools
# =============================================================================


@mcp.tool()
async def create_expense(
    ctx: Context,
    amount: float,
    description: str,
    category: str,
    expense_date: str,
    currency: str | None = None,
    merchant_name: str | None = None,
    payment_method: str | None = None,
    notes: str | None = None,
) -> dict:
    """Create a new expense record.

    Records an expense with the specified amount, description, and category.
    The expense is associated with the current user automatically.

    Args:
        amount: Expense amount (must be positive)
        description: Description of the expense
        category: Category name for the expense
        expense_date: Date of expense in YYYY-MM-DD format
        currency: Currency code (default: HKD)
        merchant_name: Merchant or vendor name
        payment_method: Payment method used
        notes: Additional notes

    Returns:
        The created expense with its assigned ID and timestamps.
    """
    # Parse and validate the expense date
    try:
        parsed_date = date.fromisoformat(expense_date)
    except ValueError:
        return {"error": f"Invalid date format: {expense_date}. Use YYYY-MM-DD format."}

    # Use default currency if not provided
    final_currency = currency if currency else settings.default_currency

    # Validate using Pydantic model with proper decimal precision
    try:
        expense_data = ExpenseCreate(
            amount=_to_decimal(amount),
            description=description,
            category_name=category,
            expense_date=parsed_date,
            currency=final_currency,
            merchant_name=merchant_name,
            payment_method=payment_method,
            notes=notes,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    expense_ctx = _get_context(ctx)

    # Create a new session for this request
    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)

        expense = await repo.create(
            amount=expense_data.amount,
            description=expense_data.description,
            category_name=expense_data.category_name,
            expense_date=expense_data.expense_date,
            currency=expense_data.currency,
            merchant_name=expense_data.merchant_name,
            payment_method=expense_data.payment_method,
            notes=expense_data.notes,
        )

        return ExpenseResponse.model_validate(expense).model_dump(mode="json")


@mcp.tool()
async def get_expense(
    ctx: Context,
    expense_id: str,
) -> dict:
    """Get a specific expense by its ID.

    Args:
        expense_id: The unique identifier of the expense

    Returns:
        The expense details if found, or an error if not found.
    """
    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)
        expense = await repo.get_by_id(expense_id)

        if expense is None:
            return {"error": f"Expense not found: {expense_id}"}

        return ExpenseResponse.model_validate(expense).model_dump(mode="json")


@mcp.tool()
async def list_expenses(
    ctx: Context,
    start_date: str | None = None,
    end_date: str | None = None,
    category: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List expenses with optional filters.

    Args:
        start_date: Filter from date (YYYY-MM-DD)
        end_date: Filter until date (YYYY-MM-DD)
        category: Filter by category name
        min_amount: Minimum expense amount
        max_amount: Maximum expense amount
        limit: Maximum results to return (default: 50)
        offset: Number of results to skip (default: 0)

    Returns:
        A list of expenses matching the criteria, ordered by date (most recent first).
    """
    # Parse dates if provided
    parsed_start = None
    parsed_end = None

    if start_date:
        try:
            parsed_start = date.fromisoformat(start_date)
        except ValueError:
            return {
                "error": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD."
            }

    if end_date:
        try:
            parsed_end = date.fromisoformat(end_date)
        except ValueError:
            return {"error": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD."}

    # Validate using Pydantic model with proper decimal precision
    try:
        filters = ExpenseListFilter(
            start_date=parsed_start,
            end_date=parsed_end,
            category=category,
            min_amount=_to_decimal(min_amount) if min_amount is not None else None,
            max_amount=_to_decimal(max_amount) if max_amount is not None else None,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)

        expenses = await repo.list(
            start_date=filters.start_date,
            end_date=filters.end_date,
            category=filters.category,
            min_amount=filters.min_amount,
            max_amount=filters.max_amount,
            limit=filters.limit,
            offset=filters.offset,
        )

        return {
            "expenses": [
                ExpenseResponse.model_validate(e).model_dump(mode="json") for e in expenses
            ],
            "count": len(expenses),
            "limit": filters.limit,
            "offset": filters.offset,
        }


@mcp.tool()
async def update_expense(
    ctx: Context,
    expense_id: str,
    amount: float | None = None,
    description: str | None = None,
    category: str | None = None,
    expense_date: str | None = None,
    currency: str | None = None,
    merchant_name: str | None = None,
    payment_method: str | None = None,
    notes: str | None = None,
) -> dict:
    """Update an existing expense.

    Updates only the fields that are provided. All fields are optional.

    Args:
        expense_id: The unique identifier of the expense to update
        amount: New expense amount
        description: New description
        category: New category name
        expense_date: New date (YYYY-MM-DD)
        currency: New currency code
        merchant_name: New merchant name
        payment_method: New payment method
        notes: New notes

    Returns:
        The updated expense or an error if not found.
    """
    # Parse date if provided
    parsed_date = None
    if expense_date:
        try:
            parsed_date = date.fromisoformat(expense_date)
        except ValueError:
            return {
                "error": f"Invalid expense_date format: {expense_date}. Use YYYY-MM-DD."
            }

    # Validate using Pydantic model with proper decimal precision
    try:
        update_data = ExpenseUpdate(
            amount=_to_decimal(amount) if amount is not None else None,
            description=description,
            category_name=category,
            expense_date=parsed_date,
            currency=currency,
            merchant_name=merchant_name,
            payment_method=payment_method,
            notes=notes,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    # Build updates dict from non-None values
    updates = update_data.model_dump(exclude_none=True)

    if not updates:
        return {"error": "No fields provided for update"}

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)

        expense = await repo.update(expense_id, **updates)

        if expense is None:
            return {"error": f"Expense not found: {expense_id}"}

        return ExpenseResponse.model_validate(expense).model_dump(mode="json")


@mcp.tool()
async def delete_expense(
    ctx: Context,
    expense_id: str,
) -> dict:
    """Delete an expense by its ID.

    Permanently removes the expense record. This action cannot be undone.

    Args:
        expense_id: The unique identifier of the expense to delete

    Returns:
        Success status or an error if the expense is not found.
    """
    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)
        deleted = await repo.delete(expense_id)

        if not deleted:
            return {"error": f"Expense not found: {expense_id}"}

        return {"success": True, "message": f"Expense {expense_id} deleted successfully"}


# =============================================================================
# Category Management Tools
# =============================================================================


@mcp.tool()
async def list_categories(
    ctx: Context,
    include_predefined: bool = True,
) -> dict:
    """List all available expense categories.

    Args:
        include_predefined: Include predefined categories (default: True)

    Returns:
        Both predefined categories (always available) and user's custom categories.
    """
    categories = []

    # Add predefined categories if requested
    if include_predefined:
        for cat in PredefinedCategory:
            categories.append(
                CategoryResponse(
                    id=None,
                    name=cat.value,
                    description=f"Predefined category: {cat.value}",
                    is_predefined=True,
                    is_active=True,
                ).model_dump(mode="json")
            )

    # Add user's custom categories
    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = CategoryRepository(session, expense_ctx.user_id)
        custom_categories = await repo.list_active()

        for cat in custom_categories:
            categories.append(
                CategoryResponse(
                    id=cat.id,
                    name=cat.name,
                    description=cat.description,
                    is_predefined=False,
                    is_active=cat.is_active,
                ).model_dump(mode="json")
            )

        return {
            "categories": categories,
            "count": len(categories),
            "predefined_count": len(PredefinedCategory) if include_predefined else 0,
            "custom_count": len(custom_categories),
        }


@mcp.tool()
async def create_category(
    ctx: Context,
    name: str,
    description: str | None = None,
) -> dict:
    """Create a new custom expense category.

    Args:
        name: Name for the new category
        description: Optional description for the category

    Returns:
        The created category. Note: Predefined category names cannot be used.
    """
    # Validate using Pydantic model
    try:
        category_data = CategoryCreate(
            name=name,
            description=description,
        )
    except ValueError as e:
        return {"error": f"Validation error: {str(e)}"}

    # Check if name conflicts with predefined categories
    if PredefinedCategory.is_predefined(category_data.name):
        return {
            "error": f"Cannot create category '{category_data.name}': "
            "this is a predefined category name"
        }

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = CategoryRepository(session, expense_ctx.user_id)

        # Check if custom category already exists
        existing = await repo.get_by_name(category_data.name)
        if existing is not None:
            if existing.is_active:
                return {"error": f"Category '{category_data.name}' already exists"}
            return {
                "error": f"Category '{category_data.name}' was previously deleted. "
                "Contact support to reactivate it."
            }

        category = await repo.create(
            name=category_data.name,
            description=category_data.description,
        )

        return CategoryResponse(
            id=category.id,
            name=category.name,
            description=category.description,
            is_predefined=False,
            is_active=category.is_active,
        ).model_dump(mode="json")


@mcp.tool()
async def delete_category(
    ctx: Context,
    name: str,
) -> dict:
    """Delete a custom expense category.

    Soft-deletes the category (marks as inactive). Existing expenses
    with this category will retain their category assignment.

    Args:
        name: Name of the category to delete

    Returns:
        Success status. Note: Predefined categories cannot be deleted.
    """
    # Check if attempting to delete a predefined category
    if PredefinedCategory.is_predefined(name):
        return {
            "error": f"Cannot delete '{name}': predefined categories cannot be deleted"
        }

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = CategoryRepository(session, expense_ctx.user_id)
        deleted = await repo.delete(name)

        if not deleted:
            return {"error": f"Category not found: {name}"}

        return {
            "success": True,
            "message": f"Category '{name}' deleted successfully",
        }


# =============================================================================
# Analytics Tools
# =============================================================================


@mcp.tool()
async def get_expense_summary(
    ctx: Context,
    start_date: str | None = None,
    end_date: str | None = None,
    group_by: str = "none",
) -> dict:
    """Get expense summary statistics.

    Args:
        start_date: Start date for summary (YYYY-MM-DD)
        end_date: End date for summary (YYYY-MM-DD)
        group_by: Group results by: 'category' or 'none' (default: none)

    Returns:
        Total spending, expense count, and average expense amount.
        Without date filters, returns summary for all time.
    """
    # Parse dates if provided
    parsed_start = None
    parsed_end = None

    if start_date:
        try:
            parsed_start = date.fromisoformat(start_date)
        except ValueError:
            return {
                "error": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD."
            }

    if end_date:
        try:
            parsed_end = date.fromisoformat(end_date)
        except ValueError:
            return {"error": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD."}

    # Validate group_by
    valid_group_by = ["none", "category"]
    if group_by not in valid_group_by:
        return {"error": f"group_by must be one of: {', '.join(valid_group_by)}"}

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)

        if group_by == "category":
            # Get grouped summary
            by_category = await repo.get_by_category(
                start_date=parsed_start,
                end_date=parsed_end,
            )

            # Calculate total for percentage
            total_amount = sum(cat["total_amount"] for cat in by_category)
            total_count = sum(cat["expense_count"] for cat in by_category)

            summaries = []
            for cat in by_category:
                percentage = (
                    (cat["total_amount"] / total_amount * 100)
                    if total_amount > 0
                    else Decimal("0")
                )
                summaries.append(
                    CategorySummary(
                        category_name=cat["category_name"],
                        total_amount=cat["total_amount"],
                        expense_count=cat["expense_count"],
                        percentage=round(percentage, 2),
                    ).model_dump(mode="json")
                )

            return GroupedExpenseSummary(
                summaries=summaries,
                total_amount=total_amount,
                total_count=total_count,
                group_by="category",
            ).model_dump(mode="json")

        # Get simple summary
        summary = await repo.get_summary(
            start_date=parsed_start,
            end_date=parsed_end,
        )

        return ExpenseSummary(
            total_amount=summary["total_amount"],
            expense_count=summary["expense_count"],
            average_amount=round(summary["average_amount"], 2),
            currency=settings.default_currency,
            start_date=parsed_start,
            end_date=parsed_end,
        ).model_dump(mode="json")


@mcp.tool()
async def get_monthly_trend(
    ctx: Context,
    months: int = 6,
) -> dict:
    """Get monthly spending trend.

    Args:
        months: Number of months to include (default: 6)

    Returns:
        Total spending and expense count for each month.
        Useful for understanding spending patterns over time.
    """
    if months < 1 or months > 24:
        return {"error": "months must be between 1 and 24"}

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)
        trend_data = await repo.get_monthly_trend(months=months)

        trends = [
            MonthlyTrend(
                year=item["year"],
                month=item["month"],
                total_amount=item["total_amount"],
                expense_count=item["expense_count"],
            ).model_dump(mode="json")
            for item in trend_data
        ]

        return MonthlyTrendResponse(
            trends=trends,
            currency=settings.default_currency,
        ).model_dump(mode="json")


@mcp.tool()
async def get_top_categories(
    ctx: Context,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5,
) -> dict:
    """Get top spending categories.

    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        limit: Maximum categories to return (default: 5)

    Returns:
        Categories with the highest total spending, ordered by amount.
        Without date filters, analyzes all time spending.
    """
    # Parse dates if provided
    parsed_start = None
    parsed_end = None

    if start_date:
        try:
            parsed_start = date.fromisoformat(start_date)
        except ValueError:
            return {
                "error": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD."
            }

    if end_date:
        try:
            parsed_end = date.fromisoformat(end_date)
        except ValueError:
            return {"error": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD."}

    if limit < 1 or limit > 20:
        return {"error": "limit must be between 1 and 20"}

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        repo = ExpenseRepository(session, expense_ctx.user_id)

        by_category = await repo.get_by_category(
            start_date=parsed_start,
            end_date=parsed_end,
        )

        # Calculate total for percentage
        total_amount = sum(cat["total_amount"] for cat in by_category)

        # Get top N categories
        top_cats = by_category[:limit]

        categories = []
        for cat in top_cats:
            percentage = (
                (cat["total_amount"] / total_amount * 100)
                if total_amount > 0
                else Decimal("0")
            )
            categories.append(
                TopCategory(
                    category_name=cat["category_name"],
                    total_amount=cat["total_amount"],
                    expense_count=cat["expense_count"],
                    percentage=round(percentage, 2),
                ).model_dump(mode="json")
            )

        return TopCategoriesResponse(
            categories=categories,
            total_amount=total_amount,
            currency=settings.default_currency,
            start_date=parsed_start,
            end_date=parsed_end,
        ).model_dump(mode="json")


if __name__ == "__main__":
    mcp.run()
