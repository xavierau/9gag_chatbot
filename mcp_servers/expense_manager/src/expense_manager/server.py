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

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from typing import Literal
import dspy

from fastmcp import Context, FastMCP
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from expense_manager.config import settings
from expense_manager.constants import PredefinedCategory

# Configure logging to stderr (MCP protocol requires stdout for JSONRPC messages only)
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # CRITICAL: Use stderr for MCP servers
)
logger = logging.getLogger(__name__)

# Type alias for category - uses Literal for schema generation
# Note: Custom categories are validated at runtime, not in type system
CategoryType = Literal[
    "Food & Dining",
    "Transportation",
    "Utilities",
    "Entertainment",
    "Shopping",
    "Healthcare",
    "Housing & Rent",
    "Education",
    "Travel",
    "Subscriptions",
    "Other",
]
from expense_manager.context import ExpenseContext
from expense_manager.models import (
    CategoryResponse,
    ExpenseCreate,
    ExpenseResponse,
    ExpenseUpdate,
)
from expense_manager.repository import CategoryRepository, ExpenseRepository


lm = dspy.LM("gemini/gemini-2.5-flash", api_key=settings.google_api_key)
dspy.settings.configure(lm=lm)

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
    category: CategoryType | str,
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
        category: Category name (predefined or custom). Predefined categories:
            Food & Dining, Transportation, Utilities, Entertainment, Shopping,
            Healthcare, Housing & Rent, Education, Travel, Subscriptions, Other.
            Custom categories can be created via create_category tool.
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
        # Validate category exists (predefined or custom)
        category_repo = CategoryRepository(session, expense_ctx.user_id)
        if not await category_repo.exists(expense_data.category_name):
            predefined = PredefinedCategory.values()
            custom_categories = await category_repo.list_active()
            custom_names = [c.name for c in custom_categories]
            all_categories = predefined + custom_names
            return {
                "error": f"Invalid category: '{expense_data.category_name}'. "
                f"Must be one of: {all_categories}"
            }

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

    Uses AI-powered SQL generation to build and execute queries dynamically.

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
    from expense_manager.text_to_sql import TextToSQLOrchestrator

    logger.info("[list_expenses] Tool called")
    logger.debug(f"[list_expenses] Parameters: start_date={start_date}, end_date={end_date}, "
                 f"category={category}, min_amount={min_amount}, max_amount={max_amount}, "
                 f"limit={limit}, offset={offset}")

    # Build natural language query from parameters
    query_parts = ["List all expenses"]
    filters_desc = []

    if start_date:
        filters_desc.append(f"from {start_date}")
    if end_date:
        filters_desc.append(f"until {end_date}")
    if category:
        filters_desc.append(f"in category '{category}'")
    if min_amount is not None:
        filters_desc.append(f"with amount >= {min_amount}")
    if max_amount is not None:
        filters_desc.append(f"with amount <= {max_amount}")

    if filters_desc:
        query_parts.append(" ".join(filters_desc))

    query_parts.append(f"limit {limit} offset {offset}")
    query_parts.append("ordered by date descending")

    natural_query = " ".join(query_parts)
    logger.info(f"[list_expenses] Natural language query: {natural_query}")

    expense_ctx = _get_context(ctx)
    logger.info(f"[list_expenses] User ID from context: {expense_ctx.user_id}")

    try:
        async with expense_ctx.get_session() as session:
            logger.debug("[list_expenses] Database session created")
            orchestrator = TextToSQLOrchestrator(
                session=session,
                user_id=expense_ctx.user_id,
                max_retries=3,
            )
            logger.debug("[list_expenses] Orchestrator created, executing...")

            lm = dspy.LM("gemini/gemini-2.5-flash", api_key=settings.google_api_key)
            with dspy.settings.context(lm=lm):
                result = await orchestrator.execute(question=natural_query)

            logger.info(f"[list_expenses] Orchestrator result: status={result.status.value}, "
                       f"row_count={result.row_count}, error={result.error}")

            if result.status.value == "success":
                logger.info(f"[list_expenses] SUCCESS - returning {result.row_count} expenses")
                return {
                    "expenses": result.data,
                    "count": result.row_count,
                    "limit": limit,
                    "offset": offset,
                    "sql_query": result.sql_query,
                }
            elif result.status.value == "no_results":
                logger.info("[list_expenses] NO_RESULTS - returning empty list")
                return {
                    "expenses": [],
                    "count": 0,
                    "limit": limit,
                    "offset": offset,
                    "sql_query": result.sql_query,
                }
            else:
                logger.error(f"[list_expenses] ERROR - {result.error}")
                return {"error": result.error or "Query failed"}
    except Exception as e:
        logger.error(f"[list_expenses] Exception: {e}", exc_info=True)
        return {"error": f"Internal error: {str(e)}"}


@mcp.tool()
async def search_expenses(
    ctx: Context,
    query: str,
    start_date: str | None = None,
    end_date: str | None = None,
    category: CategoryType | str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Search expenses by text in description, merchant name, or notes.

    Uses AI-powered SQL generation for flexible text search across fields.

    Args:
        query: Search text (searches in description, merchant_name, notes)
        start_date: Filter from date (YYYY-MM-DD)
        end_date: Filter until date (YYYY-MM-DD)
        category: Filter by category name
        limit: Maximum results to return (default: 50)
        offset: Number of results to skip (default: 0)

    Returns:
        A list of expenses matching the search query, ordered by date (most recent first).

    Examples:
        - search_expenses(query="lunch") - find all expenses mentioning "lunch"
        - search_expenses(query="大快活") - find expenses at Cafe de Coral
        - search_expenses(query="coffee", category="Food & Dining") - coffee expenses in Food category
    """
    from expense_manager.text_to_sql import TextToSQLOrchestrator

    if not query or not query.strip():
        return {"error": "Search query cannot be empty"}

    # Build natural language query
    query_parts = [f"Search expenses containing '{query}' in description, merchant name, or notes"]
    filters_desc = []

    if start_date:
        filters_desc.append(f"from {start_date}")
    if end_date:
        filters_desc.append(f"until {end_date}")
    if category:
        filters_desc.append(f"in category '{category}'")

    if filters_desc:
        query_parts.append(" ".join(filters_desc))

    query_parts.append(f"limit {limit} offset {offset}")
    query_parts.append("ordered by date descending")

    natural_query = " ".join(query_parts)

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        orchestrator = TextToSQLOrchestrator(
            session=session,
            user_id=expense_ctx.user_id,
            max_retries=3,
        )

        lm = dspy.LM("gemini/gemini-2.5-flash", api_key=settings.google_api_key)
        with dspy.settings.context(lm=lm):
            result = await orchestrator.execute(question=natural_query)

        if result.status.value == "success":
            return {
                "expenses": result.data,
                "count": result.row_count,
                "query": query,
                "limit": limit,
                "offset": offset,
                "sql_query": result.sql_query,
            }
        elif result.status.value == "no_results":
            return {
                "expenses": [],
                "count": 0,
                "query": query,
                "limit": limit,
                "offset": offset,
                "sql_query": result.sql_query,
            }
        else:
            return {"error": result.error or "Search failed"}


@mcp.tool()
async def update_expense(
    ctx: Context,
    expense_id: str,
    amount: float | None = None,
    description: str | None = None,
    category: CategoryType | str | None = None,
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
        category: New category name (predefined or custom). Predefined categories:
            Food & Dining, Transportation, Utilities, Entertainment, Shopping,
            Healthcare, Housing & Rent, Education, Travel, Subscriptions, Other.
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
        # Validate category if being updated
        if category is not None:
            category_repo = CategoryRepository(session, expense_ctx.user_id)
            if not await category_repo.exists(category):
                predefined = PredefinedCategory.values()
                custom_categories = await category_repo.list_active()
                custom_names = [c.name for c in custom_categories]
                all_categories = predefined + custom_names
                return {
                    "error": f"Invalid category: '{category}'. "
                    f"Must be one of: {all_categories}"
                }

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

#
# @mcp.tool()
# async def create_category(
#     ctx: Context,
#     name: str,
#     description: str | None = None,
# ) -> dict:
#     """Create a new custom expense category.
#
#     Args:
#         name: Name for the new category
#         description: Optional description for the category
#
#     Returns:
#         The created category. Note: Predefined category names cannot be used.
#     """
#     # Validate using Pydantic model
#     try:
#         category_data = CategoryCreate(
#             name=name,
#             description=description,
#         )
#     except ValueError as e:
#         return {"error": f"Validation error: {str(e)}"}
#
#     # Check if name conflicts with predefined categories
#     if PredefinedCategory.is_predefined(category_data.name):
#         return {
#             "error": f"Cannot create category '{category_data.name}': "
#             "this is a predefined category name"
#         }
#
#     expense_ctx = _get_context(ctx)
#
#     async with expense_ctx.get_session() as session:
#         repo = CategoryRepository(session, expense_ctx.user_id)
#
#         # Check if custom category already exists
#         existing = await repo.get_by_name(category_data.name)
#         if existing is not None:
#             if existing.is_active:
#                 return {"error": f"Category '{category_data.name}' already exists"}
#             return {
#                 "error": f"Category '{category_data.name}' was previously deleted. "
#                 "Contact support to reactivate it."
#             }
#
#         category = await repo.create(
#             name=category_data.name,
#             description=category_data.description,
#         )
#
#         return CategoryResponse(
#             id=category.id,
#             name=category.name,
#             description=category.description,
#             is_predefined=False,
#             is_active=category.is_active,
#         ).model_dump(mode="json")
#
#
# @mcp.tool()
# async def delete_category(
#     ctx: Context,
#     name: str,
# ) -> dict:
#     """Delete a custom expense category.
#
#     Soft-deletes the category (marks as inactive). Existing expenses
#     with this category will retain their category assignment.
#
#     Args:
#         name: Name of the category to delete
#
#     Returns:
#         Success status. Note: Predefined categories cannot be deleted.
#     """
#     # Check if attempting to delete a predefined category
#     if PredefinedCategory.is_predefined(name):
#         return {
#             "error": f"Cannot delete '{name}': predefined categories cannot be deleted"
#         }
#
#     expense_ctx = _get_context(ctx)
#
#     async with expense_ctx.get_session() as session:
#         repo = CategoryRepository(session, expense_ctx.user_id)
#         deleted = await repo.delete(name)
#
#         if not deleted:
#             return {"error": f"Category not found: {name}"}
#
#         return {
#             "success": True,
#             "message": f"Category '{name}' deleted successfully",
#         }


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

    Uses AI-powered SQL generation for flexible aggregation queries.

    Args:
        start_date: Start date for summary (YYYY-MM-DD)
        end_date: End date for summary (YYYY-MM-DD)
        group_by: Group results by: 'category' or 'none' (default: none)

    Returns:
        Total spending, expense count, and average expense amount.
        Without date filters, returns summary for all time.
    """
    from expense_manager.text_to_sql import TextToSQLOrchestrator

    # Validate group_by
    valid_group_by = ["none", "category"]
    if group_by not in valid_group_by:
        return {"error": f"group_by must be one of: {', '.join(valid_group_by)}"}

    # Build natural language query
    if group_by == "category":
        query_parts = ["Get total amount, count, and average amount of expenses grouped by category"]
    else:
        query_parts = ["Get total amount, count, and average amount of all expenses"]

    filters_desc = []
    if start_date:
        filters_desc.append(f"from {start_date}")
    if end_date:
        filters_desc.append(f"until {end_date}")

    if filters_desc:
        query_parts.append(" ".join(filters_desc))

    if group_by == "category":
        query_parts.append("ordered by total amount descending")

    natural_query = " ".join(query_parts)

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        orchestrator = TextToSQLOrchestrator(
            session=session,
            user_id=expense_ctx.user_id,
            max_retries=3,
        )

        result = await orchestrator.execute(question=natural_query)

        if result.status.value == "success":
            if group_by == "category":
                # Process grouped results
                summaries = []
                total_amount = Decimal("0")
                total_count = 0

                for row in result.data:
                    # Handle different possible column names from LLM
                    cat_name = row.get("category_name") or row.get("category") or "Unknown"
                    amt = Decimal(str(row.get("total_amount") or row.get("total") or row.get("sum") or 0))
                    cnt = int(row.get("expense_count") or row.get("count") or 0)
                    total_amount += amt
                    total_count += cnt
                    summaries.append({
                        "category_name": cat_name,
                        "total_amount": float(amt),
                        "expense_count": cnt,
                    })

                # Calculate percentages
                for s in summaries:
                    s["percentage"] = round(
                        (Decimal(str(s["total_amount"])) / total_amount * 100)
                        if total_amount > 0
                        else Decimal("0"),
                        2,
                    )

                return {
                    "summaries": summaries,
                    "total_amount": float(total_amount),
                    "total_count": total_count,
                    "group_by": "category",
                    "sql_query": result.sql_query,
                }
            else:
                # Process simple summary
                if result.data:
                    row = result.data[0]
                    total = Decimal(str(row.get("total_amount") or row.get("total") or row.get("sum") or 0))
                    count = int(row.get("expense_count") or row.get("count") or 0)
                    avg = Decimal(str(row.get("average_amount") or row.get("avg") or row.get("average") or 0))
                else:
                    total = Decimal("0")
                    count = 0
                    avg = Decimal("0")

                return {
                    "total_amount": float(total),
                    "expense_count": count,
                    "average_amount": float(round(avg, 2)),
                    "currency": settings.default_currency,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sql_query": result.sql_query,
                }

        elif result.status.value == "no_results":
            if group_by == "category":
                return {
                    "summaries": [],
                    "total_amount": 0.0,
                    "total_count": 0,
                    "group_by": "category",
                    "sql_query": result.sql_query,
                }
            else:
                return {
                    "total_amount": 0.0,
                    "expense_count": 0,
                    "average_amount": 0.0,
                    "currency": settings.default_currency,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sql_query": result.sql_query,
                }
        else:
            return {"error": result.error or "Summary query failed"}


@mcp.tool()
async def get_monthly_trend(
    ctx: Context,
    months: int = 6,
) -> dict:
    """Get monthly spending trend.

    Uses AI-powered SQL generation for flexible trend analysis.

    Args:
        months: Number of months to include (default: 6)

    Returns:
        Total spending and expense count for each month.
        Useful for understanding spending patterns over time.
    """
    from expense_manager.text_to_sql import TextToSQLOrchestrator

    if months < 1 or months > 24:
        return {"error": "months must be between 1 and 24"}

    # Build natural language query
    natural_query = (
        f"Get monthly spending totals for the last {months} months "
        "showing year, month, total amount, and expense count "
        "grouped by year and month, ordered by year descending then month descending"
    )

    expense_ctx = _get_context(ctx)

    async with expense_ctx.get_session() as session:
        orchestrator = TextToSQLOrchestrator(
            session=session,
            user_id=expense_ctx.user_id,
            max_retries=3,
        )

        result = await orchestrator.execute(question=natural_query)

        if result.status.value == "success":
            trends = []
            for row in result.data:
                year = int(row.get("year") or 0)
                month = int(row.get("month") or 0)
                total = float(row.get("total_amount") or row.get("total") or row.get("sum") or 0)
                count = int(row.get("expense_count") or row.get("count") or 0)
                trends.append({
                    "year": year,
                    "month": month,
                    "total_amount": total,
                    "expense_count": count,
                })

            return {
                "trends": trends,
                "currency": settings.default_currency,
                "sql_query": result.sql_query,
            }

        elif result.status.value == "no_results":
            return {
                "trends": [],
                "currency": settings.default_currency,
                "sql_query": result.sql_query,
            }
        else:
            return {"error": result.error or "Monthly trend query failed"}


@mcp.tool()
async def get_top_categories(
    ctx: Context,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 5,
) -> dict:
    """Get top spending categories.

    Uses AI-powered SQL generation for flexible category ranking.

    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        limit: Maximum categories to return (default: 5)

    Returns:
        Categories with the highest total spending, ordered by amount.
        Without date filters, analyzes all time spending.
    """
    from expense_manager.text_to_sql import TextToSQLOrchestrator

    if limit < 1 or limit > 20:
        return {"error": "limit must be between 1 and 20"}

    # Build natural language query
    query_parts = [f"Get top {limit} categories by total spending amount"]
    query_parts.append("showing category name, total amount, and expense count")

    filters_desc = []
    if start_date:
        filters_desc.append(f"from {start_date}")
    if end_date:
        filters_desc.append(f"until {end_date}")

    if filters_desc:
        query_parts.append(" ".join(filters_desc))

    query_parts.append("ordered by total amount descending")

    natural_query = " ".join(query_parts)

    expense_ctx = _get_context(ctx)

    lm = dspy.LM("gemini/gemini-2.5-flash-lite", api_key=settings.google_api_key)
    with dspy.settings.context(lm=lm):
        async with expense_ctx.get_session() as session:
            orchestrator = TextToSQLOrchestrator(
                    session=session,
                    user_id=expense_ctx.user_id,
                    max_retries=3,
                )
            result = await orchestrator.execute(question=natural_query)


        if result.status.value == "success":
            # Process results and calculate percentages
            categories = []
            total_amount = Decimal("0")

            # First pass: calculate total
            for row in result.data:
                amt = Decimal(str(row.get("total_amount") or row.get("total") or row.get("sum") or 0))
                total_amount += amt

            # Second pass: build categories with percentages
            for row in result.data:
                cat_name = row.get("category_name") or row.get("category") or "Unknown"
                amt = Decimal(str(row.get("total_amount") or row.get("total") or row.get("sum") or 0))
                cnt = int(row.get("expense_count") or row.get("count") or 0)
                percentage = (
                    round((amt / total_amount * 100), 2)
                    if total_amount > 0
                    else Decimal("0")
                )
                categories.append({
                    "category_name": cat_name,
                    "total_amount": float(amt),
                    "expense_count": cnt,
                    "percentage": float(percentage),
                })

            return {
                "categories": categories,
                "total_amount": float(total_amount),
                "currency": settings.default_currency,
                "start_date": start_date,
                "end_date": end_date,
                "sql_query": result.sql_query,
            }

        elif result.status.value == "no_results":
            return {
                "categories": [],
                "total_amount": 0.0,
                "currency": settings.default_currency,
                "start_date": start_date,
                "end_date": end_date,
                "sql_query": result.sql_query,
            }
        else:
            return {"error": result.error or "Top categories query failed"}


# =============================================================================
# Natural Language Query Tool (Text-to-SQL)
# =============================================================================

#
# @mcp.tool()
# async def query_expenses_natural_language(
#     ctx: Context,
#     question: str,
#     additional_context: str | None = None,
# ) -> dict:
#     """Query expenses using natural language.
#
#     Converts a natural language question into SQL and executes it against
#     the expense database. Supports complex queries like aggregations,
#     filtering, and comparisons.
#
#     Args:
#         question: Natural language question about expenses.
#             Examples:
#             - "How much did I spend on food last month?"
#             - "What are my top 5 expenses this year?"
#             - "Show me all transportation expenses over $100"
#             - "Compare my spending by category for January vs February"
#         additional_context: Optional clarification or additional context
#             for the query (use when follow-up is needed)
#
#     Returns:
#         Query results with data, SQL used, and explanation.
#         May return a clarification request if the question is ambiguous.
#
#     Note:
#         This tool uses AI to generate SQL, so complex or unusual queries
#         may require rephrasing. The SQL is always filtered by user_id
#         for security.
#     """
#     from expense_manager.text_to_sql import TextToSQLOrchestrator
#
#     if not question or not question.strip():
#         return {"error": "Question cannot be empty"}
#
#     expense_ctx = _get_context(ctx)
#
#     async with expense_ctx.get_session() as session:
#
#         orchestrator = TextToSQLOrchestrator(
#             session=session,
#             user_id=expense_ctx.user_id,
#             max_retries=3,
#             check_ambiguity=False,  # Can enable for more careful queries
#         )
#
#         lm = dspy.LM("gemini/gemini-2.5-flash", api_key=settings.google_api_key)
#         with dspy.settings.context(lm=lm):
#             result = await orchestrator.execute(
#                 question=question.strip(),
#                 additional_context=additional_context,
#             )
#
#         # Build response based on pipeline result
#         response: dict = {
#             "status": result.status.value,
#         }
#
#         if result.status.value == "success":
#             response["data"] = result.data
#             response["row_count"] = result.row_count
#             response["sql_query"] = result.sql_query
#             response["explanation"] = result.explanation
#
#         elif result.status.value == "needs_clarification":
#             response["needs_clarification"] = True
#             response["clarification_question"] = result.clarification_question
#             if result.explanation:
#                 response["context"] = result.explanation
#
#         elif result.status.value == "no_results":
#             response["message"] = "No expenses found matching your query"
#             response["sql_query"] = result.sql_query
#             response["suggestion"] = "Try broadening your search criteria or checking if you have expenses in the specified date range"
#
#         else:  # error
#             response["error"] = result.error
#             if result.sql_query:
#                 response["sql_query"] = result.sql_query
#
#         return response


if __name__ == "__main__":
    mcp.run()
