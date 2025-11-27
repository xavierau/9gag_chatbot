"""Repository layer for expense data access.

This module provides the data access layer for expenses and categories.
It uses SQLAlchemy for database operations and follows the Repository pattern.

All methods require a user_id parameter which is obtained from the
ExpenseContext (set via MCP_USER_ID environment variable).

ORM models are imported from the main app to maintain a single source of truth
and avoid DRY violations.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Import ORM models from the main app to avoid duplication
from app.infrastructure.database.models import ExpenseCategoryORM, ExpenseORM


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Repository Classes
# =============================================================================


class ExpenseRepository:
    """Repository for expense CRUD operations."""

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        """Initialize repository with database session and user context.

        Args:
            db: Async database session
            user_id: User identifier (from MCP_USER_ID env var)
        """
        self.db = db
        self.user_id = user_id

    async def create(
        self,
        amount: Decimal,
        description: str,
        category_name: str,
        expense_date: date,
        currency: str = "HKD",
        merchant_name: str | None = None,
        payment_method: str | None = None,
        notes: str | None = None,
    ) -> ExpenseORM:
        """Create a new expense.

        Args:
            amount: Expense amount (must be positive)
            description: Description of the expense
            category_name: Category for the expense
            expense_date: Date of the expense
            currency: Currency code (default: HKD)
            merchant_name: Optional merchant/vendor name
            payment_method: Optional payment method
            notes: Optional additional notes

        Returns:
            The created expense ORM object
        """
        expense = ExpenseORM(
            user_id=self.user_id,
            amount=amount,
            currency=currency,
            description=description,
            category_name=category_name,
            expense_date=expense_date,
            merchant_name=merchant_name,
            payment_method=payment_method,
            notes=notes,
        )
        self.db.add(expense)
        await self.db.flush()
        await self.db.refresh(expense)
        return expense

    async def get_by_id(self, expense_id: str) -> ExpenseORM | None:
        """Get an expense by ID.

        Args:
            expense_id: The expense ID

        Returns:
            The expense if found and belongs to user, None otherwise
        """
        result = await self.db.execute(
            select(ExpenseORM).where(
                and_(
                    ExpenseORM.id == expense_id,
                    ExpenseORM.user_id == self.user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        category: str | None = None,
        min_amount: Decimal | None = None,
        max_amount: Decimal | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExpenseORM]:
        """List expenses with optional filters.

        Args:
            start_date: Filter expenses on or after this date
            end_date: Filter expenses on or before this date
            category: Filter by category name
            min_amount: Filter expenses >= this amount
            max_amount: Filter expenses <= this amount
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching expenses
        """
        query = select(ExpenseORM).where(ExpenseORM.user_id == self.user_id)

        if start_date is not None:
            query = query.where(ExpenseORM.expense_date >= start_date)
        if end_date is not None:
            query = query.where(ExpenseORM.expense_date <= end_date)
        if category is not None:
            query = query.where(ExpenseORM.category_name == category)
        if min_amount is not None:
            query = query.where(ExpenseORM.amount >= min_amount)
        if max_amount is not None:
            query = query.where(ExpenseORM.amount <= max_amount)

        query = (
            query.order_by(ExpenseORM.expense_date.desc(), ExpenseORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def search(
        self,
        query_text: str,
        start_date: date | None = None,
        end_date: date | None = None,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ExpenseORM]:
        """Search expenses by text in description, merchant, and notes.

        Performs case-insensitive search across multiple text fields.

        Args:
            query_text: Text to search for (searches description, merchant_name, notes)
            start_date: Filter expenses on or after this date
            end_date: Filter expenses on or before this date
            category: Filter by category name
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching expenses ordered by date (most recent first)
        """
        # Build base query with user filter
        query = select(ExpenseORM).where(ExpenseORM.user_id == self.user_id)

        # Add text search across multiple fields (case-insensitive)
        search_pattern = f"%{query_text}%"

        query = query.where(
            or_(
                ExpenseORM.description.ilike(search_pattern),
                ExpenseORM.merchant_name.ilike(search_pattern),
                ExpenseORM.notes.ilike(search_pattern),
            )
        )

        # Apply additional filters
        if start_date is not None:
            query = query.where(ExpenseORM.expense_date >= start_date)
        if end_date is not None:
            query = query.where(ExpenseORM.expense_date <= end_date)
        if category is not None:
            query = query.where(ExpenseORM.category_name == category)

        # Order and paginate
        query = (
            query.order_by(ExpenseORM.expense_date.desc(), ExpenseORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update(
        self,
        expense_id: str,
        **updates: Any,
    ) -> ExpenseORM | None:
        """Update an expense.

        Args:
            expense_id: The expense ID
            **updates: Fields to update

        Returns:
            The updated expense if found, None otherwise
        """
        # Filter out None values to allow partial updates
        valid_updates = {k: v for k, v in updates.items() if v is not None}

        if not valid_updates:
            # No updates to make, just return the existing expense
            return await self.get_by_id(expense_id)

        # Add updated_at timestamp
        valid_updates["updated_at"] = utc_now()

        await self.db.execute(
            update(ExpenseORM)
            .where(
                and_(
                    ExpenseORM.id == expense_id,
                    ExpenseORM.user_id == self.user_id,
                )
            )
            .values(**valid_updates)
        )
        await self.db.flush()

        return await self.get_by_id(expense_id)

    async def delete(self, expense_id: str) -> bool:
        """Delete an expense.

        Args:
            expense_id: The expense ID

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            delete(ExpenseORM).where(
                and_(
                    ExpenseORM.id == expense_id,
                    ExpenseORM.user_id == self.user_id,
                )
            )
        )
        await self.db.flush()
        return result.rowcount > 0

    async def get_summary(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """Get expense summary statistics.

        Args:
            start_date: Filter expenses on or after this date
            end_date: Filter expenses on or before this date

        Returns:
            Dictionary with total_amount, expense_count, average_amount
        """
        query = select(
            func.coalesce(func.sum(ExpenseORM.amount), 0).label("total"),
            func.count(ExpenseORM.id).label("count"),
            func.coalesce(func.avg(ExpenseORM.amount), 0).label("average"),
        ).where(ExpenseORM.user_id == self.user_id)

        if start_date is not None:
            query = query.where(ExpenseORM.expense_date >= start_date)
        if end_date is not None:
            query = query.where(ExpenseORM.expense_date <= end_date)

        result = await self.db.execute(query)
        row = result.one()

        return {
            "total_amount": Decimal(str(row.total)),
            "expense_count": row.count,
            "average_amount": Decimal(str(row.average)),
        }

    async def get_by_category(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict[str, Any]]:
        """Get expenses grouped by category.

        Args:
            start_date: Filter expenses on or after this date
            end_date: Filter expenses on or before this date

        Returns:
            List of dictionaries with category_name, total_amount, expense_count
        """
        query = select(
            ExpenseORM.category_name,
            func.sum(ExpenseORM.amount).label("total"),
            func.count(ExpenseORM.id).label("count"),
        ).where(ExpenseORM.user_id == self.user_id)

        if start_date is not None:
            query = query.where(ExpenseORM.expense_date >= start_date)
        if end_date is not None:
            query = query.where(ExpenseORM.expense_date <= end_date)

        query = query.group_by(ExpenseORM.category_name).order_by(
            func.sum(ExpenseORM.amount).desc()
        )

        result = await self.db.execute(query)
        rows = result.all()

        return [
            {
                "category_name": row.category_name,
                "total_amount": Decimal(str(row.total)),
                "expense_count": row.count,
            }
            for row in rows
        ]

    async def get_monthly_trend(
        self,
        months: int = 6,
    ) -> list[dict[str, Any]]:
        """Get monthly expense trend.

        Args:
            months: Number of months to include

        Returns:
            List of dictionaries with year, month, total_amount, expense_count
        """
        # Calculate the start date for the trend
        today = date.today()
        start_year = today.year
        start_month = today.month - months + 1
        if start_month <= 0:
            start_year -= 1
            start_month += 12
        start_date = date(start_year, start_month, 1)

        query = select(
            func.extract("year", ExpenseORM.expense_date).label("year"),
            func.extract("month", ExpenseORM.expense_date).label("month"),
            func.sum(ExpenseORM.amount).label("total"),
            func.count(ExpenseORM.id).label("count"),
        ).where(
            and_(
                ExpenseORM.user_id == self.user_id,
                ExpenseORM.expense_date >= start_date,
            )
        )

        query = query.group_by(
            func.extract("year", ExpenseORM.expense_date),
            func.extract("month", ExpenseORM.expense_date),
        ).order_by(
            func.extract("year", ExpenseORM.expense_date),
            func.extract("month", ExpenseORM.expense_date),
        )

        result = await self.db.execute(query)
        rows = result.all()

        return [
            {
                "year": int(row.year),
                "month": int(row.month),
                "total_amount": Decimal(str(row.total)),
                "expense_count": row.count,
            }
            for row in rows
        ]


class CategoryRepository:
    """Repository for expense category operations."""

    def __init__(self, db: AsyncSession, user_id: str) -> None:
        """Initialize repository with database session and user context.

        Args:
            db: Async database session
            user_id: User identifier (from MCP_USER_ID env var)
        """
        self.db = db
        self.user_id = user_id

    async def create(
        self,
        name: str,
        description: str | None = None,
    ) -> ExpenseCategoryORM:
        """Create a new custom category.

        Args:
            name: Category name
            description: Optional description

        Returns:
            The created category ORM object
        """
        category = ExpenseCategoryORM(
            user_id=self.user_id,
            name=name,
            description=description,
        )
        self.db.add(category)
        await self.db.flush()
        await self.db.refresh(category)
        return category

    async def get_by_name(self, name: str) -> ExpenseCategoryORM | None:
        """Get a category by name.

        Args:
            name: Category name

        Returns:
            The category if found, None otherwise
        """
        result = await self.db.execute(
            select(ExpenseCategoryORM).where(
                and_(
                    ExpenseCategoryORM.name == name,
                    ExpenseCategoryORM.user_id == self.user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_active(self) -> list[ExpenseCategoryORM]:
        """List all active custom categories for the user.

        Returns:
            List of active custom categories
        """
        result = await self.db.execute(
            select(ExpenseCategoryORM)
            .where(
                and_(
                    ExpenseCategoryORM.user_id == self.user_id,
                    ExpenseCategoryORM.is_active == True,  # noqa: E712
                )
            )
            .order_by(ExpenseCategoryORM.name)
        )
        return list(result.scalars().all())

    async def delete(self, name: str) -> bool:
        """Delete a custom category (soft delete by setting is_active=False).

        Args:
            name: Category name

        Returns:
            True if deleted, False if not found
        """
        result = await self.db.execute(
            update(ExpenseCategoryORM)
            .where(
                and_(
                    ExpenseCategoryORM.name == name,
                    ExpenseCategoryORM.user_id == self.user_id,
                )
            )
            .values(is_active=False)
        )
        await self.db.flush()
        return result.rowcount > 0

    async def exists(self, name: str) -> bool:
        """Check if a category exists (either predefined or custom).

        Args:
            name: Category name

        Returns:
            True if category exists
        """
        # Check custom categories first
        custom = await self.get_by_name(name)
        if custom is not None and custom.is_active:
            return True

        # Check predefined categories
        from expense_manager.constants import PredefinedCategory

        return PredefinedCategory.is_predefined(name)
