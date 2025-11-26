"""Pydantic models for expense management.

These models define the request/response schemas for the MCP tools.
They are separate from the SQLAlchemy ORM models which live in the
main application's database module.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from expense_manager.config import settings
from expense_manager.constants import SUPPORTED_CURRENCIES


# =============================================================================
# Expense Models
# =============================================================================


class ExpenseCreate(BaseModel):
    """Schema for creating a new expense."""

    amount: Decimal = Field(..., gt=0, decimal_places=2, description="Expense amount")
    description: str = Field(..., min_length=1, max_length=500)
    category_name: str = Field(..., min_length=1, max_length=100)
    expense_date: date = Field(..., description="Date of the expense")
    currency: str = Field(default=settings.default_currency, max_length=3)
    merchant_name: str | None = Field(default=None, max_length=255)
    payment_method: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        v = v.upper()
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of: {', '.join(SUPPORTED_CURRENCIES)}"
            )
        return v


class ExpenseUpdate(BaseModel):
    """Schema for updating an existing expense.

    All fields are optional - only provided fields will be updated.
    """

    amount: Decimal | None = Field(default=None, gt=0, decimal_places=2)
    description: str | None = Field(default=None, min_length=1, max_length=500)
    category_name: str | None = Field(default=None, min_length=1, max_length=100)
    expense_date: date | None = Field(default=None)
    currency: str | None = Field(default=None, max_length=3)
    merchant_name: str | None = Field(default=None, max_length=255)
    payment_method: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=1000)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str | None) -> str | None:
        """Validate currency code if provided."""
        if v is None:
            return v
        v = v.upper()
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Currency must be one of: {', '.join(SUPPORTED_CURRENCIES)}"
            )
        return v


class ExpenseResponse(BaseModel):
    """Schema for expense response."""

    id: str
    amount: Decimal
    currency: str
    description: str
    category_name: str
    expense_date: date
    merchant_name: str | None
    payment_method: str | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExpenseListFilter(BaseModel):
    """Filters for listing expenses."""

    start_date: date | None = Field(default=None, description="Filter from this date")
    end_date: date | None = Field(default=None, description="Filter until this date")
    category: str | None = Field(default=None, description="Filter by category name")
    min_amount: Decimal | None = Field(default=None, ge=0)
    max_amount: Decimal | None = Field(default=None, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)

    @field_validator("end_date")
    @classmethod
    def end_date_after_start(cls, v: date | None, info) -> date | None:
        """Ensure end_date is after start_date if both provided."""
        start = info.data.get("start_date")
        if v is not None and start is not None and v < start:
            raise ValueError("end_date must be after or equal to start_date")
        return v


# =============================================================================
# Category Models
# =============================================================================


class CategoryCreate(BaseModel):
    """Schema for creating a custom category."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)


class CategoryResponse(BaseModel):
    """Schema for category response."""

    id: str | None = Field(
        default=None, description="ID (None for predefined categories)"
    )
    name: str
    description: str | None
    is_predefined: bool = Field(description="True if this is a built-in category")
    is_active: bool = Field(default=True)

    model_config = {"from_attributes": True}


# =============================================================================
# Analytics Models
# =============================================================================


class ExpenseSummary(BaseModel):
    """Summary of expenses for a time period."""

    total_amount: Decimal
    expense_count: int
    average_amount: Decimal
    currency: str
    start_date: date | None
    end_date: date | None


class CategorySummary(BaseModel):
    """Expense summary for a single category."""

    category_name: str
    total_amount: Decimal
    expense_count: int
    percentage: Decimal = Field(
        description="Percentage of total expenses in this category"
    )


class GroupedExpenseSummary(BaseModel):
    """Expense summary grouped by category or date."""

    summaries: list[CategorySummary]
    total_amount: Decimal
    total_count: int
    group_by: Literal["category", "date", "month"]


class MonthlyTrend(BaseModel):
    """Monthly expense trend data point."""

    year: int
    month: int
    total_amount: Decimal
    expense_count: int


class MonthlyTrendResponse(BaseModel):
    """Response for monthly trend analysis."""

    trends: list[MonthlyTrend]
    currency: str


class TopCategory(BaseModel):
    """Top spending category."""

    category_name: str
    total_amount: Decimal
    expense_count: int
    percentage: Decimal


class TopCategoriesResponse(BaseModel):
    """Response for top categories analysis."""

    categories: list[TopCategory]
    total_amount: Decimal
    currency: str
    start_date: date | None
    end_date: date | None
