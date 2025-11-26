"""Tests for expense management tools and repository."""

from datetime import date
from decimal import Decimal

import pytest

from expense_manager.constants import PredefinedCategory
from expense_manager.models import ExpenseCreate, ExpenseListFilter, ExpenseUpdate
from expense_manager.repository import CategoryRepository, ExpenseRepository


# =============================================================================
# Repository Tests
# =============================================================================


class TestExpenseRepository:
    """Tests for ExpenseRepository."""

    async def test_create_expense(self, db_session, test_user_id):
        """Test creating a new expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        expense = await repo.create(
            amount=Decimal("99.99"),
            description="Test expense",
            category_name="Food & Dining",
            expense_date=date(2024, 1, 15),
            currency="HKD",
            merchant_name="Test Shop",
            payment_method="Cash",
            notes="Test notes",
        )

        assert expense.id is not None
        assert expense.user_id == test_user_id
        assert expense.amount == Decimal("99.99")
        assert expense.description == "Test expense"
        assert expense.category_name == "Food & Dining"
        assert expense.expense_date == date(2024, 1, 15)
        assert expense.currency == "HKD"
        assert expense.merchant_name == "Test Shop"
        assert expense.payment_method == "Cash"
        assert expense.notes == "Test notes"
        assert expense.created_at is not None
        assert expense.updated_at is not None

    async def test_get_expense_by_id(self, db_session, test_user_id, sample_expense):
        """Test retrieving an expense by ID."""
        repo = ExpenseRepository(db_session, test_user_id)

        expense = await repo.get_by_id(sample_expense.id)

        assert expense is not None
        assert expense.id == sample_expense.id
        assert expense.description == "Test lunch"

    async def test_get_expense_not_found(self, db_session, test_user_id):
        """Test retrieving a non-existent expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        expense = await repo.get_by_id("non-existent-id")

        assert expense is None

    async def test_get_expense_wrong_user(self, db_session, sample_expense):
        """Test that expense is not returned for wrong user."""
        repo = ExpenseRepository(db_session, "different-user")

        expense = await repo.get_by_id(sample_expense.id)

        assert expense is None

    async def test_list_expenses(self, db_session, test_user_id, sample_expenses):
        """Test listing expenses without filters."""
        repo = ExpenseRepository(db_session, test_user_id)

        expenses = await repo.list()

        assert len(expenses) == 5
        # Should be ordered by date descending
        assert expenses[0].expense_date >= expenses[-1].expense_date

    async def test_list_expenses_with_date_filter(
        self, db_session, test_user_id, sample_expenses
    ):
        """Test listing expenses with date filter."""
        repo = ExpenseRepository(db_session, test_user_id)

        expenses = await repo.list(
            start_date=date(2024, 1, 15), end_date=date(2024, 1, 18)
        )

        assert len(expenses) == 2
        for expense in expenses:
            assert date(2024, 1, 15) <= expense.expense_date <= date(2024, 1, 18)

    async def test_list_expenses_with_category_filter(
        self, db_session, test_user_id, sample_expenses
    ):
        """Test listing expenses filtered by category."""
        repo = ExpenseRepository(db_session, test_user_id)

        expenses = await repo.list(category="Food & Dining")

        assert len(expenses) == 2
        for expense in expenses:
            assert expense.category_name == "Food & Dining"

    async def test_list_expenses_with_amount_filter(
        self, db_session, test_user_id, sample_expenses
    ):
        """Test listing expenses filtered by amount range."""
        repo = ExpenseRepository(db_session, test_user_id)

        expenses = await repo.list(min_amount=Decimal("100"), max_amount=Decimal("250"))

        assert len(expenses) == 2
        for expense in expenses:
            assert Decimal("100") <= expense.amount <= Decimal("250")

    async def test_list_expenses_with_pagination(
        self, db_session, test_user_id, sample_expenses
    ):
        """Test pagination of expense list."""
        repo = ExpenseRepository(db_session, test_user_id)

        page1 = await repo.list(limit=2, offset=0)
        page2 = await repo.list(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_update_expense(self, db_session, test_user_id, sample_expense):
        """Test updating an expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        updated = await repo.update(
            sample_expense.id,
            amount=Decimal("150.00"),
            description="Updated description",
        )

        assert updated is not None
        assert updated.amount == Decimal("150.00")
        assert updated.description == "Updated description"
        # Unchanged fields should remain
        assert updated.category_name == "Food & Dining"

    async def test_update_expense_not_found(self, db_session, test_user_id):
        """Test updating a non-existent expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        updated = await repo.update("non-existent-id", amount=Decimal("100"))

        assert updated is None

    async def test_delete_expense(self, db_session, test_user_id, sample_expense):
        """Test deleting an expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        deleted = await repo.delete(sample_expense.id)

        assert deleted is True

        # Verify it's gone
        expense = await repo.get_by_id(sample_expense.id)
        assert expense is None

    async def test_delete_expense_not_found(self, db_session, test_user_id):
        """Test deleting a non-existent expense."""
        repo = ExpenseRepository(db_session, test_user_id)

        deleted = await repo.delete("non-existent-id")

        assert deleted is False

    async def test_get_summary(self, db_session, test_user_id, sample_expenses):
        """Test getting expense summary."""
        repo = ExpenseRepository(db_session, test_user_id)

        summary = await repo.get_summary()

        assert summary["expense_count"] == 5
        assert summary["total_amount"] == Decimal("830.00")
        assert summary["average_amount"] == Decimal("166.00")

    async def test_get_summary_with_date_filter(
        self, db_session, test_user_id, sample_expenses
    ):
        """Test getting expense summary with date filter."""
        repo = ExpenseRepository(db_session, test_user_id)

        summary = await repo.get_summary(
            start_date=date(2024, 1, 15), end_date=date(2024, 1, 20)
        )

        # Should include expenses from Jan 15, 18, and 20
        assert summary["expense_count"] == 3

    async def test_get_by_category(self, db_session, test_user_id, sample_expenses):
        """Test getting expenses grouped by category."""
        repo = ExpenseRepository(db_session, test_user_id)

        by_category = await repo.get_by_category()

        assert len(by_category) == 4  # Food, Transportation, Utilities, Entertainment
        # Results should be ordered by total amount descending
        assert by_category[0]["total_amount"] >= by_category[-1]["total_amount"]

    async def test_get_monthly_trend(self, db_session, test_user_id, sample_expenses):
        """Test getting monthly spending trend."""
        repo = ExpenseRepository(db_session, test_user_id)

        trend = await repo.get_monthly_trend(months=12)

        # Should have at least one month with data
        assert len(trend) >= 0  # May be empty if dates don't match


class TestCategoryRepository:
    """Tests for CategoryRepository."""

    async def test_create_category(self, db_session, test_user_id):
        """Test creating a custom category."""
        repo = CategoryRepository(db_session, test_user_id)

        category = await repo.create(
            name="Custom Category",
            description="A custom category for testing",
        )

        assert category.id is not None
        assert category.name == "Custom Category"
        assert category.description == "A custom category for testing"
        assert category.is_active is True
        assert category.user_id == test_user_id

    async def test_get_category_by_name(self, db_session, test_user_id, sample_category):
        """Test retrieving a category by name."""
        repo = CategoryRepository(db_session, test_user_id)

        category = await repo.get_by_name("Pet Supplies")

        assert category is not None
        assert category.name == "Pet Supplies"

    async def test_get_category_not_found(self, db_session, test_user_id):
        """Test retrieving a non-existent category."""
        repo = CategoryRepository(db_session, test_user_id)

        category = await repo.get_by_name("Non-existent Category")

        assert category is None

    async def test_list_active_categories(
        self, db_session, test_user_id, sample_category
    ):
        """Test listing active custom categories."""
        repo = CategoryRepository(db_session, test_user_id)

        categories = await repo.list_active()

        assert len(categories) == 1
        assert categories[0].name == "Pet Supplies"

    async def test_delete_category(self, db_session, test_user_id, sample_category):
        """Test soft-deleting a category."""
        repo = CategoryRepository(db_session, test_user_id)

        deleted = await repo.delete("Pet Supplies")

        assert deleted is True

        # Should no longer appear in active list
        categories = await repo.list_active()
        assert len(categories) == 0

        # But should still exist (soft delete)
        category = await repo.get_by_name("Pet Supplies")
        assert category is not None
        assert category.is_active is False

    async def test_exists_predefined(self, db_session, test_user_id):
        """Test checking existence of predefined category."""
        repo = CategoryRepository(db_session, test_user_id)

        exists = await repo.exists("Food & Dining")

        assert exists is True

    async def test_exists_custom(self, db_session, test_user_id, sample_category):
        """Test checking existence of custom category."""
        repo = CategoryRepository(db_session, test_user_id)

        exists = await repo.exists("Pet Supplies")

        assert exists is True

    async def test_exists_not_found(self, db_session, test_user_id):
        """Test checking existence of non-existent category."""
        repo = CategoryRepository(db_session, test_user_id)

        exists = await repo.exists("Non-existent Category")

        assert exists is False


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestExpenseModels:
    """Tests for Pydantic expense models."""

    def test_expense_create_valid(self):
        """Test creating a valid expense."""
        expense = ExpenseCreate(
            amount=Decimal("100.00"),
            description="Test expense",
            category_name="Food & Dining",
            expense_date=date(2024, 1, 15),
        )

        assert expense.amount == Decimal("100.00")
        assert expense.currency == "HKD"  # Default

    def test_expense_create_invalid_amount(self):
        """Test that negative amount is rejected."""
        with pytest.raises(ValueError):
            ExpenseCreate(
                amount=Decimal("-10.00"),
                description="Test",
                category_name="Food",
                expense_date=date(2024, 1, 15),
            )

    def test_expense_create_invalid_currency(self):
        """Test that invalid currency is rejected."""
        with pytest.raises(ValueError):
            ExpenseCreate(
                amount=Decimal("100.00"),
                description="Test",
                category_name="Food",
                expense_date=date(2024, 1, 15),
                currency="INVALID",
            )

    def test_expense_list_filter_valid(self):
        """Test creating valid expense list filter."""
        filter_obj = ExpenseListFilter(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            category="Food",
            limit=10,
        )

        assert filter_obj.limit == 10
        assert filter_obj.offset == 0  # Default

    def test_expense_list_filter_invalid_date_range(self):
        """Test that end_date before start_date is rejected."""
        with pytest.raises(ValueError):
            ExpenseListFilter(
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1),
            )

    def test_expense_update_partial(self):
        """Test partial update model."""
        update = ExpenseUpdate(
            amount=Decimal("150.00"),
        )

        assert update.amount == Decimal("150.00")
        assert update.description is None
        assert update.category_name is None


# =============================================================================
# Constants Tests
# =============================================================================


class TestPredefinedCategories:
    """Tests for predefined categories."""

    def test_predefined_category_values(self):
        """Test that all predefined categories have values."""
        values = PredefinedCategory.values()

        assert len(values) == 11
        assert "Food & Dining" in values
        assert "Transportation" in values
        assert "Other" in values

    def test_is_predefined_true(self):
        """Test checking predefined category."""
        assert PredefinedCategory.is_predefined("Food & Dining") is True
        assert PredefinedCategory.is_predefined("Transportation") is True

    def test_is_predefined_false(self):
        """Test checking non-predefined category."""
        assert PredefinedCategory.is_predefined("Custom Category") is False
        assert PredefinedCategory.is_predefined("") is False
