"""Constants and enumerations for the Expense Manager.

Defines predefined expense categories that are always available to users.
Users can also create custom categories.
"""

from enum import Enum


class PredefinedCategory(str, Enum):
    """Predefined expense categories available to all users.

    These categories cannot be deleted and are always available.
    Users can create additional custom categories.
    """

    FOOD = "Food & Dining"
    TRANSPORT = "Transportation"
    UTILITIES = "Utilities"
    ENTERTAINMENT = "Entertainment"
    SHOPPING = "Shopping"
    HEALTHCARE = "Healthcare"
    HOUSING = "Housing & Rent"
    EDUCATION = "Education"
    TRAVEL = "Travel"
    SUBSCRIPTIONS = "Subscriptions"
    OTHER = "Other"

    @classmethod
    def values(cls) -> list[str]:
        """Return all predefined category values."""
        return [category.value for category in cls]

    @classmethod
    def is_predefined(cls, name: str) -> bool:
        """Check if a category name is a predefined category."""
        return name in cls.values()


# Common payment methods for expense tracking
PAYMENT_METHODS = [
    "Cash",
    "Credit Card",
    "Debit Card",
    "Bank Transfer",
    "Mobile Payment",
    "Octopus Card",
    "PayMe",
    "AliPay",
    "WeChat Pay",
    "Other",
]

# Supported currencies
SUPPORTED_CURRENCIES = [
    "HKD",  # Hong Kong Dollar (default)
    "USD",  # US Dollar
    "EUR",  # Euro
    "GBP",  # British Pound
    "CNY",  # Chinese Yuan
    "JPY",  # Japanese Yen
    "SGD",  # Singapore Dollar
    "AUD",  # Australian Dollar
]
