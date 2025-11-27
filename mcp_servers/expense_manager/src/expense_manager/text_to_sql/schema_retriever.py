"""Schema Retriever Agent for Text-to-SQL pipeline.

Retrieves relevant database schema information to provide context
for SQL generation. Uses predefined schema knowledge for the expense
management domain.
"""

import logging
from dataclasses import dataclass

import dspy

logger = logging.getLogger(__name__)


@dataclass
class TableSchema:
    """Represents a database table schema."""

    name: str
    description: str
    columns: list[dict[str, str]]
    sample_queries: list[str]


# Predefined schema for expense management tables
EXPENSE_SCHEMA = TableSchema(
    name="expenses",
    description="Stores individual expense records with amount, category, date, and metadata",
    columns=[
        {"name": "id", "type": "VARCHAR(50)", "description": "Unique identifier (UUID)"},
        {"name": "user_id", "type": "VARCHAR(255)", "description": "User who owns this expense"},
        {"name": "amount", "type": "NUMERIC(12,2)", "description": "Expense amount (positive decimal)"},
        {"name": "currency", "type": "VARCHAR(3)", "description": "Currency code (e.g., HKD, USD)"},
        {"name": "description", "type": "TEXT", "description": "Description of the expense"},
        {"name": "category_name", "type": "VARCHAR(100)", "description": "Category name (e.g., Food & Dining, Transportation)"},
        {"name": "expense_date", "type": "DATE", "description": "Date when expense occurred"},
        {"name": "merchant_name", "type": "VARCHAR(255)", "description": "Merchant/vendor name (optional)"},
        {"name": "payment_method", "type": "VARCHAR(50)", "description": "Payment method used (optional)"},
        {"name": "notes", "type": "TEXT", "description": "Additional notes (optional)"},
        {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Record creation time"},
        {"name": "updated_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Last update time"},
    ],
    sample_queries=[
        "SELECT SUM(amount) FROM expenses WHERE user_id = :user_id AND expense_date >= :start_date",
        "SELECT category_name, SUM(amount) as total FROM expenses WHERE user_id = :user_id GROUP BY category_name",
        "SELECT * FROM expenses WHERE user_id = :user_id ORDER BY expense_date DESC LIMIT 10",
    ],
)

CATEGORY_SCHEMA = TableSchema(
    name="expense_categories",
    description="Stores custom expense categories created by users",
    columns=[
        {"name": "id", "type": "VARCHAR(50)", "description": "Unique identifier (UUID)"},
        {"name": "user_id", "type": "VARCHAR(255)", "description": "User who owns this category"},
        {"name": "name", "type": "VARCHAR(100)", "description": "Category name"},
        {"name": "description", "type": "TEXT", "description": "Category description (optional)"},
        {"name": "is_active", "type": "BOOLEAN", "description": "Whether category is active (soft delete)"},
        {"name": "created_at", "type": "TIMESTAMP WITH TIME ZONE", "description": "Record creation time"},
    ],
    sample_queries=[
        "SELECT name FROM expense_categories WHERE user_id = :user_id AND is_active = true",
    ],
)

# Predefined categories (always available)
PREDEFINED_CATEGORIES = [
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


class SchemaContextSignature(dspy.Signature):
    """Identify which tables and columns are relevant for a user question."""

    question: str = dspy.InputField(desc="User's natural language question about expenses")
    available_tables: str = dspy.InputField(desc="JSON description of available database tables")

    relevant_tables: list[str] = dspy.OutputField(desc="List of table names relevant to the question")
    relevant_columns: list[str] = dspy.OutputField(desc="List of column names in format 'table.column'")
    reasoning: str = dspy.OutputField(desc="Brief explanation of why these tables/columns are relevant")


class SchemaRetriever:
    """Retrieves relevant schema context for SQL generation.

    This agent analyzes the user's question and determines which
    database tables and columns are relevant for generating the SQL query.
    """

    def __init__(self) -> None:
        """Initialize the schema retriever with predefined schemas."""
        self.schemas = {
            "expenses": EXPENSE_SCHEMA,
            "expense_categories": CATEGORY_SCHEMA,
        }
        self.predefined_categories = PREDEFINED_CATEGORIES
        self._context_predictor = dspy.Predict(SchemaContextSignature)

    def get_full_schema(self) -> str:
        """Get the full schema description as a formatted string."""
        schema_parts = []

        for table_name, schema in self.schemas.items():
            columns_desc = "\n".join(
                f"    - {col['name']} ({col['type']}): {col['description']}"
                for col in schema.columns
            )
            schema_parts.append(
                f"Table: {table_name}\n"
                f"Description: {schema.description}\n"
                f"Columns:\n{columns_desc}"
            )

        schema_parts.append(
            f"\nPredefined Categories (always available):\n"
            f"    {', '.join(self.predefined_categories)}"
        )

        return "\n\n".join(schema_parts)

    def get_schema_for_tables(self, table_names: list[str]) -> str:
        """Get schema description for specific tables."""
        schema_parts = []

        for table_name in table_names:
            if table_name in self.schemas:
                schema = self.schemas[table_name]
                columns_desc = "\n".join(
                    f"    - {col['name']} ({col['type']}): {col['description']}"
                    for col in schema.columns
                )
                schema_parts.append(
                    f"Table: {table_name}\n"
                    f"Description: {schema.description}\n"
                    f"Columns:\n{columns_desc}\n"
                    f"Sample queries:\n    " + "\n    ".join(schema.sample_queries)
                )

        return "\n\n".join(schema_parts)

    async def retrieve_context(self, question: str) -> dict:
        """Retrieve relevant schema context for a question.

        Args:
            question: User's natural language question

        Returns:
            Dictionary with relevant tables, columns, and full context
        """
        logger.info(f"[SchemaRetriever] Retrieving context for question: {question[:100]}...")

        # For simple expense queries, we can use rule-based retrieval
        # More complex scenarios could use the LLM-based predictor

        relevant_tables = ["expenses"]
        relevant_columns = [
            "expenses.amount",
            "expenses.category_name",
            "expenses.expense_date",
            "expenses.description",
            "expenses.merchant_name",
        ]

        # Check if question mentions categories specifically
        category_keywords = ["category", "categories", "type", "types"]
        if any(kw in question.lower() for kw in category_keywords):
            logger.debug("[SchemaRetriever] Question mentions categories, adding expense_categories table")
            relevant_tables.append("expense_categories")
            relevant_columns.extend([
                "expense_categories.name",
                "expense_categories.is_active",
            ])

        schema_context = self.get_schema_for_tables(relevant_tables)
        logger.info(f"[SchemaRetriever] Returning context for tables: {relevant_tables}")
        logger.debug(f"[SchemaRetriever] Relevant columns: {relevant_columns}")
        logger.debug(f"[SchemaRetriever] Schema context length: {len(schema_context)} chars")

        return {
            "relevant_tables": relevant_tables,
            "relevant_columns": relevant_columns,
            "schema_context": schema_context,
            "full_schema": self.get_full_schema(),
            "predefined_categories": self.predefined_categories,
        }

    async def retrieve_context_with_llm(self, question: str) -> dict:
        """Use LLM to determine relevant schema context.

        This method uses DSPy to intelligently select relevant
        tables and columns based on the question semantics.

        Args:
            question: User's natural language question

        Returns:
            Dictionary with relevant tables, columns, and reasoning
        """
        available_tables_json = self.get_full_schema()

        result = self._context_predictor(
            question=question,
            available_tables=available_tables_json,
        )

        return {
            "relevant_tables": result.relevant_tables,
            "relevant_columns": result.relevant_columns,
            "reasoning": result.reasoning,
            "schema_context": self.get_schema_for_tables(result.relevant_tables),
            "full_schema": self.get_full_schema(),
            "predefined_categories": self.predefined_categories,
        }
