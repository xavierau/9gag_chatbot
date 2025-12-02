"""SQL Executor Tool for Text-to-SQL pipeline.

Safely executes SQL queries against the database with
proper parameter binding and error handling.
"""

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Pattern for ISO date strings (YYYY-MM-DD)
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Pattern for ISO datetime strings (YYYY-MM-DDTHH:MM:SS or with timezone)
DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


@dataclass
class ExecutionResult:
    """Result of SQL execution."""

    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    error_type: str | None = None  # 'db_error', 'empty_result', 'timeout'


class SQLExecutor:
    """Executes SQL queries safely against the database.

    Only executes SELECT queries with proper parameter binding
    to prevent SQL injection.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize executor with database session.

        Args:
            session: Async SQLAlchemy session
        """
        self.session = session

    def _convert_parameter(self, value: Any) -> Any:
        """Convert string parameters to appropriate Python types for asyncpg.

        asyncpg requires native Python types (date, datetime) instead of
        ISO format strings when binding to date/timestamp columns.

        Args:
            value: Parameter value to convert

        Returns:
            Converted value (date/datetime object or original value)
        """
        if not isinstance(value, str):
            return value

        # Try to parse as date (YYYY-MM-DD)
        if DATE_PATTERN.match(value):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass

        # Try to parse as datetime (YYYY-MM-DDTHH:MM:SS...)
        if DATETIME_PATTERN.match(value):
            try:
                # Handle both 'T' and space separators
                return datetime.fromisoformat(value.replace(" ", "T"))
            except ValueError:
                pass

        return value

    def _convert_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Convert all parameters in a dictionary.

        Args:
            parameters: Dictionary of parameter names to values

        Returns:
            New dictionary with converted values
        """
        return {key: self._convert_parameter(value) for key, value in parameters.items()}

    def _serialize_value(self, value: Any) -> Any:
        """Convert database values to JSON-serializable types.

        Args:
            value: Value from database row

        Returns:
            JSON-serializable value
        """
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _row_to_dict(self, row: Any, columns: list[str]) -> dict[str, Any]:
        """Convert a database row to a dictionary.

        Args:
            row: Database row (tuple or mapping)
            columns: Column names

        Returns:
            Dictionary with column names as keys
        """
        if hasattr(row, "_mapping"):
            # SQLAlchemy Row object
            return {
                col: self._serialize_value(row._mapping[col])
                for col in columns
            }
        if hasattr(row, "_asdict"):
            # Named tuple
            return {
                k: self._serialize_value(v)
                for k, v in row._asdict().items()
            }
        # Regular tuple
        return {
            col: self._serialize_value(val)
            for col, val in zip(columns, row)
        }

    async def execute(
        self,
        sql: str,
        parameters: dict[str, Any],
        max_rows: int = 1000,
    ) -> ExecutionResult:
        """Execute a SQL query with parameters.

        Args:
            sql: SQL query with :param_name placeholders
            parameters: Dictionary of parameter values
            max_rows: Maximum rows to return (safety limit)

        Returns:
            ExecutionResult with rows or error
        """
        logger.info(f"[SQLExecutor] Executing SQL query...")
        logger.debug(f"[SQLExecutor] SQL: {sql}")
        logger.debug(f"[SQLExecutor] Parameters (raw): {parameters}")

        # Convert string dates/datetimes to Python objects for asyncpg
        parameters = self._convert_parameters(parameters)
        logger.debug(f"[SQLExecutor] Parameters (converted): {parameters}")
        logger.debug(f"[SQLExecutor] Max rows: {max_rows}")

        # Validate it's a SELECT query
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            logger.error("[SQLExecutor] Query is not a SELECT statement")
            return ExecutionResult(
                success=False,
                error="Only SELECT queries are allowed",
                error_type="validation_error",
            )

        # Add LIMIT if not present (safety measure)
        if "LIMIT" not in sql_stripped:
            sql = sql.rstrip(";") + f" LIMIT {max_rows}"
            logger.debug(f"[SQLExecutor] Added LIMIT clause: {sql}")

        try:
            logger.debug("[SQLExecutor] Executing query against database...")
            result = await self.session.execute(text(sql), parameters)

            # Get column names
            columns = list(result.keys()) if result.keys() else []
            logger.debug(f"[SQLExecutor] Result columns: {columns}")

            # Fetch all rows
            raw_rows = result.fetchall()
            logger.info(f"[SQLExecutor] Query returned {len(raw_rows)} raw rows")

            if not raw_rows:
                logger.info("[SQLExecutor] Query returned no results (empty)")
                return ExecutionResult(
                    success=True,
                    rows=[],
                    row_count=0,
                    columns=columns,
                    error="Query returned no results",
                    error_type="empty_result",
                )

            # Convert rows to dictionaries
            rows = [self._row_to_dict(row, columns) for row in raw_rows]
            logger.info(f"[SQLExecutor] Successfully converted {len(rows)} rows")
            if rows:
                logger.debug(f"[SQLExecutor] Sample row: {rows[0]}")

            return ExecutionResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                columns=columns,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SQLExecutor] Database execution error: {error_msg}", exc_info=True)

            # Categorize error type
            error_type = "db_error"
            if "timeout" in error_msg.lower():
                error_type = "timeout"
            elif "syntax" in error_msg.lower():
                error_type = "syntax_error"
            elif "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                error_type = "column_error"
            elif "relation" in error_msg.lower() and "does not exist" in error_msg.lower():
                error_type = "table_error"

            logger.error(f"[SQLExecutor] Categorized error type: {error_type}")
            return ExecutionResult(
                success=False,
                error=error_msg,
                error_type=error_type,
            )

    async def execute_with_explanation(
        self,
        sql: str,
        parameters: dict[str, Any],
    ) -> ExecutionResult:
        """Execute a SQL query and include EXPLAIN output.

        Useful for debugging performance issues.

        Args:
            sql: SQL query
            parameters: Query parameters

        Returns:
            ExecutionResult with EXPLAIN information
        """
        # Convert parameters for asyncpg compatibility
        converted_params = self._convert_parameters(parameters)

        # First get EXPLAIN output
        explain_sql = f"EXPLAIN {sql}"
        try:
            explain_result = await self.session.execute(text(explain_sql), converted_params)
            explain_rows = explain_result.fetchall()
            explain_output = "\n".join(str(row[0]) for row in explain_rows)
        except Exception:
            explain_output = "Could not generate query plan"

        # Then execute the actual query (will also convert, but already converted)
        result = await self.execute(sql, converted_params)

        # Add explain output to the result
        if result.success:
            result.rows.append({"_explain": explain_output})

        return result
