"""SQL Coder Agent for Text-to-SQL pipeline.

Generates SQL queries from natural language questions using
the provided schema context.
"""

import logging
from dataclasses import dataclass

import dspy

logger = logging.getLogger(__name__)


class SQLGenerationSignature(dspy.Signature):
    """Generate a SQL query from a natural language question.

    IMPORTANT RULES:
    1. Always filter by user_id = :user_id for security
    2. Use parameterized queries with :param_name syntax
    3. Only SELECT queries are allowed (no INSERT, UPDATE, DELETE)
    4. Use PostgreSQL syntax
    """

    question: str = dspy.InputField(desc="User's natural language question about expenses")
    schema_context: str = dspy.InputField(desc="Relevant database schema information")
    user_id: str = dspy.InputField(desc="Current user's ID for filtering")

    sql_query: str = dspy.OutputField(
        desc="Valid PostgreSQL SELECT query with :user_id parameter for security"
    )
    explanation: str = dspy.OutputField(
        desc="Brief explanation of what the query does"
    )
    parameters: dict = dspy.OutputField(
        desc="Dictionary of query parameters (excluding user_id which is always included)"
    )


class SQLCorrectionSignature(dspy.Signature):
    """Correct a SQL query based on error feedback.

    Fix the query to address the specific error while maintaining
    the original intent and security constraints.
    """

    original_question: str = dspy.InputField(desc="Original user question")
    original_sql: str = dspy.InputField(desc="The SQL query that failed")
    error_message: str = dspy.InputField(desc="Error message from execution")
    schema_context: str = dspy.InputField(desc="Database schema information")

    corrected_sql: str = dspy.OutputField(desc="Corrected SQL query")
    correction_explanation: str = dspy.OutputField(
        desc="Explanation of what was wrong and how it was fixed"
    )


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""

    sql_query: str
    explanation: str
    parameters: dict
    success: bool = True
    error: str | None = None


class SQLCoder:
    """Generates SQL queries from natural language questions.

    Uses DSPy to convert natural language questions into valid
    PostgreSQL queries with proper security constraints.
    """

    def __init__(self) -> None:
        """Initialize the SQL coder with DSPy predictors."""
        self._generator = dspy.Predict(SQLGenerationSignature)
        self._corrector = dspy.Predict(SQLCorrectionSignature)

    def _validate_sql(self, sql: str) -> tuple[bool, str | None]:
        """Validate that SQL is safe to execute.

        Args:
            sql: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_upper = sql.upper().strip()

        # Only allow SELECT statements
        if not sql_upper.startswith("SELECT"):
            return False, "Only SELECT queries are allowed"

        # Block dangerous keywords
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
            "ALTER", "CREATE", "GRANT", "REVOKE", "EXECUTE",
            "INTO OUTFILE", "INTO DUMPFILE", "LOAD_FILE",
        ]
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Dangerous keyword '{keyword}' is not allowed"

        # Ensure user_id filter is present
        if ":user_id" not in sql.lower() and "user_id" not in sql.lower():
            return False, "Query must filter by user_id for security"

        return True, None

    def _sanitize_sql(self, sql: str, user_id: str) -> str:
        """Ensure SQL has proper user_id filtering.

        Args:
            sql: SQL query to sanitize
            user_id: User ID to filter by

        Returns:
            Sanitized SQL query
        """
        # Ensure the query uses parameterized user_id
        if ":user_id" not in sql:
            # Try to add user_id filter if missing
            sql_upper = sql.upper()
            if "WHERE" in sql_upper:
                # Add to existing WHERE clause
                where_idx = sql_upper.index("WHERE")
                sql = sql[:where_idx + 5] + " user_id = :user_id AND" + sql[where_idx + 5:]
            else:
                # Add WHERE clause before ORDER BY, GROUP BY, LIMIT, or end
                for keyword in ["ORDER BY", "GROUP BY", "LIMIT", ";"]:
                    if keyword in sql_upper:
                        idx = sql_upper.index(keyword)
                        sql = sql[:idx] + " WHERE user_id = :user_id " + sql[idx:]
                        break
                else:
                    sql = sql.rstrip(";") + " WHERE user_id = :user_id"

        return sql

    async def generate(
        self,
        question: str,
        schema_context: str,
        user_id: str,
    ) -> SQLGenerationResult:
        """Generate a SQL query from a natural language question.

        Args:
            question: User's natural language question
            schema_context: Relevant database schema information
            user_id: Current user's ID for security filtering

        Returns:
            SQLGenerationResult with the generated query
        """
        logger.info(f"[SQLCoder] Generating SQL for question: {question[:100]}...")
        logger.debug(f"[SQLCoder] Schema context length: {len(schema_context)} chars")
        logger.debug(f"[SQLCoder] User ID: {user_id}")

        try:
            logger.debug("[SQLCoder] Calling DSPy generator...")
            result = self._generator(
                question=question,
                schema_context=schema_context,
                user_id=user_id,
            )
            logger.info(f"[SQLCoder] DSPy returned SQL: {result.sql_query[:200] if result.sql_query else 'None'}...")
            logger.debug(f"[SQLCoder] DSPy explanation: {result.explanation}")
            logger.debug(f"[SQLCoder] DSPy parameters: {result.parameters}")

            sql = result.sql_query.strip()

            # Sanitize and validate
            logger.debug("[SQLCoder] Sanitizing SQL...")
            sql = self._sanitize_sql(sql, user_id)
            logger.debug(f"[SQLCoder] Sanitized SQL: {sql}")

            logger.debug("[SQLCoder] Validating SQL...")
            is_valid, error = self._validate_sql(sql)
            logger.info(f"[SQLCoder] SQL validation: is_valid={is_valid}, error={error}")

            if not is_valid:
                logger.warning(f"[SQLCoder] SQL validation failed: {error}")
                return SQLGenerationResult(
                    sql_query=sql,
                    explanation=result.explanation,
                    parameters=result.parameters or {},
                    success=False,
                    error=error,
                )

            # Ensure user_id is in parameters
            parameters = result.parameters or {}
            parameters["user_id"] = user_id

            logger.info(f"[SQLCoder] Successfully generated SQL with {len(parameters)} parameters")
            return SQLGenerationResult(
                sql_query=sql,
                explanation=result.explanation,
                parameters=parameters,
                success=True,
            )

        except Exception as e:
            logger.error(f"[SQLCoder] Generation failed with exception: {e}", exc_info=True)
            return SQLGenerationResult(
                sql_query="",
                explanation="",
                parameters={},
                success=False,
                error=f"Generation failed: {str(e)}",
            )

    async def correct(
        self,
        original_question: str,
        original_sql: str,
        error_message: str,
        schema_context: str,
        user_id: str,
    ) -> SQLGenerationResult:
        """Correct a failed SQL query based on error feedback.

        Args:
            original_question: Original user question
            original_sql: The SQL query that failed
            error_message: Error message from execution
            schema_context: Database schema information
            user_id: Current user's ID

        Returns:
            SQLGenerationResult with the corrected query
        """
        try:
            result = self._corrector(
                original_question=original_question,
                original_sql=original_sql,
                error_message=error_message,
                schema_context=schema_context,
            )

            sql = result.corrected_sql.strip()

            # Sanitize and validate
            sql = self._sanitize_sql(sql, user_id)
            is_valid, error = self._validate_sql(sql)

            if not is_valid:
                return SQLGenerationResult(
                    sql_query=sql,
                    explanation=result.correction_explanation,
                    parameters={"user_id": user_id},
                    success=False,
                    error=error,
                )

            return SQLGenerationResult(
                sql_query=sql,
                explanation=result.correction_explanation,
                parameters={"user_id": user_id},
                success=True,
            )

        except Exception as e:
            return SQLGenerationResult(
                sql_query="",
                explanation="",
                parameters={},
                success=False,
                error=f"Correction failed: {str(e)}",
            )
