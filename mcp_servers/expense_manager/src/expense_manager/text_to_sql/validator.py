"""Result Validator for Text-to-SQL pipeline.

Validates SQL execution results and determines if they
satisfy the user's original question.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import dspy

from expense_manager.text_to_sql.sql_executor import ExecutionResult


class ValidationStatus(str, Enum):
    """Status of result validation."""

    VALID = "valid"  # Results are valid and answer the question
    EMPTY = "empty"  # Query returned no results
    ERROR = "error"  # Query failed with an error
    SUSPICIOUS = "suspicious"  # Results seem incorrect or incomplete
    NEEDS_CLARIFICATION = "needs_clarification"  # Cannot determine validity


@dataclass
class ValidationResult:
    """Result of validation check."""

    status: ValidationStatus
    is_valid: bool
    message: str
    suggestions: list[str] | None = None
    should_retry: bool = False


class ResultValidationSignature(dspy.Signature):
    """Validate if query results answer the user's question."""

    original_question: str = dspy.InputField(desc="User's original question")
    sql_query: str = dspy.InputField(desc="SQL query that was executed")
    result_summary: str = dspy.InputField(desc="Summary of query results")
    row_count: int = dspy.InputField(desc="Number of rows returned")

    is_valid: bool = dspy.OutputField(desc="Whether results answer the question")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    issues: list[str] = dspy.OutputField(desc="List of potential issues with results")
    suggestions: list[str] = dspy.OutputField(desc="Suggestions for improvement")


class ResultValidator:
    """Validates SQL execution results.

    Checks if the query results are valid, complete, and
    actually answer the user's question.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        self._llm_validator = dspy.Predict(ResultValidationSignature)

    def _summarize_results(self, rows: list[dict[str, Any]], max_rows: int = 5) -> str:
        """Create a summary of query results for LLM validation.

        Args:
            rows: Query result rows
            max_rows: Maximum rows to include in summary

        Returns:
            Text summary of results
        """
        if not rows:
            return "No results returned"

        # Get column names from first row
        columns = list(rows[0].keys())
        summary_parts = [f"Columns: {', '.join(columns)}"]
        summary_parts.append(f"Total rows: {len(rows)}")

        # Add sample rows
        sample_rows = rows[:max_rows]
        summary_parts.append("Sample data:")
        for i, row in enumerate(sample_rows, 1):
            row_str = ", ".join(f"{k}={v}" for k, v in row.items())
            summary_parts.append(f"  Row {i}: {row_str}")

        if len(rows) > max_rows:
            summary_parts.append(f"  ... and {len(rows) - max_rows} more rows")

        return "\n".join(summary_parts)

    def validate_execution_result(
        self,
        execution_result: ExecutionResult,
    ) -> ValidationResult:
        """Validate an execution result for basic correctness.

        Args:
            execution_result: Result from SQL executor

        Returns:
            ValidationResult with status and recommendations
        """
        # Check for execution errors
        if not execution_result.success:
            error_type = execution_result.error_type or "unknown"

            suggestions = []
            should_retry = False

            if error_type == "syntax_error":
                suggestions.append("Check SQL syntax and fix any typos")
                should_retry = True
            elif error_type == "column_error":
                suggestions.append("Verify column names match the schema")
                should_retry = True
            elif error_type == "table_error":
                suggestions.append("Verify table names match the schema")
                should_retry = True
            elif error_type == "timeout":
                suggestions.append("Simplify the query or add more specific filters")
                should_retry = True
            else:
                suggestions.append("Review the error message and adjust the query")
                should_retry = True

            return ValidationResult(
                status=ValidationStatus.ERROR,
                is_valid=False,
                message=f"Query execution failed: {execution_result.error}",
                suggestions=suggestions,
                should_retry=should_retry,
            )

        # Check for empty results
        if execution_result.row_count == 0 or execution_result.error_type == "empty_result":
            return ValidationResult(
                status=ValidationStatus.EMPTY,
                is_valid=False,
                message="Query returned no results",
                suggestions=[
                    "Check if the date range is correct",
                    "Verify the category name matches exactly",
                    "Try a broader query without strict filters",
                ],
                should_retry=True,
            )

        # Basic validation passed
        return ValidationResult(
            status=ValidationStatus.VALID,
            is_valid=True,
            message=f"Query returned {execution_result.row_count} rows",
        )

    async def validate_with_llm(
        self,
        original_question: str,
        sql_query: str,
        execution_result: ExecutionResult,
    ) -> ValidationResult:
        """Use LLM to validate if results answer the question.

        Args:
            original_question: User's original question
            sql_query: The SQL query that was executed
            execution_result: Result from SQL executor

        Returns:
            ValidationResult with LLM assessment
        """
        # First do basic validation
        basic_result = self.validate_execution_result(execution_result)
        if not basic_result.is_valid:
            return basic_result

        # Use LLM for semantic validation
        try:
            result_summary = self._summarize_results(execution_result.rows)

            llm_result = self._llm_validator(
                original_question=original_question,
                sql_query=sql_query,
                result_summary=result_summary,
                row_count=execution_result.row_count,
            )

            if llm_result.is_valid and llm_result.confidence >= 0.7:
                return ValidationResult(
                    status=ValidationStatus.VALID,
                    is_valid=True,
                    message=f"Results validated with {llm_result.confidence:.0%} confidence",
                    suggestions=llm_result.suggestions if llm_result.suggestions else None,
                )
            elif llm_result.confidence >= 0.4:
                return ValidationResult(
                    status=ValidationStatus.SUSPICIOUS,
                    is_valid=False,
                    message="Results may not fully answer the question",
                    suggestions=llm_result.suggestions + llm_result.issues,
                    should_retry=True,
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.NEEDS_CLARIFICATION,
                    is_valid=False,
                    message="Cannot determine if results are correct",
                    suggestions=llm_result.suggestions,
                    should_retry=True,
                )

        except Exception as e:
            # Fall back to basic validation if LLM fails
            return ValidationResult(
                status=ValidationStatus.VALID,
                is_valid=True,
                message=f"Query returned {execution_result.row_count} rows (LLM validation skipped: {e})",
            )
