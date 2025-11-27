"""Debugger Agent for Text-to-SQL pipeline.

Analyzes failed queries and generates corrected SQL
based on error messages and empty result analysis.
"""

from dataclasses import dataclass
from enum import Enum

import dspy

from expense_manager.text_to_sql.sql_executor import ExecutionResult
from expense_manager.text_to_sql.validator import ValidationResult, ValidationStatus


class DebugAction(str, Enum):
    """Actions the debugger can recommend."""

    FIX_SYNTAX = "fix_syntax"
    FIX_COLUMN = "fix_column"
    FIX_TABLE = "fix_table"
    BROADEN_FILTER = "broaden_filter"
    CHANGE_DATE_RANGE = "change_date_range"
    CHANGE_CATEGORY = "change_category"
    SIMPLIFY_QUERY = "simplify_query"
    CLARIFY_QUESTION = "clarify_question"
    GIVE_UP = "give_up"


class SQLDebugSignature(dspy.Signature):
    """Analyze a failed SQL query and suggest fixes."""

    original_question: str = dspy.InputField(desc="User's original question")
    failed_sql: str = dspy.InputField(desc="SQL query that failed")
    error_message: str = dspy.InputField(desc="Error message or reason for failure")
    error_type: str = dspy.InputField(desc="Type of error (db_error, empty_result, etc.)")
    schema_context: str = dspy.InputField(desc="Database schema information")
    attempt_number: int = dspy.InputField(desc="Current retry attempt number")

    diagnosis: str = dspy.OutputField(desc="Analysis of what went wrong")
    corrected_sql: str = dspy.OutputField(desc="Corrected SQL query")
    explanation: str = dspy.OutputField(desc="Explanation of the fix")
    confidence: float = dspy.OutputField(desc="Confidence in the fix (0-1)")
    should_clarify: bool = dspy.OutputField(desc="Whether to ask user for clarification")
    clarification_question: str = dspy.OutputField(desc="Question to ask user if clarification needed")


class EmptyResultAnalysisSignature(dspy.Signature):
    """Analyze why a query returned no results."""

    original_question: str = dspy.InputField(desc="User's original question")
    sql_query: str = dspy.InputField(desc="SQL query that returned no results")
    schema_context: str = dspy.InputField(desc="Database schema information")

    likely_reasons: list[str] = dspy.OutputField(desc="Likely reasons for empty results")
    suggested_modifications: list[str] = dspy.OutputField(desc="Suggested query modifications")
    alternative_sql: str = dspy.OutputField(desc="Alternative SQL query to try")
    should_ask_user: bool = dspy.OutputField(desc="Whether to ask user for more context")
    question_for_user: str = dspy.OutputField(desc="Question to ask user if needed")


@dataclass
class DebugResult:
    """Result of debugging analysis."""

    action: DebugAction
    corrected_sql: str | None
    diagnosis: str
    explanation: str
    confidence: float
    should_clarify: bool = False
    clarification_question: str | None = None
    can_retry: bool = True


class SQLDebugger:
    """Debugs failed SQL queries and suggests corrections.

    Analyzes errors and empty results to generate improved
    SQL queries or ask for user clarification.
    """

    def __init__(self, max_retries: int = 3) -> None:
        """Initialize the debugger.

        Args:
            max_retries: Maximum retry attempts before giving up
        """
        self.max_retries = max_retries
        self._debug_predictor = dspy.Predict(SQLDebugSignature)
        self._empty_analyzer = dspy.Predict(EmptyResultAnalysisSignature)

    def _determine_action(self, error_type: str | None, error_message: str) -> DebugAction:
        """Determine the appropriate debug action based on error.

        Args:
            error_type: Type of error from executor
            error_message: Error message

        Returns:
            Recommended debug action
        """
        if error_type == "syntax_error":
            return DebugAction.FIX_SYNTAX
        if error_type == "column_error":
            return DebugAction.FIX_COLUMN
        if error_type == "table_error":
            return DebugAction.FIX_TABLE
        if error_type == "empty_result":
            return DebugAction.BROADEN_FILTER
        if error_type == "timeout":
            return DebugAction.SIMPLIFY_QUERY

        # Analyze error message for more specific actions
        error_lower = error_message.lower()
        if "column" in error_lower:
            return DebugAction.FIX_COLUMN
        if "table" in error_lower or "relation" in error_lower:
            return DebugAction.FIX_TABLE
        if "syntax" in error_lower:
            return DebugAction.FIX_SYNTAX
        if "date" in error_lower:
            return DebugAction.CHANGE_DATE_RANGE

        return DebugAction.SIMPLIFY_QUERY

    async def debug(
        self,
        original_question: str,
        failed_sql: str,
        execution_result: ExecutionResult,
        validation_result: ValidationResult,
        schema_context: str,
        attempt_number: int,
    ) -> DebugResult:
        """Debug a failed query and suggest corrections.

        Args:
            original_question: User's original question
            failed_sql: SQL that failed or returned bad results
            execution_result: Result from SQL executor
            validation_result: Result from validator
            schema_context: Database schema information
            attempt_number: Current retry attempt number

        Returns:
            DebugResult with correction recommendations
        """
        # Check if we've exceeded max retries
        if attempt_number >= self.max_retries:
            return DebugResult(
                action=DebugAction.GIVE_UP,
                corrected_sql=None,
                diagnosis="Maximum retry attempts exceeded",
                explanation="Unable to generate a working query after multiple attempts",
                confidence=0.0,
                should_clarify=True,
                clarification_question="I'm having trouble understanding your question. Could you rephrase it or provide more details?",
                can_retry=False,
            )

        # Determine initial action
        error_message = execution_result.error or validation_result.message
        error_type = execution_result.error_type or (
            "empty_result" if validation_result.status == ValidationStatus.EMPTY else "unknown"
        )
        action = self._determine_action(error_type, error_message)

        # Use LLM for detailed analysis
        try:
            llm_result = self._debug_predictor(
                original_question=original_question,
                failed_sql=failed_sql,
                error_message=error_message,
                error_type=error_type,
                schema_context=schema_context,
                attempt_number=attempt_number,
            )

            return DebugResult(
                action=action,
                corrected_sql=llm_result.corrected_sql,
                diagnosis=llm_result.diagnosis,
                explanation=llm_result.explanation,
                confidence=llm_result.confidence,
                should_clarify=llm_result.should_clarify,
                clarification_question=llm_result.clarification_question if llm_result.should_clarify else None,
                can_retry=llm_result.confidence >= 0.3,
            )

        except Exception as e:
            # Fallback to rule-based debugging
            return self._rule_based_debug(
                action=action,
                failed_sql=failed_sql,
                error_message=error_message,
                error_type=error_type,
            )

    def _rule_based_debug(
        self,
        action: DebugAction,
        failed_sql: str,
        error_message: str,
        error_type: str,
    ) -> DebugResult:
        """Fallback rule-based debugging when LLM is unavailable.

        Args:
            action: Determined debug action
            failed_sql: The failed SQL query
            error_message: Error message
            error_type: Type of error

        Returns:
            DebugResult with rule-based corrections
        """
        diagnosis = f"Error type: {error_type}. Message: {error_message}"
        explanation = "Applying rule-based fix"
        corrected_sql = failed_sql

        if action == DebugAction.FIX_COLUMN:
            explanation = "Check column names against schema"
        elif action == DebugAction.FIX_TABLE:
            explanation = "Check table names against schema"
        elif action == DebugAction.FIX_SYNTAX:
            explanation = "Fix SQL syntax errors"
        elif action == DebugAction.BROADEN_FILTER:
            # Try removing some WHERE conditions
            explanation = "Trying to broaden filters"
            if "WHERE" in failed_sql.upper():
                # Simple heuristic: suggest removing date constraints
                corrected_sql = failed_sql.replace("AND expense_date", "-- AND expense_date")
        elif action == DebugAction.SIMPLIFY_QUERY:
            explanation = "Simplifying query to reduce complexity"

        return DebugResult(
            action=action,
            corrected_sql=corrected_sql,
            diagnosis=diagnosis,
            explanation=explanation,
            confidence=0.5,
            should_clarify=False,
            can_retry=True,
        )

    async def analyze_empty_result(
        self,
        original_question: str,
        sql_query: str,
        schema_context: str,
    ) -> DebugResult:
        """Analyze why a query returned no results.

        Args:
            original_question: User's original question
            sql_query: Query that returned empty results
            schema_context: Database schema information

        Returns:
            DebugResult with analysis and suggestions
        """
        try:
            result = self._empty_analyzer(
                original_question=original_question,
                sql_query=sql_query,
                schema_context=schema_context,
            )

            return DebugResult(
                action=DebugAction.BROADEN_FILTER,
                corrected_sql=result.alternative_sql,
                diagnosis="\n".join(result.likely_reasons),
                explanation="\n".join(result.suggested_modifications),
                confidence=0.6,
                should_clarify=result.should_ask_user,
                clarification_question=result.question_for_user if result.should_ask_user else None,
                can_retry=True,
            )

        except Exception:
            return DebugResult(
                action=DebugAction.BROADEN_FILTER,
                corrected_sql=None,
                diagnosis="Query returned no results",
                explanation="Try broadening your search criteria or checking if data exists for the specified filters",
                confidence=0.3,
                should_clarify=True,
                clarification_question="Your query returned no results. Could you provide different criteria or date range?",
                can_retry=False,
            )
