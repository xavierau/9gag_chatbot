"""Orchestrator for Text-to-SQL pipeline.

Coordinates the entire flow from natural language question
to SQL execution and result validation, including error
handling and retry logic.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import dspy
from sqlalchemy.ext.asyncio import AsyncSession

from expense_manager.text_to_sql.schema_retriever import SchemaRetriever
from expense_manager.text_to_sql.sql_coder import SQLCoder, SQLGenerationResult
from expense_manager.text_to_sql.sql_executor import SQLExecutor, ExecutionResult
from expense_manager.text_to_sql.validator import ResultValidator, ValidationResult, ValidationStatus
from expense_manager.text_to_sql.debugger import SQLDebugger, DebugResult

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Status of the pipeline execution."""

    SUCCESS = "success"
    ERROR = "error"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_RESULTS = "no_results"


class AmbiguityCheckSignature(dspy.Signature):
    """Check if a user question is ambiguous and needs clarification."""

    question: str = dspy.InputField(desc="User's natural language question")
    schema_context: str = dspy.InputField(desc="Available database schema")

    is_ambiguous: bool = dspy.OutputField(desc="Whether the question is ambiguous")
    ambiguous_terms: list[str] = dspy.OutputField(desc="List of ambiguous terms")
    clarification_question: str = dspy.OutputField(desc="Question to ask user for clarification")
    assumed_interpretation: str = dspy.OutputField(desc="How the question would be interpreted if proceeding")


@dataclass
class PipelineResult:
    """Result of the Text-to-SQL pipeline."""

    status: PipelineStatus
    data: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    sql_query: str | None = None
    explanation: str | None = None
    error: str | None = None
    needs_clarification: bool = False
    clarification_question: str | None = None
    execution_trace: list[dict[str, Any]] = field(default_factory=list)


class TextToSQLOrchestrator:
    """Orchestrates the Text-to-SQL pipeline.

    Coordinates:
    1. Schema retrieval for context
    2. Ambiguity checking (optional)
    3. SQL generation
    4. Execution and validation
    5. Debugging and retry on failure

    Flow:
        User Question
            → Schema Retriever (context)
            → Ambiguity Check (optional)
            → SQL Coder (generate SQL)
            → SQL Executor (run query)
            → Result Validator (check results)
            → Debugger (if errors) → retry loop
            → Final Result
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        max_retries: int = 3,
        check_ambiguity: bool = False,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            session: Database session for query execution
            user_id: Current user's ID for security filtering
            max_retries: Maximum retry attempts for failed queries
            check_ambiguity: Whether to check for ambiguous questions
        """
        self.session = session
        self.user_id = user_id
        self.max_retries = max_retries
        self.check_ambiguity = check_ambiguity

        # Initialize components
        self.schema_retriever = SchemaRetriever()
        self.sql_coder = SQLCoder()
        self.sql_executor = SQLExecutor(session)
        self.validator = ResultValidator()
        self.debugger = SQLDebugger(max_retries=max_retries)

        # Ambiguity checker (lazy initialized)
        self._ambiguity_checker: dspy.Predict | None = None

    def _get_ambiguity_checker(self) -> dspy.Predict:
        """Get or create the ambiguity checker."""
        if self._ambiguity_checker is None:
            self._ambiguity_checker = dspy.Predict(AmbiguityCheckSignature)
        return self._ambiguity_checker

    def _add_trace(
        self,
        trace: list[dict[str, Any]],
        step: str,
        details: dict[str, Any],
    ) -> None:
        """Add a step to the execution trace."""
        trace.append({
            "step": step,
            **details,
        })

    async def _check_ambiguity(
        self,
        question: str,
        schema_context: str,
        trace: list[dict[str, Any]],
    ) -> tuple[bool, str | None, str | None]:
        """Check if the question is ambiguous.

        Returns:
            Tuple of (is_ambiguous, clarification_question, assumed_interpretation)
        """
        try:
            checker = self._get_ambiguity_checker()
            result = checker(
                question=question,
                schema_context=schema_context,
            )

            self._add_trace(trace, "ambiguity_check", {
                "is_ambiguous": result.is_ambiguous,
                "ambiguous_terms": result.ambiguous_terms,
            })

            if result.is_ambiguous:
                return True, result.clarification_question, result.assumed_interpretation

            return False, None, None

        except Exception as e:
            self._add_trace(trace, "ambiguity_check", {
                "error": str(e),
                "skipped": True,
            })
            return False, None, None

    async def execute(
        self,
        question: str,
        additional_context: str | None = None,
    ) -> PipelineResult:
        """Execute the full Text-to-SQL pipeline.

        Args:
            question: User's natural language question
            additional_context: Optional context from previous clarifications

        Returns:
            PipelineResult with query results or clarification request
        """
        logger.info(f"[TextToSQL] Starting pipeline execution for user_id={self.user_id}")
        logger.info(f"[TextToSQL] Question: {question}")
        if additional_context:
            logger.info(f"[TextToSQL] Additional context: {additional_context}")

        trace: list[dict[str, Any]] = []

        # Phase 1: Get schema context
        self._add_trace(trace, "start", {"question": question})
        logger.debug("[TextToSQL] Phase 1: Retrieving schema context")

        try:
            schema_result = await self.schema_retriever.retrieve_context(question)
            schema_context = schema_result["schema_context"]
            logger.info(f"[TextToSQL] Schema retrieved, relevant_tables={schema_result['relevant_tables']}")
            logger.debug(f"[TextToSQL] Schema context length: {len(schema_context)} chars")
        except Exception as e:
            logger.error(f"[TextToSQL] Schema retrieval failed: {e}", exc_info=True)
            return PipelineResult(
                status=PipelineStatus.ERROR,
                error=f"Schema retrieval failed: {str(e)}",
                execution_trace=trace,
            )

        self._add_trace(trace, "schema_retrieval", {
            "relevant_tables": schema_result["relevant_tables"],
        })

        # Phase 1b: Ambiguity check (if enabled)
        if self.check_ambiguity:
            is_ambiguous, clarification_q, assumed = await self._check_ambiguity(
                question, schema_context, trace
            )

            if is_ambiguous and clarification_q:
                return PipelineResult(
                    status=PipelineStatus.NEEDS_CLARIFICATION,
                    needs_clarification=True,
                    clarification_question=clarification_q,
                    explanation=f"Assumed interpretation: {assumed}" if assumed else None,
                    execution_trace=trace,
                )

        # Combine question with additional context if provided
        full_question = question
        if additional_context:
            full_question = f"{question}\n\nAdditional context: {additional_context}"

        # Phase 2: Generate SQL
        logger.debug("[TextToSQL] Phase 2: Generating SQL")
        try:
            generation_result = await self.sql_coder.generate(
                question=full_question,
                schema_context=schema_context,
                user_id=self.user_id,
            )
            logger.info(f"[TextToSQL] SQL generation success={generation_result.success}")
            if generation_result.success:
                logger.info(f"[TextToSQL] Generated SQL: {generation_result.sql_query}")
                logger.debug(f"[TextToSQL] SQL explanation: {generation_result.explanation}")
            else:
                logger.warning(f"[TextToSQL] SQL generation failed: {generation_result.error}")
        except Exception as e:
            logger.error(f"[TextToSQL] SQL generation exception: {e}", exc_info=True)
            return PipelineResult(
                status=PipelineStatus.ERROR,
                error=f"SQL generation exception: {str(e)}",
                execution_trace=trace,
            )

        self._add_trace(trace, "sql_generation", {
            "success": generation_result.success,
            "sql": generation_result.sql_query,
            "explanation": generation_result.explanation,
        })

        if not generation_result.success:
            logger.error(f"[TextToSQL] Pipeline failed at SQL generation: {generation_result.error}")
            return PipelineResult(
                status=PipelineStatus.ERROR,
                error=f"Failed to generate SQL: {generation_result.error}",
                execution_trace=trace,
            )

        # Phase 3: Execute and validate with retry loop
        logger.debug("[TextToSQL] Phase 3: Execute and validate with retry loop")
        current_sql = generation_result.sql_query
        current_params = generation_result.parameters
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"[TextToSQL] Execution attempt {attempt}/{self.max_retries}")

            # Execute query
            try:
                execution_result = await self.sql_executor.execute(
                    sql=current_sql,
                    parameters=current_params,
                )
                logger.info(f"[TextToSQL] Execution success={execution_result.success}, row_count={execution_result.row_count}")
                if execution_result.error:
                    logger.warning(f"[TextToSQL] Execution error: {execution_result.error}")
                    logger.warning(f"[TextToSQL] Error type: {execution_result.error_type}")
            except Exception as e:
                logger.error(f"[TextToSQL] Execution exception: {e}", exc_info=True)
                return PipelineResult(
                    status=PipelineStatus.ERROR,
                    error=f"Execution exception: {str(e)}",
                    sql_query=current_sql,
                    execution_trace=trace,
                )

            self._add_trace(trace, f"execution_attempt_{attempt}", {
                "sql": current_sql,
                "success": execution_result.success,
                "row_count": execution_result.row_count,
                "error": execution_result.error,
            })

            # Validate results
            validation_result = self.validator.validate_execution_result(execution_result)
            logger.info(f"[TextToSQL] Validation status={validation_result.status.value}, is_valid={validation_result.is_valid}")
            if not validation_result.is_valid:
                logger.warning(f"[TextToSQL] Validation message: {validation_result.message}")

            self._add_trace(trace, f"validation_attempt_{attempt}", {
                "status": validation_result.status.value,
                "is_valid": validation_result.is_valid,
                "message": validation_result.message,
            })

            # Success case
            if validation_result.is_valid:
                logger.info(f"[TextToSQL] Pipeline SUCCESS - returned {execution_result.row_count} rows")
                return PipelineResult(
                    status=PipelineStatus.SUCCESS,
                    data=execution_result.rows,
                    row_count=execution_result.row_count,
                    sql_query=current_sql,
                    explanation=generation_result.explanation,
                    execution_trace=trace,
                )

            # Check if we should retry
            if not validation_result.should_retry:
                logger.info("[TextToSQL] Validation says should_retry=False, breaking loop")
                break

            # Debug and get correction
            logger.debug(f"[TextToSQL] Attempting debug for attempt {attempt}")
            try:
                debug_result = await self.debugger.debug(
                    original_question=full_question,
                    failed_sql=current_sql,
                    execution_result=execution_result,
                    validation_result=validation_result,
                    schema_context=schema_context,
                    attempt_number=attempt,
                )
                logger.info(f"[TextToSQL] Debug action={debug_result.action.value}, confidence={debug_result.confidence}")
                logger.debug(f"[TextToSQL] Debug diagnosis: {debug_result.diagnosis}")
            except Exception as e:
                logger.error(f"[TextToSQL] Debug exception: {e}", exc_info=True)
                break

            self._add_trace(trace, f"debug_attempt_{attempt}", {
                "action": debug_result.action.value,
                "diagnosis": debug_result.diagnosis,
                "confidence": debug_result.confidence,
                "can_retry": debug_result.can_retry,
            })

            # Check if debugger wants clarification
            if debug_result.should_clarify and debug_result.clarification_question:
                logger.info(f"[TextToSQL] Debugger requesting clarification: {debug_result.clarification_question}")
                return PipelineResult(
                    status=PipelineStatus.NEEDS_CLARIFICATION,
                    needs_clarification=True,
                    clarification_question=debug_result.clarification_question,
                    sql_query=current_sql,
                    explanation=debug_result.diagnosis,
                    execution_trace=trace,
                )

            # Check if we can retry
            if not debug_result.can_retry or not debug_result.corrected_sql:
                logger.info(f"[TextToSQL] Cannot retry: can_retry={debug_result.can_retry}, has_corrected_sql={bool(debug_result.corrected_sql)}")
                break

            # Update SQL for next attempt
            logger.info(f"[TextToSQL] Retrying with corrected SQL: {debug_result.corrected_sql}")
            current_sql = debug_result.corrected_sql

        # All retries exhausted
        final_status = (
            PipelineStatus.NO_RESULTS
            if validation_result.status == ValidationStatus.EMPTY
            else PipelineStatus.ERROR
        )
        logger.warning(f"[TextToSQL] Pipeline ended with status={final_status.value} after {attempt} attempts")
        logger.warning(f"[TextToSQL] Final error: {validation_result.message}")

        return PipelineResult(
            status=final_status,
            sql_query=current_sql,
            error=validation_result.message,
            explanation=f"Failed after {attempt} attempts",
            execution_trace=trace,
        )

    async def execute_sql_directly(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute a SQL query directly without generation.

        Useful for testing or when SQL is already known.

        Args:
            sql: SQL query to execute
            parameters: Query parameters

        Returns:
            PipelineResult with query results
        """
        trace: list[dict[str, Any]] = []

        params = parameters or {}
        params["user_id"] = self.user_id

        execution_result = await self.sql_executor.execute(
            sql=sql,
            parameters=params,
        )

        self._add_trace(trace, "direct_execution", {
            "sql": sql,
            "success": execution_result.success,
            "row_count": execution_result.row_count,
            "error": execution_result.error,
        })

        if execution_result.success and execution_result.row_count > 0:
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                data=execution_result.rows,
                row_count=execution_result.row_count,
                sql_query=sql,
                execution_trace=trace,
            )

        if execution_result.error_type == "empty_result":
            return PipelineResult(
                status=PipelineStatus.NO_RESULTS,
                sql_query=sql,
                error="Query returned no results",
                execution_trace=trace,
            )

        return PipelineResult(
            status=PipelineStatus.ERROR,
            sql_query=sql,
            error=execution_result.error,
            execution_trace=trace,
        )
