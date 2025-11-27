"""Text-to-SQL agentic system for expense queries.

This module implements a multi-agent pipeline for converting natural language
questions into SQL queries against the expense database.

Architecture:
    1. Orchestrator: Coordinates the entire pipeline
    2. Schema Retriever: Retrieves relevant schema context via RAG
    3. Ambiguity Checker: Detects ambiguous terms and requests clarification
    4. SQL Coder: Generates SQL from natural language
    5. SQL Executor: Executes SQL and returns results
    6. Result Validator: Validates execution results
    7. Debugger: Fixes SQL errors and empty results
"""

from expense_manager.text_to_sql.orchestrator import TextToSQLOrchestrator
from expense_manager.text_to_sql.schema_retriever import SchemaRetriever
from expense_manager.text_to_sql.sql_coder import SQLCoder
from expense_manager.text_to_sql.sql_executor import SQLExecutor
from expense_manager.text_to_sql.validator import ResultValidator
from expense_manager.text_to_sql.debugger import SQLDebugger

__all__ = [
    "TextToSQLOrchestrator",
    "SchemaRetriever",
    "SQLCoder",
    "SQLExecutor",
    "ResultValidator",
    "SQLDebugger",
]
