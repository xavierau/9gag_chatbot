"""Unit tests for the MCPSubAgent.

These tests verify the sub-agent's behavior including:
- SubAgentResult dataclass
- SubAgentSignature
- MCPSubAgent initialization and execution
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.llm.subagent import (
    MCPSubAgent,
    SubAgentResult,
    SubAgentSignature,
)


class TestSubAgentResult:
    """Tests for the SubAgentResult dataclass."""

    def test_result_success(self):
        """Test creating a successful result."""
        result = SubAgentResult(
            success=True,
            result="Task completed successfully",
            duration_ms=150.5,
            iterations=2,
            tools_called=["create_expense", "list_expenses"],
        )

        assert result.success is True
        assert result.result == "Task completed successfully"
        assert result.error is None
        assert result.error_code is None
        assert result.duration_ms == 150.5
        assert result.iterations == 2
        assert result.tools_called == ["create_expense", "list_expenses"]

    def test_result_failure(self):
        """Test creating a failure result."""
        result = SubAgentResult(
            success=False,
            error="Connection failed",
            error_code="CONNECTION_FAILED",
            duration_ms=50.0,
        )

        assert result.success is False
        assert result.result is None
        assert result.error == "Connection failed"
        assert result.error_code == "CONNECTION_FAILED"
        assert result.duration_ms == 50.0

    def test_result_default_tools_called(self):
        """Test that tools_called defaults to empty list."""
        result = SubAgentResult(success=True, result="Done")

        assert result.tools_called == []

    def test_result_error_codes(self):
        """Test various error codes."""
        error_codes = ["CONNECTION_FAILED", "EXECUTION_ERROR", "TIMEOUT"]

        for code in error_codes:
            result = SubAgentResult(
                success=False,
                error=f"Error: {code}",
                error_code=code,
            )
            assert result.error_code == code


class TestSubAgentSignature:
    """Tests for the SubAgentSignature."""

    def test_signature_has_required_fields(self):
        """Test that signature has all required fields."""
        fields = SubAgentSignature.model_fields

        # Check input fields
        assert "task" in fields
        assert "context" in fields

        # Check output fields
        assert "result" in fields

    def test_signature_field_types(self):
        """Test that signature fields have correct types."""
        fields = SubAgentSignature.model_fields

        assert fields["task"].annotation == str
        assert fields["context"].annotation == str
        assert fields["result"].annotation == str

    def test_signature_context_has_default(self):
        """Test that context field has a default value."""
        fields = SubAgentSignature.model_fields

        # Context should have a default empty string
        assert fields["context"].default == ""


class TestMCPSubAgent:
    """Tests for the MCPSubAgent class."""

    def test_subagent_initialization(self):
        """Test sub-agent initializes correctly."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        subagent = MCPSubAgent(
            mcp_config=EXPENSE_MANAGER_CONFIG,
            user_id="user123",
            max_iters=4,
        )

        assert subagent.mcp_config == EXPENSE_MANAGER_CONFIG
        assert subagent.user_id == "user123"
        assert subagent.max_iters == 4

    def test_subagent_default_max_iters(self):
        """Test sub-agent has default max_iters of 4."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        subagent = MCPSubAgent(
            mcp_config=EXPENSE_MANAGER_CONFIG,
            user_id="user123",
        )

        assert subagent.max_iters == 4

    @pytest.mark.asyncio
    async def test_subagent_execute_success(self):
        """Test successful sub-agent execution."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        # Mock the MCP session and tools
        mock_session = AsyncMock()
        mock_tools = [MagicMock(name="create_expense")]

        # Mock ReAct result
        mock_react_result = MagicMock()
        mock_react_result.result = "Expense created: ID 123"
        mock_react_result.trajectory = []

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session, patch(
            "app.infrastructure.llm.subagent.create_mcp_tools_for_user"
        ) as mock_create_tools, patch(
            "dspy.ReAct"
        ) as mock_react_class:
            # Setup mocks
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_create_tools.return_value = mock_tools

            mock_react = MagicMock()
            mock_react.acall = AsyncMock(return_value=mock_react_result)
            mock_react_class.return_value = mock_react

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            result = await subagent.execute(task="Create an expense for $50 lunch")

            assert result.success is True
            assert result.result == "Expense created: ID 123"
            assert result.error is None
            assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_subagent_execute_connection_failed(self):
        """Test sub-agent handles connection failure."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session:
            # Simulate connection failure
            mock_get_session.return_value.__aenter__ = AsyncMock(
                side_effect=RuntimeError("Connection refused")
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            result = await subagent.execute(task="Create an expense")

            assert result.success is False
            assert result.error_code == "CONNECTION_FAILED"
            assert "Connection" in result.error or "connection" in result.error

    @pytest.mark.asyncio
    async def test_subagent_execute_file_not_found(self):
        """Test sub-agent handles missing server file."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session:
            # Simulate file not found
            mock_get_session.return_value.__aenter__ = AsyncMock(
                side_effect=FileNotFoundError("Server script not found")
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            result = await subagent.execute(task="Create an expense")

            assert result.success is False
            assert result.error_code == "CONNECTION_FAILED"
            assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_subagent_execute_timeout(self):
        """Test sub-agent handles timeout."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session:
            # Simulate timeout
            mock_get_session.return_value.__aenter__ = AsyncMock(
                side_effect=TimeoutError("Operation timed out")
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            result = await subagent.execute(task="Create an expense")

            assert result.success is False
            assert result.error_code == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_subagent_execute_generic_error(self):
        """Test sub-agent handles generic errors."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session:
            # Simulate generic error
            mock_get_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            result = await subagent.execute(task="Create an expense")

            assert result.success is False
            assert result.error_code == "EXECUTION_ERROR"

    @pytest.mark.asyncio
    async def test_subagent_execute_with_context(self):
        """Test sub-agent passes context to ReAct."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        mock_session = AsyncMock()
        mock_tools = [MagicMock(name="create_expense")]

        mock_react_result = MagicMock()
        mock_react_result.result = "Done"
        mock_react_result.trajectory = []

        with patch(
            "app.infrastructure.llm.subagent.get_mcp_session"
        ) as mock_get_session, patch(
            "app.infrastructure.llm.subagent.create_mcp_tools_for_user"
        ) as mock_create_tools, patch(
            "dspy.ReAct"
        ) as mock_react_class:
            mock_get_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session
            )
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=None)
            mock_create_tools.return_value = mock_tools

            mock_react = MagicMock()
            mock_react.acall = AsyncMock(return_value=mock_react_result)
            mock_react_class.return_value = mock_react

            subagent = MCPSubAgent(
                mcp_config=EXPENSE_MANAGER_CONFIG,
                user_id="user123",
            )

            await subagent.execute(
                task="Create an expense",
                context="User mentioned it was for a team lunch",
            )

            # Verify context was passed to acall
            mock_react.acall.assert_called_once()
            call_kwargs = mock_react.acall.call_args[1]
            assert call_kwargs["context"] == "User mentioned it was for a team lunch"
