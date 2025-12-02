"""Unit tests for MCP orchestration tools.

These tests verify the meta-tools used by the hierarchical agent:
- get_available_mcp_servers
- create_delegate_to_subagent_tool
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.llm.tools import (
    create_delegate_to_subagent_tool,
    get_available_mcp_servers,
)


class TestGetAvailableMCPServers:
    """Tests for the get_available_mcp_servers tool."""

    def test_returns_json_string(self):
        """Test that function returns a valid JSON string."""
        result = get_available_mcp_servers()

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_contains_all_servers(self):
        """Test that result contains all registered servers."""
        result = get_available_mcp_servers()
        parsed = json.loads(result)

        server_names = [s["name"] for s in parsed]

        assert "expense_manager" in server_names
        assert "note_manager" in server_names

    def test_each_server_has_required_fields(self):
        """Test that each server entry has required fields."""
        result = get_available_mcp_servers()
        parsed = json.loads(result)

        for server in parsed:
            assert "name" in server
            assert "description" in server
            assert "use_when" in server

    def test_descriptions_are_helpful(self):
        """Test that descriptions provide useful information."""
        result = get_available_mcp_servers()
        parsed = json.loads(result)

        for server in parsed:
            # Description should be non-empty and meaningful
            assert len(server["description"]) > 10
            # Use_when should be non-empty
            assert len(server["use_when"]) > 10


class TestCreateDelegateToSubagentTool:
    """Tests for the create_delegate_to_subagent_tool factory."""

    def test_factory_returns_callable(self):
        """Test that factory returns a callable function."""
        tool = create_delegate_to_subagent_tool("user123")

        assert callable(tool)

    def test_tool_has_correct_name(self):
        """Test that the returned tool has the expected name."""
        tool = create_delegate_to_subagent_tool("user123")

        assert tool.__name__ == "delegate_to_subagent"

    def test_tool_has_docstring(self):
        """Test that the returned tool has a docstring."""
        tool = create_delegate_to_subagent_tool("user123")

        assert tool.__doc__ is not None
        assert len(tool.__doc__) > 0
        assert "delegate" in tool.__doc__.lower()

    @pytest.mark.asyncio
    async def test_delegate_to_unknown_server(self):
        """Test delegation to an unknown server returns error."""
        tool = create_delegate_to_subagent_tool("user123")

        result = await tool(
            server_name="nonexistent_server",
            task="Do something",
        )

        assert "Error" in result
        assert "Unknown server" in result
        assert "nonexistent_server" in result
        # Should list available servers
        assert "expense_manager" in result

    @pytest.mark.asyncio
    async def test_delegate_to_expense_manager_success(self):
        """Test successful delegation to expense_manager."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(
            success=True,
            result="Created expense #123 for $50",
            duration_ms=200.0,
        )

        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(return_value=mock_result)
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("user123")
            result = await tool(
                server_name="expense_manager",
                task="Create a $50 lunch expense",
            )

            assert "Created expense #123" in result
            mock_subagent_class.assert_called_once()
            mock_subagent.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegate_with_context(self):
        """Test delegation passes context to sub-agent."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(success=True, result="Done")

        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(return_value=mock_result)
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("user123")
            await tool(
                server_name="expense_manager",
                task="Create expense",
                context="This is additional context",
            )

            # Verify context was passed
            mock_subagent.execute.assert_called_once_with(
                task="Create expense",
                context="This is additional context",
            )

    @pytest.mark.asyncio
    async def test_delegate_failure_returns_error(self):
        """Test that delegation failure returns error message."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(
            success=False,
            error="Connection to MCP server failed",
            error_code="CONNECTION_FAILED",
        )

        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(return_value=mock_result)
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("user123")
            result = await tool(
                server_name="expense_manager",
                task="Create expense",
            )

            assert "Error" in result
            assert "expense_manager" in result
            assert "Connection" in result

    @pytest.mark.asyncio
    async def test_delegate_exception_returns_error(self):
        """Test that exceptions are caught and returned as errors."""
        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(
                side_effect=Exception("Unexpected error")
            )
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("user123")
            result = await tool(
                server_name="expense_manager",
                task="Create expense",
            )

            assert "Error" in result
            assert "expense_manager" in result

    @pytest.mark.asyncio
    async def test_delegate_uses_correct_user_id(self):
        """Test that the tool uses the bound user_id."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(success=True, result="Done")

        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(return_value=mock_result)
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("specific_user_456")
            await tool(server_name="expense_manager", task="Test")

            # Verify user_id was passed to MCPSubAgent
            call_kwargs = mock_subagent_class.call_args[1]
            assert call_kwargs["user_id"] == "specific_user_456"

    @pytest.mark.asyncio
    async def test_delegate_to_all_servers(self):
        """Test that delegation works for all registered servers."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(success=True, result="Done")

        servers = [
            "expense_manager",
            "note_manager",
        ]

        for server in servers:
            with patch(
                "app.infrastructure.llm.subagent.MCPSubAgent"
            ) as mock_subagent_class:
                mock_subagent = MagicMock()
                mock_subagent.execute = AsyncMock(return_value=mock_result)
                mock_subagent_class.return_value = mock_subagent

                tool = create_delegate_to_subagent_tool("user123")
                result = await tool(server_name=server, task="Test task")

                assert "Done" in result, f"Failed for server: {server}"

    @pytest.mark.asyncio
    async def test_delegate_success_with_empty_result(self):
        """Test delegation handles success with None result."""
        from app.infrastructure.llm.subagent import SubAgentResult

        mock_result = SubAgentResult(
            success=True,
            result=None,  # No result message
        )

        with patch(
            "app.infrastructure.llm.subagent.MCPSubAgent"
        ) as mock_subagent_class:
            mock_subagent = MagicMock()
            mock_subagent.execute = AsyncMock(return_value=mock_result)
            mock_subagent_class.return_value = mock_subagent

            tool = create_delegate_to_subagent_tool("user123")
            result = await tool(
                server_name="expense_manager",
                task="Create expense",
            )

            # Should return default success message
            assert "Task completed successfully" in result


class TestDelegateToolIntegration:
    """Integration-style tests for the delegate tool."""

    @pytest.mark.asyncio
    async def test_delegate_uses_correct_config(self):
        """Test that delegate uses the correct MCP config for each server."""
        from app.infrastructure.llm.subagent import SubAgentResult
        from app.infrastructure.mcp.client import (
            EXPENSE_MANAGER_CONFIG,
            NOTE_MANAGER_CONFIG,
        )

        mock_result = SubAgentResult(success=True, result="Done")

        server_configs = {
            "expense_manager": EXPENSE_MANAGER_CONFIG,
            "note_manager": NOTE_MANAGER_CONFIG,
        }

        for server_name, expected_config in server_configs.items():
            with patch(
                "app.infrastructure.llm.subagent.MCPSubAgent"
            ) as mock_subagent_class:
                mock_subagent = MagicMock()
                mock_subagent.execute = AsyncMock(return_value=mock_result)
                mock_subagent_class.return_value = mock_subagent

                tool = create_delegate_to_subagent_tool("user123")
                await tool(server_name=server_name, task="Test")

                # Verify correct config was used
                call_kwargs = mock_subagent_class.call_args[1]
                assert call_kwargs["mcp_config"] == expected_config, (
                    f"Wrong config for {server_name}"
                )
