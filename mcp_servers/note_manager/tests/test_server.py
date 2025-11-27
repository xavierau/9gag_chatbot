"""Tests for the note manager MCP server tools.

These tests use FastMCP's Client for testing tool functionality.
"""

import pytest
from fastmcp import Client

from note_manager.server import mcp


class TestNoteManagerServer:
    """Integration tests for the note manager MCP server."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that all expected tools are registered."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            # Verify all 5 CRUD tools are registered
            assert "create_note" in tool_names
            assert "get_note" in tool_names
            assert "list_notes" in tool_names
            assert "update_note" in tool_names
            assert "delete_note" in tool_names

    @pytest.mark.asyncio
    async def test_tool_descriptions(self):
        """Test that all tools have proper descriptions."""
        async with Client(mcp) as client:
            tools = await client.list_tools()

            for tool in tools:
                assert tool.description is not None
                assert len(tool.description) > 10, f"Tool {tool.name} has too short description"

    @pytest.mark.asyncio
    async def test_create_note_tool_schema(self):
        """Test create_note tool has correct input schema."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            create_note_tool = next((t for t in tools if t.name == "create_note"), None)

            assert create_note_tool is not None
            schema = create_note_tool.inputSchema

            # Check required properties
            assert "title" in schema["properties"]
            assert "content" in schema["properties"]
            assert schema["properties"]["title"]["type"] == "string"
            assert schema["properties"]["content"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_get_note_tool_schema(self):
        """Test get_note tool has correct input schema."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            get_note_tool = next((t for t in tools if t.name == "get_note"), None)

            assert get_note_tool is not None
            schema = get_note_tool.inputSchema

            # Check required properties
            assert "note_id" in schema["properties"]
            assert schema["properties"]["note_id"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_list_notes_tool_schema(self):
        """Test list_notes tool has correct input schema with semantic search support."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            list_notes_tool = next((t for t in tools if t.name == "list_notes"), None)

            assert list_notes_tool is not None
            schema = list_notes_tool.inputSchema

            # Check optional pagination properties
            assert "limit" in schema["properties"]
            assert "offset" in schema["properties"]
            assert schema["properties"]["limit"]["type"] == "integer"
            assert schema["properties"]["offset"]["type"] == "integer"

            # Check semantic search query parameter exists
            assert "query" in schema["properties"]

    @pytest.mark.asyncio
    async def test_update_note_tool_schema(self):
        """Test update_note tool has correct input schema."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            update_note_tool = next((t for t in tools if t.name == "update_note"), None)

            assert update_note_tool is not None
            schema = update_note_tool.inputSchema

            # Check properties
            assert "note_id" in schema["properties"]
            assert "title" in schema["properties"]
            assert "content" in schema["properties"]

    @pytest.mark.asyncio
    async def test_delete_note_tool_schema(self):
        """Test delete_note tool has correct input schema."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            delete_note_tool = next((t for t in tools if t.name == "delete_note"), None)

            assert delete_note_tool is not None
            schema = delete_note_tool.inputSchema

            # Check required properties
            assert "note_id" in schema["properties"]
            assert schema["properties"]["note_id"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_only_crud_tools_registered(self):
        """Test that only CRUD tools are registered (AI is integrated, not separate)."""
        async with Client(mcp) as client:
            tools = await client.list_tools()
            tool_names = [t.name for t in tools]

            # Verify only 5 CRUD tools are registered
            assert len(tool_names) == 5
            assert set(tool_names) == {"create_note", "get_note", "list_notes", "update_note", "delete_note"}

            # Verify old standalone AI tools are NOT registered
            assert "summarize_note" not in tool_names
            assert "semantic_search" not in tool_names
            assert "generate_embedding" not in tool_names
