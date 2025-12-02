"""Unit tests for the MCP Server Registry.

These tests verify the registry correctly stores and retrieves
MCP server metadata for hierarchical agent orchestration.
"""

import json

import pytest

from app.infrastructure.mcp.registry import (
    MCP_SERVER_REGISTRY,
    MCPServerMetadata,
    get_all_server_names,
    get_server_metadata,
    get_servers_summary,
)


class TestMCPServerMetadata:
    """Tests for the MCPServerMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata with required fields."""
        from app.infrastructure.mcp.client import EXPENSE_MANAGER_CONFIG

        metadata = MCPServerMetadata(
            name="test_server",
            config=EXPENSE_MANAGER_CONFIG,
            description="Test server for unit tests",
        )

        assert metadata.name == "test_server"
        assert metadata.config == EXPENSE_MANAGER_CONFIG
        assert metadata.description == "Test server for unit tests"
        assert metadata.capabilities == []
        assert metadata.use_when == ""

    def test_metadata_with_all_fields(self):
        """Test creating metadata with all fields."""
        from app.infrastructure.mcp.client import NOTE_MANAGER_CONFIG

        metadata = MCPServerMetadata(
            name="note_manager",
            config=NOTE_MANAGER_CONFIG,
            description="Manage notes",
            capabilities=["create", "read", "update", "delete"],
            use_when="User wants to manage notes",
        )

        assert metadata.name == "note_manager"
        assert metadata.capabilities == ["create", "read", "update", "delete"]
        assert metadata.use_when == "User wants to manage notes"


class TestMCPServerRegistry:
    """Tests for the MCP_SERVER_REGISTRY."""

    def test_registry_contains_expected_servers(self):
        """Test that registry contains all expected servers."""
        expected_servers = [
            "expense_manager",
            "note_manager",
        ]

        for server_name in expected_servers:
            assert server_name in MCP_SERVER_REGISTRY, (
                f"Server {server_name} not found in registry"
            )

    def test_registry_entries_have_required_fields(self):
        """Test that all registry entries have required fields."""
        for name, metadata in MCP_SERVER_REGISTRY.items():
            assert isinstance(metadata, MCPServerMetadata)
            assert metadata.name == name
            assert metadata.config is not None
            assert len(metadata.description) > 0
            assert len(metadata.capabilities) > 0
            assert len(metadata.use_when) > 0

    def test_expense_manager_metadata(self):
        """Test expense_manager metadata is correct."""
        metadata = MCP_SERVER_REGISTRY["expense_manager"]

        assert metadata.name == "expense_manager"
        assert "expense" in metadata.description.lower()
        assert "create_expense" in metadata.capabilities
        assert "list_expenses" in metadata.capabilities

    def test_note_manager_metadata(self):
        """Test note_manager metadata is correct."""
        metadata = MCP_SERVER_REGISTRY["note_manager"]

        assert metadata.name == "note_manager"
        assert "note" in metadata.description.lower()
        assert "create_note" in metadata.capabilities

class TestGetServerMetadata:
    """Tests for the get_server_metadata function."""

    def test_get_existing_server(self):
        """Test getting metadata for an existing server."""
        metadata = get_server_metadata("expense_manager")

        assert metadata is not None
        assert metadata.name == "expense_manager"

    def test_get_nonexistent_server(self):
        """Test getting metadata for a non-existent server returns None."""
        metadata = get_server_metadata("nonexistent_server")

        assert metadata is None

    def test_get_all_registered_servers(self):
        """Test that all registered servers can be retrieved."""
        for server_name in MCP_SERVER_REGISTRY.keys():
            metadata = get_server_metadata(server_name)
            assert metadata is not None
            assert metadata.name == server_name


class TestGetAllServerNames:
    """Tests for the get_all_server_names function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        names = get_all_server_names()
        assert isinstance(names, list)

    def test_contains_expected_servers(self):
        """Test that list contains expected server names."""
        names = get_all_server_names()

        assert "expense_manager" in names
        assert "note_manager" in names

    def test_count_matches_registry(self):
        """Test that count matches registry entries."""
        names = get_all_server_names()
        assert len(names) == len(MCP_SERVER_REGISTRY)


class TestGetServersSummary:
    """Tests for the get_servers_summary function."""

    def test_returns_valid_json(self):
        """Test that function returns valid JSON string."""
        summary = get_servers_summary()

        # Should not raise
        parsed = json.loads(summary)
        assert isinstance(parsed, list)

    def test_summary_contains_all_servers(self):
        """Test that summary contains all registered servers."""
        summary = get_servers_summary()
        parsed = json.loads(summary)

        server_names = [s["name"] for s in parsed]

        for name in MCP_SERVER_REGISTRY.keys():
            assert name in server_names

    def test_summary_entry_structure(self):
        """Test that each summary entry has expected fields."""
        summary = get_servers_summary()
        parsed = json.loads(summary)

        for entry in parsed:
            assert "name" in entry
            assert "description" in entry
            assert "use_when" in entry

    def test_summary_matches_registry(self):
        """Test that summary data matches registry data."""
        summary = get_servers_summary()
        parsed = json.loads(summary)

        for entry in parsed:
            name = entry["name"]
            metadata = MCP_SERVER_REGISTRY[name]

            assert entry["description"] == metadata.description
            assert entry["use_when"] == metadata.use_when
