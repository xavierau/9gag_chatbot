"""Unit tests for the ChatBotAgent.

These tests verify the agent's behavior using mocked dependencies
to ensure reliable, fast test execution.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.llm.agent import (
    AgentResponse,
    ChatBotAgent,
    ConversationContext,
    SimpleChatBotAgent,
)
from app.infrastructure.llm.signatures import ChatBotSignature
from app.infrastructure.llm.tools import (
    calculate,
    create_memory_list_tool,
    create_memory_search_tool,
    create_memory_store_tool,
    get_current_time,
)


class TestConversationContext:
    """Tests for the ConversationContext dataclass."""

    def test_conversation_context_creation(self):
        """Test creating a ConversationContext with required fields."""
        ctx = ConversationContext(user_id="user123")
        assert ctx.user_id == "user123"
        assert ctx.session_id is None
        assert ctx.conversation_history == ""
        assert ctx.memory_context == ""

    def test_conversation_context_with_all_fields(self):
        """Test creating a ConversationContext with all fields."""
        ctx = ConversationContext(
            user_id="user123",
            session_id="session456",
            conversation_history="User: Hello\nAssistant: Hi!",
            memory_context="User prefers formal language",
        )
        assert ctx.user_id == "user123"
        assert ctx.session_id == "session456"
        assert ctx.conversation_history == "User: Hello\nAssistant: Hi!"
        assert ctx.memory_context == "User prefers formal language"


class TestAgentResponse:
    """Tests for the AgentResponse dataclass."""

    def test_agent_response_minimal(self):
        """Test creating an AgentResponse with required fields."""
        response = AgentResponse(response="Hello, how can I help?")
        assert response.response == "Hello, how can I help?"
        assert response.should_store_memory is False
        assert response.memory_to_store == ""
        assert response.reasoning_trace is None

    def test_agent_response_with_memory(self):
        """Test creating an AgentResponse with memory fields."""
        response = AgentResponse(
            response="I will remember that.",
            should_store_memory=True,
            memory_to_store="User prefers Python over JavaScript",
        )
        assert response.should_store_memory is True
        assert response.memory_to_store == "User prefers Python over JavaScript"

    def test_agent_response_with_trace(self):
        """Test creating an AgentResponse with reasoning trace."""
        trace = ["Step 1: Analyze query", "Step 2: Search memories"]
        response = AgentResponse(response="Test", reasoning_trace=trace)
        assert response.reasoning_trace == trace


class TestTools:
    """Tests for the agent tools."""

    def test_get_current_time(self):
        """Test that get_current_time returns a formatted string."""
        result = get_current_time()
        assert isinstance(result, str)
        # Should contain date components
        assert any(day in result for day in [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ])

    def test_calculate_simple_addition(self):
        """Test simple addition calculation."""
        result = calculate("2 + 2")
        assert result == "Result: 4"

    def test_calculate_complex_expression(self):
        """Test complex expression calculation."""
        result = calculate("(10 + 5) * 2")
        assert result == "Result: 30"

    def test_calculate_division(self):
        """Test division calculation."""
        result = calculate("10 / 4")
        assert result == "Result: 2.5"

    def test_calculate_invalid_characters(self):
        """Test that invalid characters are rejected."""
        result = calculate("import os; os.system('ls')")
        assert "Invalid characters" in result

    def test_calculate_malicious_input(self):
        """Test that malicious input is rejected."""
        result = calculate("__import__('os').system('ls')")
        assert "Invalid characters" in result


class TestMemoryTools:
    """Tests for memory-related tools."""

    def test_memory_search_tool_returns_results(self):
        """Test memory search returns formatted results."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"memory": "User likes Python"},
                {"memory": "User is a developer"},
            ]
        }

        search_tool = create_memory_search_tool(mock_memory, "user123")
        result = search_tool("programming languages")

        assert "User likes Python" in result
        assert "User is a developer" in result
        mock_memory.search.assert_called_once()

    def test_memory_search_tool_no_results(self):
        """Test memory search with no results."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        search_tool = create_memory_search_tool(mock_memory, "user123")
        result = search_tool("nonexistent topic")

        assert "No relevant memories found" in result

    def test_memory_search_tool_handles_error(self):
        """Test memory search handles exceptions gracefully."""
        mock_memory = MagicMock()
        mock_memory.search.side_effect = Exception("Connection error")

        search_tool = create_memory_search_tool(mock_memory, "user123")
        result = search_tool("test query")

        assert "Error searching memories" in result

    def test_memory_store_tool_success(self):
        """Test memory store tool stores successfully."""
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"id": "mem123"}

        store_tool = create_memory_store_tool(mock_memory, "user123")
        result = store_tool("User prefers dark mode")

        assert "Memory stored successfully" in result
        mock_memory.add.assert_called_once()

    def test_memory_store_tool_with_category(self):
        """Test memory store tool with category."""
        mock_memory = MagicMock()

        store_tool = create_memory_store_tool(mock_memory, "user123")
        result = store_tool("User prefers dark mode", category="preference")

        assert "Memory stored successfully" in result
        # Verify category was passed in metadata
        call_kwargs = mock_memory.add.call_args[1]
        assert call_kwargs.get("metadata", {}).get("category") == "preference"

    def test_memory_list_tool_returns_all(self):
        """Test memory list tool returns all memories."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {
            "results": [
                {"memory": "Memory 1"},
                {"memory": "Memory 2"},
                {"memory": "Memory 3"},
            ]
        }

        list_tool = create_memory_list_tool(mock_memory, "user123")
        result = list_tool()

        assert "Memory 1" in result
        assert "Memory 2" in result
        assert "Memory 3" in result

    def test_memory_list_tool_empty(self):
        """Test memory list tool with no memories."""
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {"results": []}

        list_tool = create_memory_list_tool(mock_memory, "user123")
        result = list_tool()

        assert "No memories stored" in result


class TestChatBotAgent:
    """Tests for the ChatBotAgent class."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(memory_client=mock_memory, max_iters=5)

        assert agent.memory_client == mock_memory
        assert agent.max_iters == 5
        assert agent.include_reasoning_trace is False

    def test_agent_initialization_with_trace(self):
        """Test agent initializes with reasoning trace enabled."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(
            memory_client=mock_memory, include_reasoning_trace=True
        )

        assert agent.include_reasoning_trace is True

    def test_format_conversation_history_empty(self):
        """Test formatting empty conversation history."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(memory_client=mock_memory)

        result = agent._format_conversation_history([])
        assert result == "No previous conversation history."

    def test_format_conversation_history_with_messages(self):
        """Test formatting conversation history with messages."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(memory_client=mock_memory)

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = agent._format_conversation_history(history)

        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: How are you?" in result

    def test_format_conversation_history_truncation(self):
        """Test that history is truncated to max_messages."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(memory_client=mock_memory)

        # Create 15 messages
        history = [{"role": "user", "content": f"Message {i}"} for i in range(15)]

        result = agent._format_conversation_history(history, max_messages=5)

        # Should only contain last 5 messages (10-14)
        assert "Message 10" in result
        assert "Message 14" in result
        assert "Message 5" not in result

    def test_create_tools_for_user(self):
        """Test that user-specific tools are created."""
        mock_memory = MagicMock()
        agent = ChatBotAgent(memory_client=mock_memory)

        tools = agent._create_tools_for_user("user123")

        # Should have base tools + 3 memory tools
        assert len(tools) >= 5  # get_current_time, calculate, search, store, list

    def test_retrieve_memory_context_with_results(self):
        """Test memory context retrieval with results."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"memory": "User is a software engineer"},
                {"memory": "User prefers concise answers"},
            ]
        }

        agent = ChatBotAgent(memory_client=mock_memory)
        result = agent._retrieve_memory_context("user123", "Hello")

        assert "User is a software engineer" in result
        assert "User prefers concise answers" in result

    def test_retrieve_memory_context_no_results(self):
        """Test memory context retrieval with no results."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        agent = ChatBotAgent(memory_client=mock_memory)
        result = agent._retrieve_memory_context("user123", "Hello")

        assert "No relevant memories" in result

    def test_retrieve_memory_context_handles_error(self):
        """Test memory context retrieval handles errors."""
        mock_memory = MagicMock()
        mock_memory.search.side_effect = Exception("Database error")

        agent = ChatBotAgent(memory_client=mock_memory)
        result = agent._retrieve_memory_context("user123", "Hello")

        assert "Memory retrieval unavailable" in result


class TestSimpleChatBotAgent:
    """Tests for the SimpleChatBotAgent class."""

    def test_simple_agent_initialization(self):
        """Test simple agent initializes correctly."""
        mock_memory = MagicMock()

        with patch("dspy.ChainOfThought"):
            agent = SimpleChatBotAgent(memory_client=mock_memory)
            assert agent.memory_client == mock_memory


class TestChatBotSignature:
    """Tests for the ChatBotSignature."""

    def test_signature_has_required_fields(self):
        """Test that signature has all required fields."""
        # DSPy signatures store fields in model_fields
        fields = ChatBotSignature.model_fields

        # Check input fields
        assert "user_message" in fields
        assert "conversation_history" in fields
        assert "memory_context" in fields
        assert "user_id" in fields

        # Check output fields
        assert "response" in fields
        assert "should_store_memory" in fields
        assert "memory_to_store" in fields

    def test_signature_field_types(self):
        """Test that signature fields have correct types."""
        fields = ChatBotSignature.model_fields

        # Input fields should be strings
        assert fields["user_message"].annotation == str
        assert fields["conversation_history"].annotation == str
        assert fields["memory_context"].annotation == str
        assert fields["user_id"].annotation == str

        # Output fields
        assert fields["response"].annotation == str
        assert fields["should_store_memory"].annotation == bool
        assert fields["memory_to_store"].annotation == str

    def test_signature_field_descriptions(self):
        """Test that signature fields have descriptions."""
        fields = ChatBotSignature.model_fields

        # All fields should have descriptions
        for field_name, field_info in fields.items():
            json_extra = field_info.json_schema_extra or {}
            assert "desc" in json_extra, f"Field {field_name} missing description"

    def test_signature_input_output_classification(self):
        """Test that fields are correctly classified as input or output."""
        fields = ChatBotSignature.model_fields

        input_fields = ["user_message", "conversation_history", "memory_context", "user_id"]
        output_fields = ["response", "should_store_memory", "memory_to_store"]

        for field_name in input_fields:
            json_extra = fields[field_name].json_schema_extra or {}
            assert json_extra.get("__dspy_field_type") == "input", (
                f"Field {field_name} should be input"
            )

        for field_name in output_fields:
            json_extra = fields[field_name].json_schema_extra or {}
            assert json_extra.get("__dspy_field_type") == "output", (
                f"Field {field_name} should be output"
            )
