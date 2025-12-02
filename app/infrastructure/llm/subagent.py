"""MCP Sub-agent for hierarchical agent orchestration.

This module provides the MCPSubAgent class that executes tasks using
a single MCP server's tools. Sub-agents are spawned on-demand by the
main ChatBotAgent when it needs to delegate specialized tasks.

Architecture:
    ChatBotAgent (orchestrator) -> delegate_to_subagent() -> MCPSubAgent
                                                                |
                                                                v
                                                          MCP Server tools

The sub-agent uses DSPy ReAct for reasoning and tool execution,
but with a focused tool set from a single MCP server.
"""

import logging
import time
from dataclasses import dataclass

import dspy

from app.infrastructure.llm.dspy_config import get_tracer
from app.infrastructure.mcp.client import (
    MCPServerConfig,
    create_mcp_tools_for_user,
    get_mcp_session,
)

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution.

    Attributes:
        success: Whether the task completed successfully
        result: The result string from the sub-agent (if successful)
        error: Error message (if failed)
        error_code: Error classification for metrics
        duration_ms: Execution time in milliseconds
        iterations: Number of ReAct iterations used
        tools_called: List of tool names that were invoked
    """

    success: bool
    result: str | None = None
    error: str | None = None
    error_code: str | None = None  # "CONNECTION_FAILED", "EXECUTION_ERROR", "TIMEOUT"
    duration_ms: float = 0
    iterations: int = 0
    tools_called: list[str] | None = None

    def __post_init__(self) -> None:
        if self.tools_called is None:
            self.tools_called = []


class SubAgentSignature(dspy.Signature):
    """Signature for sub-agent task execution.

    Sub-agents receive a focused task from the orchestrator and execute it
    using their specialized MCP tools. They should complete the task and
    return a clear, concise result.
    """

    task: str = dspy.InputField(
        desc="The specific task to accomplish using the available tools. "
        "This task has been delegated by the main agent."
    )
    context: str = dspy.InputField(
        desc="Additional context from the conversation that may be relevant "
        "to completing the task.",
        default="",
    )

    result: str = dspy.OutputField(
        desc="A clear, concise summary of what was accomplished. "
        "Include relevant details like IDs, values, or confirmations. "
        "If the task failed, explain why."
    )


class MCPSubAgent(dspy.Module):
    """Sub-agent specialized for a single MCP server.

    This agent is spawned on-demand by the orchestrator (ChatBotAgent)
    to handle tasks that require a specific MCP server's capabilities.

    The sub-agent:
    1. Connects to a single MCP server
    2. Loads only that server's tools
    3. Uses ReAct to execute the delegated task
    4. Returns a structured result to the orchestrator

    Attributes:
        mcp_config: Configuration for the MCP server to connect to
        user_id: User ID for MCP operations (set via environment)
        max_iters: Maximum ReAct iterations (default: 4, fewer than orchestrator)
    """

    def __init__(
        self,
        mcp_config: MCPServerConfig,
        user_id: str,
        max_iters: int = 4,
    ):
        """Initialize the sub-agent.

        Args:
            mcp_config: MCP server configuration
            user_id: User ID for this execution
            max_iters: Maximum ReAct iterations (default: 4)
        """
        super().__init__()
        self.mcp_config = mcp_config
        self.user_id = user_id
        self.max_iters = max_iters
        self._signature = SubAgentSignature

    async def execute(self, task: str, context: str = "") -> SubAgentResult:
        """Execute a task using this MCP server's tools.

        This method:
        1. Opens a connection to the MCP server
        2. Loads the server's tools
        3. Creates a ReAct agent with those tools
        4. Executes the task
        5. Returns a structured result

        Args:
            task: The task description to execute
            context: Additional context for the task

        Returns:
            SubAgentResult with success status and result/error
        """
        start_time = time.time()
        tools_called: list[str] = []

        # Get tracer for custom spans
        tracer = get_tracer()

        try:
            logger.info(
                f"SubAgent starting task for {self.mcp_config.name}: {task[:100]}..."
            )

            async with get_mcp_session(self.mcp_config, self.user_id) as session:
                # Load tools from this MCP server
                tools = await create_mcp_tools_for_user(
                    self.mcp_config, self.user_id, session
                )

                logger.debug(
                    f"SubAgent loaded {len(tools)} tools from {self.mcp_config.name}"
                )

                # Create ReAct agent with the loaded tools
                react = dspy.ReAct(
                    signature=self._signature,
                    tools=tools,
                    max_iters=self.max_iters,
                )

                # Execute with tracing span if available
                if tracer:
                    with tracer.start_as_current_span(
                        "MCPSubAgent.execute",
                        attributes={
                            "agent.type": "subagent",
                            "agent.mcp_server": self.mcp_config.name,
                            "agent.user_id": self.user_id,
                            "agent.tools_count": len(tools),
                            "agent.max_iters": self.max_iters,
                            "input.task_length": len(task),
                            "input.context_length": len(context),
                        },
                    ) as span:
                        # Execute the task
                        result = await react.acall(task=task, context=context)

                        # Extract result
                        response = getattr(result, "result", str(result))

                        # Try to extract tools called from trajectory
                        iterations = 0
                        if hasattr(result, "trajectory"):
                            iterations = len(result.trajectory)
                            for step in result.trajectory:
                                if hasattr(step, "tool") and step.tool:
                                    tools_called.append(step.tool)

                        duration_ms = (time.time() - start_time) * 1000

                        # Add output attributes to span
                        span.set_attribute("output.success", True)
                        span.set_attribute("output.response_length", len(response))
                        span.set_attribute("output.iterations", iterations)
                        span.set_attribute("output.duration_ms", duration_ms)
                        span.set_attribute(
                            "output.tools_called", ",".join(tools_called)
                        )
                else:
                    # Execute the task without tracing
                    result = await react.acall(task=task, context=context)

                    # Extract result
                    response = getattr(result, "result", str(result))

                    # Try to extract tools called from trajectory
                    iterations = 0
                    if hasattr(result, "trajectory"):
                        iterations = len(result.trajectory)
                        for step in result.trajectory:
                            if hasattr(step, "tool") and step.tool:
                                tools_called.append(step.tool)

                    duration_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"SubAgent completed task for {self.mcp_config.name} "
                    f"in {duration_ms:.0f}ms with {iterations} iterations"
                )

                return SubAgentResult(
                    success=True,
                    result=response,
                    duration_ms=duration_ms,
                    iterations=iterations,
                    tools_called=tools_called,
                )

        except FileNotFoundError as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"MCP server not found: {e}"
            logger.error(f"SubAgent error for {self.mcp_config.name}: {error_msg}")
            return SubAgentResult(
                success=False,
                error=error_msg,
                error_code="CONNECTION_FAILED",
                duration_ms=duration_ms,
            )

        except RuntimeError as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"MCP connection failed: {e}"
            logger.error(f"SubAgent error for {self.mcp_config.name}: {error_msg}")
            return SubAgentResult(
                success=False,
                error=error_msg,
                error_code="CONNECTION_FAILED",
                duration_ms=duration_ms,
            )

        except TimeoutError as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Task timed out: {e}"
            logger.error(f"SubAgent timeout for {self.mcp_config.name}: {error_msg}")
            return SubAgentResult(
                success=False,
                error=error_msg,
                error_code="TIMEOUT",
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"Task execution failed: {e}"
            logger.error(
                f"SubAgent error for {self.mcp_config.name}: {error_msg}",
                exc_info=True,
            )
            return SubAgentResult(
                success=False,
                error=error_msg,
                error_code="EXECUTION_ERROR",
                duration_ms=duration_ms,
            )

    def forward(self, task: str, context: str = "") -> SubAgentResult:
        """Synchronous execution - not recommended for production.

        Use execute() (async) instead for proper async MCP handling.

        Args:
            task: The task description to execute
            context: Additional context for the task

        Returns:
            SubAgentResult with success status and result/error
        """
        import asyncio

        return asyncio.run(self.execute(task, context))
