# MCP Server Logging to Stdout Causes JSONRPC Parse Errors

**Date:** 2025-12-22
**Status:** Fixed
**Severity:** Critical (Production Failure)

## Problem

The `expense_manager` MCP server was failing in production with JSONRPC validation errors:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for JSONRPCMessage
  Invalid JSON: trailing characters at line 1 column 5 [type=json_invalid,
  input_value='2025-12-22 03:38:26,656 ...ct pg_catalog.version()', input_type=str]
```

The MCP client was unable to parse messages from the server because SQLAlchemy SQL logs were being written to stdout, which is reserved for JSONRPC messages in the MCP protocol.

## Root Cause

In both `expense_manager` and `note_manager` MCP servers:

**mcp_servers/expense_manager/src/expense_manager/server.py**:
```python
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
```

**Problem:** `logging.basicConfig()` defaults to logging to stdout. In an MCP server context:
- Stdout is reserved exclusively for JSONRPC messages
- Any other output (logs, debug messages, SQL statements) must go to stderr

When SQLAlchemy executed queries with `echo=True` (or even with echo disabled at INFO level), the logs went to stdout and corrupted the JSONRPC message stream.

## Impact

- MCP server failed to respond to tool calls
- AI agent could not use expense management tools
- Production chatbot functionality broken

## Solution

Updated both MCP servers to explicitly configure logging to stderr:

**mcp_servers/expense_manager/src/expense_manager/server.py:31-36**:
```python
# Configure logging to stderr (MCP protocol requires stdout for JSONRPC messages only)
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # CRITICAL: Use stderr for MCP servers
)
```

**mcp_servers/note_manager/src/note_manager/server.py:40-45**: Same fix applied.

## Prevention

Added to **mcp_servers/CLAUDE.md** development guidelines:

### MCP Server Logging Requirements

**CRITICAL:** All MCP servers MUST configure logging to stderr:

```python
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # CRITICAL: stdout is reserved for JSONRPC
)
```

**Why:** The MCP protocol uses stdio transport where:
- **stdout** = JSONRPC messages only
- **stderr** = Logs, debug output, errors

Any logs to stdout will corrupt the JSONRPC stream and cause parse errors.

## Testing

After the fix:
1. Restart the production server
2. Verify MCP server starts without JSONRPC errors
3. Test expense creation tool call
4. Confirm SQL logs appear in stderr, not stdout
5. Confirm tool responses are valid JSON

## Related Files

- `mcp_servers/expense_manager/src/expense_manager/server.py:31-36`
- `mcp_servers/note_manager/src/note_manager/server.py:40-45`

## References

- [MCP Protocol Specification](https://modelcontextprotocol.io)
- [FastMCP Documentation](https://gofastmcp.com)
- Python logging: [basicConfig stream parameter](https://docs.python.org/3/library/logging.html#logging.basicConfig)
