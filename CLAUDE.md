# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A chatbot application built with FastAPI, mem0 (memory layer), DSPy (LLM programming framework), and PostgreSQL.

## Development Commands

```bash
# Install dependencies (using uv - the project uses pyproject.toml)
uv sync

# Run the FastAPI development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run a single test file
pytest tests/test_example.py

# Run a specific test
pytest tests/test_example.py::test_function_name -v

# Type checking
mypy app/

# Linting
ruff check app/
ruff format app/

# Database migrations (using alembic)
alembic upgrade head
alembic revision --autogenerate -m "description"
```

## Architecture

### Intended Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── api/                 # API route handlers
│   └── routes/          # Route modules by domain
├── core/                # Configuration, settings, dependencies
├── domain/              # Business logic and domain models
├── infrastructure/      # External services (DB, mem0, DSPy)
│   ├── database/        # SQLAlchemy models, repositories
│   ├── memory/          # mem0 integration
│   └── llm/             # DSPy modules and signatures
└── schemas/             # Pydantic request/response models
tests/
├── unit/
├── integration/
└── conftest.py
```

### Key Integration Patterns

**mem0**: Used for conversation memory and context persistence. Configure with PostgreSQL as the vector store backend.

**DSPy**: Define signatures for LLM interactions. Use modules to compose complex reasoning chains. Leverage optimizers for prompt tuning.

**PostgreSQL**: Primary data store. Use SQLAlchemy with async support (asyncpg driver). mem0 can share the same PostgreSQL instance with pgvector extension.

### Dependency Injection

Use FastAPI's dependency injection system. Define dependencies in `app/core/dependencies.py` for database sessions, mem0 client, and DSPy modules.

## Configuration

Environment variables (use `.env` file):
- `DATABASE_URL`: PostgreSQL connection string
- `GOOGLE_API_KEY`: For Gemini API access
- `MEM0_CONFIG`: mem0 configuration (or configure programmatically)

### Default Models

- **LLM**: `gemini-2.5-flash` (Google Gemini)
- **Embeddings**: `gemini-embedding-001` (Google Gemini)

Configure DSPy with Gemini:
```python
import dspy
lm = dspy.LM("gemini/gemini-2.5-flash")
dspy.configure(lm=lm)
```

## Testing Strategy

- Unit tests: Mock external services (DB, LLM, mem0)
- Integration tests: Use test database, mock LLM responses
- Use pytest fixtures in `conftest.py` for shared setup
