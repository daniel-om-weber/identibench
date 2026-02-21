# CLAUDE.md

## Project Overview

identibench downloads and prepares various system identification benchmark datasets in a unified HDF5 format.

## Development Workflow

Edit `.py` files in `identibench/` directly. Tests are in `tests/` as pytest files.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Lint
ruff check identibench/

# Format
ruff format identibench/

# Install/sync dependencies (uses uv.lock)
uv sync --extra dev
```

## Environment

Use `uv sync --extra dev` to install dependencies — it creates/manages the `.venv` automatically and respects `uv.lock` for reproducible installs.

## Code Style

- Inline type hints on all public API signatures
- Google-style docstrings
- Module docstrings: one-liner stating what the module provides
