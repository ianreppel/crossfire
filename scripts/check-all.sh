#!/bin/bash
set -e

echo "Running pre-commit checks..."
uv run pre-commit run --all-files

echo "Running tests with coverage..."
uv run pytest --cov=crossfire --cov-report=term-missing

echo "All checks passed! Ready to push."
