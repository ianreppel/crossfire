#!/bin/bash
# After-edit hook: runs docformatter + ruff + mypy (same as pre-commit hook)

input=$(cat)
file_path=$(echo "$input" | jq -r '.path // empty')

if [[ -z "$file_path" || "$file_path" != *.py ]]; then
  exit 0
fi

if [[ ! -f "$file_path" ]]; then
  exit 0
fi

uv run docformatter --wrap-summaries=120 --wrap-descriptions=120 --in-place "$file_path" 2>/dev/null
uv run ruff check --fix --quiet "$file_path" 2>/dev/null
uv run ruff format --quiet "$file_path" 2>/dev/null
uv run mypy --ignore-missing-imports --no-error-summary "$file_path" 2>/dev/null

exit 0
