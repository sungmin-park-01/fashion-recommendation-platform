#!/usr/bin/env bash
set -euo pipefail

command -v uv >/dev/null 2>&1 || {
  echo "uv is not installed. Install it first:"
  echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
}

uv python install 3.11
uv sync
cp -n .env.example .env || true
uv run ruff check .
uv run pytest -m "not integration"
uv run hm-recsys check

echo "Bootstrap complete. Put the three H&M CSV files in data/raw/."
