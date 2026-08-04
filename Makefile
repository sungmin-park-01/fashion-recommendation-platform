.PHONY: install check test lint format typecheck api docker-up docker-down

install:
	uv sync

check:
	uv run hm-recsys check

lint:
	uv run ruff check .

test:
	uv run pytest -m "not integration"

format:
	uv run ruff format .

typecheck:
	uv run mypy src

api:
	uv run --group api uvicorn hm_recsys.api.main:app --reload --workers 1

docker-up:
	docker compose up --build api postgres

docker-down:
	docker compose down
