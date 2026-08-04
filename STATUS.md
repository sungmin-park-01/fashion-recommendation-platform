# Project Status

## Current stage

Stage 1 — repository and development environment

## Stage 1 checklist

- [ ] `uv sync` completes successfully
- [ ] VS Code selects `.venv/bin/python`
- [ ] `uv run pytest -m "not integration"` passes
- [ ] `uv run ruff check .` passes
- [ ] `uv run hm-recsys check` finds all three CSV files
- [ ] Docker Desktop memory is set to 6–8 GB
- [ ] `docker compose config` succeeds
- [ ] `uv.lock` is committed
- [ ] Stage 1 Git commit is created

## Next stage

Stage 2 — convert the raw CSV files to typed Parquet files with Polars lazy scanning.
