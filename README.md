# H&M Recommendation System Platform

A memory-bounded recommendation-system portfolio project using Polars, implicit ALS,
LightGBM ranking, FastAPI, Docker, PostgreSQL, MLflow, Prefect, and Prometheus.

## Required raw data

Place only these files in `data/raw/` for the first data pipeline:

- `articles.csv`
- `customers.csv`
- `transactions_train.csv`

H&M product images and `sample_submission.csv` are not needed for the initial collaborative-filtering pipeline.

## Local setup on macOS

```bash
brew install libomp
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || true

cd hm-recsys-platform
./scripts/bootstrap.sh
```

Then open the repository root in VS Code and select `.venv/bin/python` as the interpreter.

## Dependency groups

```bash
uv sync                 # core modeling + development tools
uv sync --group api     # FastAPI, PostgreSQL, Prometheus client
uv sync --group mlops   # MLflow and Prefect; install later
uv sync --group notebook # Jupyter; optional
uv sync --all-groups    # not recommended on a 24 GB laptop during normal work
```

## Validation commands

```bash
uv run hm-recsys check
uv run pytest -m "not integration"
uv run ruff check .
uv run mypy src
```

## API smoke test

```bash
uv sync --group api
uv run --group api uvicorn hm_recsys.api.main:app --reload --workers 1
curl http://127.0.0.1:8000/health
```

## Docker

Create the lock file locally before the first Docker build:

```bash
uv lock
cp .env.example .env
docker compose config
docker compose up --build api postgres
```

Open `http://localhost:8000/health` after both containers are healthy.

## Memory rules

- Normal target: 8–10 GB for the project
- Warning: 10–12 GB
- Reduce workload: 12–14 GB
- Abort the experiment: above 14 GB or sustained red memory pressure
- Run the API with one worker because every worker can load another model copy
