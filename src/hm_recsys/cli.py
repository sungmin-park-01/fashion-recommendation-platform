from pathlib import Path
from typing import Annotated

import typer

from hm_recsys.config import load_config
from hm_recsys.data.ingest import ingest_raw_csvs

app = typer.Typer(
    no_args_is_help=True,
    help="H&M recommendation system utilities",
)


@app.callback()
def main() -> None:
    """H&M recommendation system command-line tools."""


@app.command()
def check(config: Path = Path("configs/data_small.yaml")) -> None:
    """Validate the environment, configuration, and expected raw-data names."""
    loaded = load_config(config)
    raw_dir = Path(loaded["paths"]["raw_dir"])
    expected = ["articles.csv", "customers.csv", "transactions_train.csv"]

    typer.echo(f"Config: {config}")
    typer.echo(f"Sample users: {loaded['sample']['users']}")
    typer.echo(f"Candidate K: {loaded['ranking']['candidates_per_user']}")
    typer.echo("Raw data files:")
    for filename in expected:
        state = "FOUND" if (raw_dir / filename).exists() else "MISSING"
        typer.echo(f"  [{state}] {raw_dir / filename}")


@app.command()
def ingest(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="YAML configuration path.",
        ),
    ] = Path("configs/base.yaml"),
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Replace existing Bronze Parquet files.",
        ),
    ] = False,
) -> None:
    """Convert raw H&M CSV files into typed Bronze Parquet files."""
    manifest_path = ingest_raw_csvs(config, overwrite=overwrite)
    typer.echo(f"Bronze ingest completed: {manifest_path}")


if __name__ == "__main__":
    app()
