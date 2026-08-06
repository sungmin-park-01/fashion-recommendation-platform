from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hm_recsys.config import PROJECT_ROOT, load_config

TRANSACTIONS_SCHEMA = {
    "t_dat": pl.String,
    "customer_id": pl.String,
    "article_id": pl.Int32,
    "price": pl.Float32,
    "sales_channel_id": pl.UInt8,
}

CUSTOMERS_SCHEMA = {
    "customer_id": pl.String,
    "FN": pl.Float32,
    "Active": pl.Float32,
    "club_member_status": pl.String,
    "fashion_news_frequency": pl.String,
    "age": pl.Float32,
    "postal_code": pl.String,
}

ARTICLES_SCHEMA = {
    "article_id": pl.Int32,
    "product_code": pl.Int32,
    "prod_name": pl.String,
    "product_type_no": pl.Int32,
    "product_type_name": pl.String,
    "product_group_name": pl.String,
    "graphical_appearance_no": pl.Int32,
    "graphical_appearance_name": pl.String,
    "colour_group_code": pl.Int32,
    "colour_group_name": pl.String,
    "perceived_colour_value_id": pl.Int32,
    "perceived_colour_value_name": pl.String,
    "perceived_colour_master_id": pl.Int32,
    "perceived_colour_master_name": pl.String,
    "department_no": pl.Int32,
    "department_name": pl.String,
    "index_code": pl.String,
    "index_name": pl.String,
    "index_group_no": pl.Int32,
    "index_group_name": pl.String,
    "section_no": pl.Int32,
    "section_name": pl.String,
    "garment_group_no": pl.Int32,
    "garment_group_name": pl.String,
    "detail_desc": pl.String,
}


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _scan_transactions(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            path,
            schema=TRANSACTIONS_SCHEMA,
            low_memory=True,
            rechunk=False,
        )
        .with_columns(
            pl.col("t_dat").str.strptime(pl.Date, "%Y-%m-%d", strict=True),
            pl.col("article_id").cast(pl.Int32),
            pl.col("price").cast(pl.Float32),
            pl.col("sales_channel_id").cast(pl.UInt8),
        )
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id",
        )
    )


def _scan_customers(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            path,
            schema=CUSTOMERS_SCHEMA,
            low_memory=True,
            rechunk=False,
        )
        .with_columns(
            pl.col("FN").cast(pl.UInt8, strict=False),
            pl.col("Active").cast(pl.UInt8, strict=False),
            pl.col("age").cast(pl.UInt8, strict=False),
        )
        .select(
            "customer_id",
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
            "postal_code",
        )
    )


def _scan_articles(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        schema=ARTICLES_SCHEMA,
        low_memory=True,
        rechunk=False,
    ).select(*ARTICLES_SCHEMA.keys())


def _count_rows(lazy_frame: pl.LazyFrame) -> int:
    result = lazy_frame.select(pl.len().alias("row_count")).collect(engine="streaming")
    return int(result.item())


def _serialize_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _profile_parquet(path: Path, date_column: str | None = None) -> dict[str, Any]:
    lazy_frame = pl.scan_parquet(path)
    schema = lazy_frame.collect_schema()

    expressions: list[pl.Expr] = [pl.len().alias("row_count")]
    expressions.extend(
        pl.col(column).null_count().alias(f"{column}__nulls") for column in schema.names()
    )

    if date_column is not None:
        expressions.extend(
            [
                pl.col(date_column).min().alias("date_min"),
                pl.col(date_column).max().alias("date_max"),
            ]
        )

    stats = lazy_frame.select(expressions).collect(engine="streaming").row(0, named=True)

    date_range: dict[str, str | None] | None = None
    if date_column is not None:
        date_range = {
            "min": _serialize_date(stats["date_min"]),
            "max": _serialize_date(stats["date_max"]),
        }

    return {
        "row_count": int(stats["row_count"]),
        "schema": {column: str(dtype) for column, dtype in schema.items()},
        "null_counts": {column: int(stats[f"{column}__nulls"]) for column in schema.names()},
        "date_range": date_range,
    }


def _write_parquet(
    lazy_frame: pl.LazyFrame,
    output_path: Path,
    *,
    overwrite: bool,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    temporary_path = output_path.with_suffix(".tmp.parquet")
    temporary_path.unlink(missing_ok=True)

    started_at = time.perf_counter()
    lazy_frame.sink_parquet(
        temporary_path,
        compression="zstd",
        compression_level=3,
        statistics=True,
        engine="streaming",
    )
    elapsed_seconds = time.perf_counter() - started_at

    temporary_path.replace(output_path)
    return elapsed_seconds


def _ingest_dataset(
    *,
    name: str,
    source_path: Path,
    output_path: Path,
    lazy_frame: pl.LazyFrame,
    overwrite: bool,
    date_column: str | None = None,
) -> dict[str, Any]:
    if not source_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {source_path}")

    print(f"[ingest] Counting rows: {_display_path(source_path)}")
    source_rows = _count_rows(lazy_frame)

    print(f"[ingest] Writing Parquet: {_display_path(output_path)}")
    elapsed_seconds = _write_parquet(lazy_frame, output_path, overwrite=overwrite)
    profile = _profile_parquet(output_path, date_column=date_column)
    output_rows = int(profile["row_count"])

    if source_rows != output_rows:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Row count mismatch for {name}: source={source_rows}, output={output_rows}"
        )

    source_size = source_path.stat().st_size
    output_size = output_path.stat().st_size

    print(
        f"[done] {name}: {output_rows:,} rows, "
        f"{output_size / (1024**2):.1f} MiB, {elapsed_seconds:.1f}s"
    )

    return {
        "name": name,
        "source_path": _display_path(source_path),
        "output_path": _display_path(output_path),
        "source_rows": source_rows,
        "output_rows": output_rows,
        "source_size_bytes": source_size,
        "output_size_bytes": output_size,
        "output_to_source_size_ratio": round(output_size / source_size, 4),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "schema": profile["schema"],
        "null_counts": profile["null_counts"],
        "date_range": profile["date_range"],
    }


def ingest_raw_csvs(config_path: str | Path, *, overwrite: bool = False) -> Path:
    """Convert the three H&M raw CSV files into typed Bronze Parquet files."""
    config = load_config(config_path)
    paths = config["paths"]

    raw_dir = _resolve_project_path(str(paths["raw_dir"]))
    bronze_dir = _resolve_project_path(str(paths["bronze_dir"]))
    manifest_dir = _resolve_project_path(str(paths["manifest_dir"]))

    bronze_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        _ingest_dataset(
            name="articles",
            source_path=raw_dir / "articles.csv",
            output_path=bronze_dir / "articles.parquet",
            lazy_frame=_scan_articles(raw_dir / "articles.csv"),
            overwrite=overwrite,
        ),
        _ingest_dataset(
            name="customers",
            source_path=raw_dir / "customers.csv",
            output_path=bronze_dir / "customers.parquet",
            lazy_frame=_scan_customers(raw_dir / "customers.csv"),
            overwrite=overwrite,
        ),
        _ingest_dataset(
            name="transactions",
            source_path=raw_dir / "transactions_train.csv",
            output_path=bronze_dir / "transactions.parquet",
            lazy_frame=_scan_transactions(raw_dir / "transactions_train.csv"),
            overwrite=overwrite,
            date_column="t_dat",
        ),
    ]

    manifest = {
        "stage": "bronze_ingest",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "polars_version": pl.__version__,
        "parquet": {
            "compression": "zstd",
            "compression_level": 3,
            "statistics": True,
            "engine": "streaming",
        },
        "datasets": datasets,
    }

    manifest_path = manifest_dir / "bronze_snapshot.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[manifest] {_display_path(manifest_path)}")
    return manifest_path
