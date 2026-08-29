from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hm_recsys.config import PROJECT_ROOT, load_config

OUTPUT_FILES = (
    "transactions.parquet",
    "customers.parquet",
    "articles.parquet",
    "user_mapping.parquet",
    "item_mapping.parquet",
)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sample_name(config_path: str | Path, sample_config: dict[str, Any]) -> str:
    configured_name = sample_config.get("name")
    if configured_name:
        return str(configured_name)

    stem = Path(config_path).stem
    return stem.removeprefix("data_")


def _require_positive_int(config: dict[str, Any], key: str) -> int:
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"sample.{key} must be greater than zero, got {value}")
    return value


def _stable_user_sample(
    customers_path: Path,
    *,
    user_count: int,
    seed: int,
) -> pl.DataFrame:
    return (
        pl.scan_parquet(customers_path)
        .select("customer_id")
        .drop_nulls()
        .unique()
        .with_columns(pl.col("customer_id").hash(seed=seed).alias("_sample_key"))
        .sort("_sample_key", "customer_id")
        .head(user_count)
        .drop("_sample_key")
        .collect(engine="streaming")
    )


def _load_sampled_transactions(
    transactions_path: Path,
    sampled_users: pl.DataFrame,
) -> pl.DataFrame:
    return (
        pl.scan_parquet(transactions_path)
        .join(sampled_users.lazy(), on="customer_id", how="inner")
        .select(
            "t_dat",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id",
        )
        .collect(engine="streaming")
    )


def _select_top_items(transactions: pl.DataFrame, max_items: int) -> pl.DataFrame:
    return (
        transactions.group_by("article_id")
        .agg(
            pl.len().alias("purchase_count"),
            pl.col("t_dat").max().alias("last_purchase_date"),
        )
        .sort(
            ["purchase_count", "last_purchase_date", "article_id"],
            descending=[True, True, False],
        )
        .head(max_items)
        .select("article_id")
    )


def _cap_transactions(transactions: pl.DataFrame, max_transactions: int) -> pl.DataFrame:
    if transactions.height <= max_transactions:
        return transactions

    return transactions.sort(
        ["t_dat", "customer_id", "article_id", "sales_channel_id", "price"],
        descending=[True, False, False, False, False],
    ).head(max_transactions)


def _build_mappings(transactions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    user_mapping = (
        transactions.select("customer_id")
        .unique()
        .sort("customer_id")
        .with_row_index("user_idx")
        .with_columns(pl.col("user_idx").cast(pl.Int32))
        .select("user_idx", "customer_id")
    )

    item_mapping = (
        transactions.select("article_id")
        .unique()
        .sort("article_id")
        .with_row_index("item_idx")
        .with_columns(pl.col("item_idx").cast(pl.Int32))
        .select("item_idx", "article_id")
    )

    return user_mapping, item_mapping


def _build_snapshot_id(
    *,
    name: str,
    seed: int,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
    transaction_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(seed).encode("utf-8"))
    digest.update(str(transaction_count).encode("utf-8"))

    for customer_id in user_mapping.get_column("customer_id"):
        digest.update(str(customer_id).encode("utf-8"))
        digest.update(b"\n")

    for article_id in item_mapping.get_column("article_id"):
        digest.update(str(article_id).encode("utf-8"))
        digest.update(b"\n")

    return f"{name}-{digest.hexdigest()[:12]}"


def _validate_snapshot(
    *,
    transactions: pl.DataFrame,
    customers: pl.DataFrame,
    articles: pl.DataFrame,
    user_mapping: pl.DataFrame,
    item_mapping: pl.DataFrame,
    requested_users: int,
    max_items: int,
    max_transactions: int,
) -> dict[str, Any]:
    transaction_users = transactions.select("customer_id").unique()
    transaction_items = transactions.select("article_id").unique()

    missing_users = transaction_users.join(
        customers.select("customer_id"), on="customer_id", how="anti"
    ).height
    missing_items = transaction_items.join(
        articles.select("article_id"), on="article_id", how="anti"
    ).height

    user_indices = user_mapping.get_column("user_idx")
    item_indices = item_mapping.get_column("item_idx")

    user_indices_contiguous = user_indices.to_list() == list(range(user_mapping.height))
    item_indices_contiguous = item_indices.to_list() == list(range(item_mapping.height))

    checks = {
        "users_not_above_requested": user_mapping.height <= requested_users,
        "items_not_above_limit": item_mapping.height <= max_items,
        "transactions_not_above_limit": transactions.height <= max_transactions,
        "transaction_users_exist_in_customers": missing_users == 0,
        "transaction_items_exist_in_articles": missing_items == 0,
        "user_idx_contiguous": user_indices_contiguous,
        "item_idx_contiguous": item_indices_contiguous,
        "user_reverse_mapping_unique": user_mapping.get_column("customer_id").n_unique()
        == user_mapping.height,
        "item_reverse_mapping_unique": item_mapping.get_column("article_id").n_unique()
        == item_mapping.height,
        "transactions_have_no_null_indices": transactions.select(
            pl.col("user_idx").null_count() + pl.col("item_idx").null_count()
        ).item()
        == 0,
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Small snapshot validation failed: {', '.join(failed)}")

    return {
        "checks": checks,
        "missing_customer_references": missing_users,
        "missing_article_references": missing_items,
    }


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    frame.write_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )


def build_sample_snapshot(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Create a deterministic sampled dataset from Bronze Parquet files."""
    started_at = time.perf_counter()
    config = load_config(config_path)
    paths = config["paths"]
    sample_config = config["sample"]

    name = _sample_name(config_path, sample_config)
    requested_users = _require_positive_int(sample_config, "users")
    max_items = _require_positive_int(sample_config, "max_items")
    max_transactions = _require_positive_int(sample_config, "max_transactions")
    seed = int(sample_config.get("random_seed", config["project"]["random_seed"]))

    bronze_dir = _resolve_project_path(str(paths["bronze_dir"]))
    silver_dir = _resolve_project_path(str(paths["silver_dir"]))
    manifest_dir = _resolve_project_path(str(paths["manifest_dir"]))

    customers_path = bronze_dir / "customers.parquet"
    articles_path = bronze_dir / "articles.parquet"
    transactions_path = bronze_dir / "transactions.parquet"

    for required_path in (customers_path, articles_path, transactions_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Bronze file not found: {_display_path(required_path)}. Run Stage 2 first."
            )

    output_dir = silver_dir / name
    manifest_path = manifest_dir / f"{name}_snapshot.json"

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Snapshot already exists: {_display_path(output_dir)}. Use --overwrite to replace it."
        )

    temporary_dir = silver_dir / f".{name}.tmp"
    shutil.rmtree(temporary_dir, ignore_errors=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[sample] Selecting up to {requested_users:,} users with seed={seed}")
        sampled_users = _stable_user_sample(
            customers_path,
            user_count=requested_users,
            seed=seed,
        )

        print("[sample] Loading transactions for sampled users")
        transactions = _load_sampled_transactions(transactions_path, sampled_users)
        if transactions.is_empty():
            raise RuntimeError("No transactions found for the sampled users.")

        print(f"[sample] Selecting up to {max_items:,} most-used items")
        selected_items = _select_top_items(transactions, max_items)
        transactions = transactions.join(selected_items, on="article_id", how="inner")
        transactions = _cap_transactions(transactions, max_transactions)

        if transactions.is_empty():
            raise RuntimeError("No transactions remain after applying item and row limits.")

        user_mapping, item_mapping = _build_mappings(transactions)

        transactions = (
            transactions.join(user_mapping, on="customer_id", how="inner")
            .join(item_mapping, on="article_id", how="inner")
            .select(
                "t_dat",
                "customer_id",
                "user_idx",
                "article_id",
                "item_idx",
                "price",
                "sales_channel_id",
            )
            .sort("user_idx", "t_dat", "item_idx")
        )

        customers = (
            pl.scan_parquet(customers_path)
            .join(user_mapping.lazy(), on="customer_id", how="inner")
            .collect(engine="streaming")
            .sort("user_idx")
        )

        articles = (
            pl.scan_parquet(articles_path)
            .join(item_mapping.lazy(), on="article_id", how="inner")
            .collect(engine="streaming")
            .sort("item_idx")
        )

        validation = _validate_snapshot(
            transactions=transactions,
            customers=customers,
            articles=articles,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            requested_users=requested_users,
            max_items=max_items,
            max_transactions=max_transactions,
        )

        frames = {
            "transactions.parquet": transactions,
            "customers.parquet": customers,
            "articles.parquet": articles,
            "user_mapping.parquet": user_mapping,
            "item_mapping.parquet": item_mapping,
        }

        for filename, frame in frames.items():
            output_path = temporary_dir / filename
            _write_frame(frame, output_path)
            print(
                f"[write] {_display_path(output_path)}: "
                f"{frame.height:,} rows, {output_path.stat().st_size / (1024**2):.1f} MiB"
            )

        snapshot_id = _build_snapshot_id(
            name=name,
            seed=seed,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            transaction_count=transactions.height,
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)

        manifest = {
            "stage": "sample_snapshot",
            "snapshot_id": snapshot_id,
            "name": name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "random_seed": seed,
            "limits": {
                "requested_users": requested_users,
                "max_items": max_items,
                "max_transactions": max_transactions,
            },
            "counts": {
                "sampled_customer_candidates": sampled_users.height,
                "users": user_mapping.height,
                "items": item_mapping.height,
                "transactions": transactions.height,
            },
            "date_range": {
                "min": str(transactions.get_column("t_dat").min()),
                "max": str(transactions.get_column("t_dat").max()),
            },
            "outputs": {
                filename: {
                    "path": _display_path(output_dir / filename),
                    "rows": frames[filename].height,
                    "size_bytes": (output_dir / filename).stat().st_size,
                }
                for filename in OUTPUT_FILES
            },
            "validation": validation,
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "polars_version": pl.__version__,
        }

        temporary_manifest = manifest_path.with_suffix(".tmp.json")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)

        print(f"[manifest] {_display_path(manifest_path)}")
        print(
            f"[done] snapshot_id={snapshot_id}, users={user_mapping.height:,}, "
            f"items={item_mapping.height:,}, transactions={transactions.height:,}"
        )
        return manifest_path
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
