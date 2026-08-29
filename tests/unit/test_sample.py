from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import yaml

from hm_recsys.data.sample import build_sample_snapshot


def _write_config(root: Path) -> Path:
    base_config = {
        "project": {"name": "test-recsys", "random_seed": 42},
        "paths": {
            "bronze_dir": str(root / "bronze"),
            "silver_dir": str(root / "silver"),
            "manifest_dir": str(root / "manifests"),
        },
    }
    small_config = {
        "extends": "base.yaml",
        "sample": {
            "name": "small",
            "users": 4,
            "max_items": 3,
            "max_transactions": 8,
            "random_seed": 42,
        },
    }

    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "base.yaml").write_text(yaml.safe_dump(base_config), encoding="utf-8")
    config_path = config_dir / "data_small.yaml"
    config_path.write_text(yaml.safe_dump(small_config), encoding="utf-8")
    return config_path


def _write_bronze_data(root: Path) -> None:
    bronze_dir = root / "bronze"
    bronze_dir.mkdir(parents=True)

    customers = pl.DataFrame(
        {
            "customer_id": [f"customer-{index}" for index in range(6)],
            "FN": [1, 0, 1, 0, 1, 0],
            "Active": [1, 1, 1, 1, 1, 1],
            "club_member_status": ["ACTIVE"] * 6,
            "fashion_news_frequency": ["Regularly"] * 6,
            "age": [20, 21, 22, 23, 24, 25],
            "postal_code": [f"postal-{index}" for index in range(6)],
        }
    )
    articles = pl.DataFrame(
        {
            "article_id": [101, 102, 103, 104, 105],
            "product_code": [1, 1, 2, 2, 3],
            "prod_name": ["A", "B", "C", "D", "E"],
        }
    )
    transactions = pl.DataFrame(
        {
            "t_dat": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
            ],
            "customer_id": [
                "customer-0",
                "customer-0",
                "customer-1",
                "customer-1",
                "customer-2",
                "customer-2",
                "customer-3",
                "customer-3",
                "customer-4",
                "customer-4",
                "customer-5",
                "customer-5",
            ],
            "article_id": [101, 102, 101, 103, 101, 104, 102, 103, 101, 105, 102, 104],
            "price": [0.1] * 12,
            "sales_channel_id": [1] * 12,
        }
    ).with_columns(pl.col("t_dat").str.to_date())

    customers.write_parquet(bronze_dir / "customers.parquet")
    articles.write_parquet(bronze_dir / "articles.parquet")
    transactions.write_parquet(bronze_dir / "transactions.parquet")


def test_build_sample_snapshot_is_valid_and_reproducible(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_bronze_data(tmp_path)

    manifest_path = build_sample_snapshot(config_path)
    manifest_first = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot_dir = tmp_path / "silver" / "small"

    user_mapping_first = pl.read_parquet(snapshot_dir / "user_mapping.parquet")
    item_mapping_first = pl.read_parquet(snapshot_dir / "item_mapping.parquet")
    transactions = pl.read_parquet(snapshot_dir / "transactions.parquet")
    customers = pl.read_parquet(snapshot_dir / "customers.parquet")
    articles = pl.read_parquet(snapshot_dir / "articles.parquet")

    assert user_mapping_first.get_column("user_idx").to_list() == list(
        range(user_mapping_first.height)
    )
    assert item_mapping_first.get_column("item_idx").to_list() == list(
        range(item_mapping_first.height)
    )
    assert transactions.height <= 8
    assert item_mapping_first.height <= 3
    assert transactions.join(
        customers.select("customer_id"), on="customer_id", how="anti"
    ).is_empty()
    assert transactions.join(articles.select("article_id"), on="article_id", how="anti").is_empty()
    assert transactions.get_column("user_idx").null_count() == 0
    assert transactions.get_column("item_idx").null_count() == 0

    build_sample_snapshot(config_path, overwrite=True)
    manifest_second = json.loads(manifest_path.read_text(encoding="utf-8"))
    user_mapping_second = pl.read_parquet(snapshot_dir / "user_mapping.parquet")
    item_mapping_second = pl.read_parquet(snapshot_dir / "item_mapping.parquet")

    assert manifest_first["snapshot_id"] == manifest_second["snapshot_id"]
    assert user_mapping_first.equals(user_mapping_second)
    assert item_mapping_first.equals(item_mapping_second)
