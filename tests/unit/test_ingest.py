import json
from pathlib import Path

import polars as pl

from hm_recsys.data.ingest import ingest_raw_csvs


def test_ingest_raw_csvs_creates_typed_parquet_and_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    manifest_dir = tmp_path / "manifests"
    raw_dir.mkdir()

    pl.DataFrame(
        {
            "t_dat": ["2020-01-01", "2020-01-02"],
            "customer_id": ["customer-a", "customer-b"],
            "article_id": [108775015, 108775044],
            "price": [0.01, 0.02],
            "sales_channel_id": [1, 2],
        }
    ).write_csv(raw_dir / "transactions_train.csv")

    pl.DataFrame(
        {
            "customer_id": ["customer-a", "customer-b"],
            "FN": [1.0, None],
            "Active": [1.0, None],
            "club_member_status": ["ACTIVE", "PRE-CREATE"],
            "fashion_news_frequency": ["Regularly", "NONE"],
            "age": [25.0, None],
            "postal_code": ["postal-a", "postal-b"],
        }
    ).write_csv(raw_dir / "customers.csv")

    pl.DataFrame(
        {
            "article_id": [108775015],
            "product_code": [108775],
            "prod_name": ["Strap top"],
            "product_type_no": [253],
            "product_type_name": ["Vest top"],
            "product_group_name": ["Garment Upper body"],
            "graphical_appearance_no": [1010016],
            "graphical_appearance_name": ["Solid"],
            "colour_group_code": [9],
            "colour_group_name": ["Black"],
            "perceived_colour_value_id": [4],
            "perceived_colour_value_name": ["Dark"],
            "perceived_colour_master_id": [5],
            "perceived_colour_master_name": ["Black"],
            "department_no": [1676],
            "department_name": ["Jersey Basic"],
            "index_code": ["A"],
            "index_name": ["Ladieswear"],
            "index_group_no": [1],
            "index_group_name": ["Ladieswear"],
            "section_no": [16],
            "section_name": ["Womens Everyday Basics"],
            "garment_group_no": [1002],
            "garment_group_name": ["Jersey Basic"],
            "detail_desc": ["A basic jersey top."],
        }
    ).write_csv(raw_dir / "articles.csv")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_dir: {raw_dir}",
                f"  bronze_dir: {bronze_dir}",
                f"  manifest_dir: {manifest_dir}",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = ingest_raw_csvs(config_path)

    transactions = pl.read_parquet(bronze_dir / "transactions.parquet")
    customers = pl.read_parquet(bronze_dir / "customers.parquet")
    articles = pl.read_parquet(bronze_dir / "articles.parquet")

    assert transactions.height == 2
    assert transactions.schema["t_dat"] == pl.Date
    assert transactions.schema["article_id"] == pl.Int32
    assert transactions.schema["price"] == pl.Float32
    assert transactions.schema["sales_channel_id"] == pl.UInt8

    assert customers.schema["FN"] == pl.UInt8
    assert customers.schema["Active"] == pl.UInt8
    assert customers.schema["age"] == pl.UInt8
    assert articles.schema["article_id"] == pl.Int32

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "bronze_ingest"

    datasets = {dataset["name"]: dataset for dataset in manifest["datasets"]}
    assert datasets["transactions"]["source_rows"] == 2
    assert datasets["transactions"]["output_rows"] == 2
    assert datasets["transactions"]["date_range"] == {
        "min": "2020-01-01",
        "max": "2020-01-02",
    }
