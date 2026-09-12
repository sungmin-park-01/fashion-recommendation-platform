from __future__ import annotations

import json
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl

from hm_recsys.config import PROJECT_ROOT, load_config


# project-relative path -> absolute path
def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


# make paths easier to read in logs
def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# calculate split boundary dates
def _compute_split_dates(
    transactions: pl.DataFrame,
    *,
    label_weeks: int,
    validation_weeks: int,
    test_weeks: int,
) -> dict[str, Any]:
    min_date = transactions.get_column("t_dat").min()
    max_date = transactions.get_column("t_dat").max()

    if min_date is None or max_date is None:
        raise ValueError("Transactions contain no valid dates.")

    # split windows must be positive
    if label_weeks <= 0:
        raise ValueError("split.label_weeks must be greater than zero")

    if validation_weeks <= 0:
        raise ValueError("split.validation_weeks must be greater than zero")

    if test_weeks <= 0:
        raise ValueError("split.test_weeks must be greater than zero")

    # count backward from the latest date
    test_start = max_date - timedelta(weeks=test_weeks) + timedelta(days=1)

    validation_start = test_start - timedelta(weeks=validation_weeks)

    label_start = validation_start - timedelta(weeks=label_weeks)

    # make sure some train history remains
    if label_start <= min_date:
        raise ValueError(
            "Temporal split leaves no usable train period. "
            "Reduce label/validation/test window sizes."
        )

    return {
        "min_date": min_date,
        "max_date": max_date,
        "label_start": label_start,
        "validation_start": validation_start,
        "test_start": test_start,
    }


# split transactions by time
def _split_transactions(
    transactions: pl.DataFrame,
    dates: dict[str, Any],
) -> dict[str, pl.DataFrame]:
    label_start = dates["label_start"]
    validation_start = dates["validation_start"]
    test_start = dates["test_start"]

    # old history used for training
    train = transactions.filter(pl.col("t_dat") < label_start)

    # labels used to train the ranker later
    ranker_label = transactions.filter(
        (pl.col("t_dat") >= label_start) & (pl.col("t_dat") < validation_start)
    )

    # validation period
    validation = transactions.filter(
        (pl.col("t_dat") >= validation_start) & (pl.col("t_dat") < test_start)
    )

    # most recent period
    test = transactions.filter(pl.col("t_dat") >= test_start)

    return {
        "train": train,
        "ranker_label": ranker_label,
        "validation": validation,
        "test": test,
    }


# make sure every split has data
def _validate_non_empty(
    splits: dict[str, pl.DataFrame],
) -> None:
    empty_splits = [name for name, frame in splits.items() if frame.is_empty()]

    if empty_splits:
        raise RuntimeError("Temporal split contains empty partitions: " + ", ".join(empty_splits))


# check that time periods do not overlap
def _validate_no_leakage(
    splits: dict[str, pl.DataFrame],
) -> dict[str, bool]:
    train = splits["train"]
    ranker_label = splits["ranker_label"]
    validation = splits["validation"]
    test = splits["test"]

    checks = {
        "train_before_label": (
            train.get_column("t_dat").max() < ranker_label.get_column("t_dat").min()
        ),
        "label_before_validation": (
            ranker_label.get_column("t_dat").max() < validation.get_column("t_dat").min()
        ),
        "validation_before_test": (
            validation.get_column("t_dat").max() < test.get_column("t_dat").min()
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise RuntimeError("Temporal leakage detected: " + ", ".join(failed))

    return checks


# count new users and items not seen in history
def _cold_entity_counts(
    history: pl.DataFrame,
    target: pl.DataFrame,
) -> dict[str, int]:
    history_users = history.select("customer_id").unique()

    target_users = target.select("customer_id").unique()

    history_items = history.select("article_id").unique()

    target_items = target.select("article_id").unique()

    # users only found in the target period
    cold_users = target_users.join(
        history_users,
        on="customer_id",
        how="anti",
    ).height

    # items only found in the target period
    cold_items = target_items.join(
        history_items,
        on="article_id",
        how="anti",
    ).height

    return {
        "cold_users": cold_users,
        "cold_items": cold_items,
    }


# save dataframe as parquet
def _write_frame(
    frame: pl.DataFrame,
    path: Path,
) -> None:
    frame.write_parquet(
        path,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )


# main Stage 4 pipeline
def build_temporal_splits(
    config_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    config = load_config(config_path)

    paths = config["paths"]
    sample_config = config["sample"]
    split_config = config["split"]

    sample_name = str(sample_config["name"])

    # load configured directories
    silver_dir = _resolve_project_path(str(paths["silver_dir"]))

    gold_dir = _resolve_project_path(str(paths["gold_dir"]))

    manifest_dir = _resolve_project_path(str(paths["manifest_dir"]))

    # Stage 3 transaction data
    transactions_path = silver_dir / sample_name / "transactions.parquet"

    if not transactions_path.exists():
        raise FileNotFoundError(
            f"Sample transactions not found: {_display_path(transactions_path)}. Run Stage 3 first."
        )

    # Stage 4 output directory
    output_dir = gold_dir / sample_name / "splits"

    # metadata file
    manifest_path = manifest_dir / f"{sample_name}_temporal_split.json"

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Temporal splits already exist: "
            f"{_display_path(output_dir)}. "
            "Use --overwrite to replace them."
        )

    # load sampled transactions
    transactions = pl.read_parquet(transactions_path)

    if transactions.is_empty():
        raise RuntimeError("Sample transactions are empty.")

    # read split window sizes from config
    label_weeks = int(split_config["label_weeks"])

    validation_weeks = int(split_config["validation_weeks"])

    test_weeks = int(split_config["test_weeks"])

    # calculate boundaries
    dates = _compute_split_dates(
        transactions,
        label_weeks=label_weeks,
        validation_weeks=validation_weeks,
        test_weeks=test_weeks,
    )

    # create four time-based splits
    splits = _split_transactions(
        transactions,
        dates,
    )

    # basic validation
    _validate_non_empty(splits)

    leakage_checks = _validate_no_leakage(splits)

    # count cold-start users/items
    validation_cold = _cold_entity_counts(
        splits["train"],
        splits["validation"],
    )

    test_cold = _cold_entity_counts(
        splits["train"],
        splits["test"],
    )

    # write into temp folder first
    temporary_dir = gold_dir / sample_name / ".splits.tmp"

    shutil.rmtree(
        temporary_dir,
        ignore_errors=True,
    )

    temporary_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        # save each split
        for name, frame in splits.items():
            output_path = temporary_dir / f"{name}.parquet"

            _write_frame(
                frame,
                output_path,
            )

            print(
                f"[split] {name}: "
                f"{frame.height:,} rows, "
                f"{frame.get_column('t_dat').min()} "
                f"to "
                f"{frame.get_column('t_dat').max()}"
            )

        # replace old output only after writing succeeds
        if output_dir.exists():
            shutil.rmtree(output_dir)

        temporary_dir.replace(output_dir)

        # record split information
        metadata = {
            "stage": "temporal_split",
            "sample": sample_name,
            "config_path": str(config_path),
            "source": _display_path(transactions_path),
            "source_date_range": {
                "min": str(dates["min_date"]),
                "max": str(dates["max_date"]),
            },
            "window_weeks": {
                "label": label_weeks,
                "validation": validation_weeks,
                "test": test_weeks,
            },
            "boundaries": {
                "label_start": str(dates["label_start"]),
                "validation_start": str(dates["validation_start"]),
                "test_start": str(dates["test_start"]),
            },
            "splits": {
                name: {
                    "rows": frame.height,
                    "min_date": str(frame.get_column("t_dat").min()),
                    "max_date": str(frame.get_column("t_dat").max()),
                    "users": frame.get_column("customer_id").n_unique(),
                    "items": frame.get_column("article_id").n_unique(),
                    "path": _display_path(output_dir / f"{name}.parquet"),
                }
                for name, frame in splits.items()
            },
            "cold_start": {
                "validation_vs_train": validation_cold,
                "test_vs_train": test_cold,
            },
            "leakage_checks": leakage_checks,
        }

        # save metadata safely
        temporary_manifest = manifest_path.with_suffix(".tmp.json")

        temporary_manifest.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary_manifest.replace(manifest_path)

        print(f"[manifest] {_display_path(manifest_path)}")

        print("[done] temporal split completed")

        return manifest_path

    except Exception:
        # clean up partial files if something fails
        shutil.rmtree(
            temporary_dir,
            ignore_errors=True,
        )
        raise
