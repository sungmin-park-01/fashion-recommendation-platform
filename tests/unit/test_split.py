from __future__ import annotations

import polars as pl
import pytest

from hm_recsys.data.split import (
    _cold_entity_counts,
    _compute_split_dates,
    _split_transactions,
    _validate_no_leakage,
    _validate_non_empty,
)


# small transaction data for tests
def _transactions() -> pl.DataFrame:
    return pl.DataFrame(
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
            ],
            "customer_id": [
                "u1",
                "u1",
                "u2",
                "u2",
                "u3",
                "u3",
                "u4",
                "u5",
            ],
            "article_id": [
                101,
                102,
                101,
                103,
                104,
                103,
                105,
                106,
            ],
        }
    ).with_columns(pl.col("t_dat").str.to_date())


# check split boundary dates
def test_compute_split_dates() -> None:
    transactions = pl.DataFrame(
        {
            "t_dat": [
                "2020-01-01",
                "2020-01-29",
            ]
        }
    ).with_columns(pl.col("t_dat").str.to_date())

    dates = _compute_split_dates(
        transactions,
        label_weeks=1,
        validation_weeks=1,
        test_weeks=1,
    )

    assert str(dates["min_date"]) == "2020-01-01"
    assert str(dates["max_date"]) == "2020-01-29"
    assert str(dates["label_start"]) == "2020-01-09"
    assert str(dates["validation_start"]) == "2020-01-16"
    assert str(dates["test_start"]) == "2020-01-23"


# check that each split gets the correct rows
def test_split_transactions() -> None:
    transactions = _transactions()

    dates = {
        "label_start": transactions["t_dat"][2],
        "validation_start": transactions["t_dat"][4],
        "test_start": transactions["t_dat"][6],
    }

    splits = _split_transactions(
        transactions,
        dates,
    )

    assert splits["train"].height == 2
    assert splits["ranker_label"].height == 2
    assert splits["validation"].height == 2
    assert splits["test"].height == 2

    # check split boundary dates
    assert str(splits["train"].get_column("t_dat").max()) == "2020-01-02"

    assert str(splits["ranker_label"].get_column("t_dat").min()) == "2020-01-03"

    assert str(splits["validation"].get_column("t_dat").min()) == "2020-01-05"

    assert str(splits["test"].get_column("t_dat").min()) == "2020-01-07"


# check that time periods do not overlap
def test_no_temporal_leakage() -> None:
    transactions = _transactions()

    dates = {
        "label_start": transactions["t_dat"][2],
        "validation_start": transactions["t_dat"][4],
        "test_start": transactions["t_dat"][6],
    }

    splits = _split_transactions(
        transactions,
        dates,
    )

    checks = _validate_no_leakage(splits)

    assert all(checks.values())


# check that empty splits are rejected
def test_empty_split_is_rejected() -> None:
    transactions = _transactions()

    splits = {
        "train": transactions.head(2),
        "ranker_label": transactions.slice(2, 2),
        "validation": transactions.slice(4, 2),
        "test": transactions.head(0),
    }

    with pytest.raises(RuntimeError, match="test"):
        _validate_non_empty(splits)


# check cold users and items
def test_cold_entities_are_counted() -> None:
    transactions = _transactions()

    train = transactions.head(4)
    target = transactions.tail(2)

    cold = _cold_entity_counts(
        train,
        target,
    )

    assert cold["cold_users"] == 2
    assert cold["cold_items"] == 2
