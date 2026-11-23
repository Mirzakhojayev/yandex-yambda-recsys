"""Turn a test split into targets and print one results row per model."""
import polars as pl

from yambda_rec.metrics import evaluate

KS = (10, 100)


def build_targets(test: pl.LazyFrame) -> dict[int, list[int]]:
    """uid -> list of test-day item ids (duplicates kept, as in the official code)."""
    df = test.group_by("uid").agg("item_id").collect()
    return dict(zip(df["uid"].to_list(), df["item_id"].to_list()))


def num_items(train: pl.LazyFrame) -> int:
    """Catalog size for Coverage: tracks seen in train (as the official code does)."""
    return train.select(pl.col("item_id").n_unique()).collect().item()


def report(name: str, recs, targets, n_items: int) -> dict:
    row = {"model": name}
    for k in KS:
        row.update(evaluate(recs, targets, k, n_items))
    return row