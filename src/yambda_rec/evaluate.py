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

def new_tracks_only(test: pl.LazyFrame, train: pl.LazyFrame) -> pl.LazyFrame:
    seen = train.select("uid", "item_id").unique()
    return test.join(seen, on=["uid", "item_id"], how="anti")

def seen_items(train: pl.LazyFrame) -> dict[int, set[int]]:
    df = train.group_by("uid").agg(pl.col("item_id").unique()).collect()
    return {uid: set(items) for uid, items in zip(df["uid"].to_list(), df["item_id"].to_list())}

def exclude_seen(recs: dict[int, list[int]], seen: dict[int, set[int]], k: int = 100) -> dict[int, set[int]]:
    return {
        uid: [item for item in items if item not in seen.get(uid, ())][:k]
        for uid, items in recs.items()
    }