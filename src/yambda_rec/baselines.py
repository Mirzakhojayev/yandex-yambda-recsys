"""Non-personalised baselines: the same list for every user."""
import polars as pl


def most_pop(train: pl.LazyFrame, k: int = 100) -> list[int]:
    """The k tracks with the most events in train."""
    return (
        train.group_by("item_id")
        .agg(pl.len().alias("count"))
        .sort(["count", "item_id"], descending=[True, False])
        .head(k)
        .collect()["item_id"]
        .to_list()
    )