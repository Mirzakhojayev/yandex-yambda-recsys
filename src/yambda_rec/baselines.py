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

def decay_pop(train: pl.LazyFrame, hours: float, k: int = 100, decay: float = 0.9) -> list[int]:
    """Popularity where each event's weight is multiplied by `decay` for every
    `hours` of age. Age is measured from the last train timestamp (official formula)."""
    max_ts = train.select(pl.col("timestamp").max()).collect().item()
    step = hours * 3600
    age_in_steps = (max_ts - pl.col("timestamp")) / step
    return (
        train.with_columns(pl.lit(decay).pow(age_in_steps).alias("weight"))
        .group_by("item_id")
        .agg(pl.col("weight").sum().alias("score"))
        .sort(["score", "item_id"], descending=[True, False])
        .head(k)
        .collect()["item_id"]
        .to_list()
    )