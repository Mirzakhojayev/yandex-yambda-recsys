"""Per-user listening sequences for SASRec, built from train only."""
import polars as pl


def build_sequences(train: pl.LazyFrame, catalog: pl.DataFrame) -> pl.DataFrame:
    """One row per user: uid and a list of item_idx in time order (oldest first)."""
    return (
        train.join(catalog.lazy().select("item_id", "item_idx"), on="item_id", how="inner")
        .sort(["uid", "timestamp", "item_id"])
        .group_by("uid", maintain_order=True)
        .agg(pl.col("item_idx"))
        .collect()
    )