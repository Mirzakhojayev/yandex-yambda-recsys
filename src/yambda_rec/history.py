"""Personalised baseline: recommend each user's own tracks, recent plays weighted higher."""
import polars as pl


def user_history(
    train: pl.LazyFrame, hours: float, k: int = 100, decay: float = 0.9
) -> dict[int, list[int]]:
    """uid -> up to k of the user's own train tracks, ranked by time-decayed play count.

    Each play weighs decay ** (age / hours), with age measured from the last train
    timestamp (one reference time for every user, like DecayPop). For small `hours`,
    old plays underflow to weight 0.0; those ties are broken by the most recent play.
    """
    max_ts = train.select(pl.col("timestamp").max()).collect().item()
    age_in_steps = (max_ts - pl.col("timestamp")) / (hours * 3600)
    df = (
        train.with_columns(pl.lit(decay).pow(age_in_steps).alias("weight"))
        .group_by("uid", "item_id")
        .agg(
            pl.col("weight").sum().alias("score"),
            pl.col("timestamp").max().alias("last_ts"),
        )
        .group_by("uid")
        .agg(
            pl.col("item_id")
            .sort_by(["score", "last_ts", "item_id"], descending=[True, True, False])
            .head(k)
        )        .collect()
    )
    return dict(zip(df["uid"].to_list(), df["item_id"].to_list()))


def fill_with(
    recs: dict[int, list[int]], fallback: list[int], users, k: int = 100
) -> dict[int, list[int]]:
    """Recs for every user in `users`, padded to k with fallback items they don't already have.

    `fallback` should hold at least 2*k items so padding never runs short."""
    out = {}
    for uid in users:
        own = recs.get(uid, [])[:k]
        seen = set(own)
        extra = [item for item in fallback if item not in seen]
        out[uid] = own + extra[: k - len(own)]
    return out
