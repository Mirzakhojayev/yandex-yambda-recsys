"""Load Yambda-50M and make the official global temporal split."""
from pathlib import Path

import polars as pl

RAW_PATH = (
    Path(__file__).resolve().parents[2]
    / "data" / "raw" / "flat" / "50m" / "multi_event.parquet"
)

# Copied from the official benchmark code (benchmarks/yambda/constants.py)
DAY = 24 * 60 * 60
GAP = 30 * 60
LAST_TIMESTAMP = 26_000_000
TEST_START = LAST_TIMESTAMP - DAY      # 25,913,600
VAL_END = TEST_START - GAP             # validation day ends where the test gap begins
VAL_START = VAL_END - DAY
LISTEN_THRESHOLD = 50                  # Listen+ = played_ratio_pct >= 50

COLUMNS = ["uid", "item_id", "timestamp", "is_organic"]


def load_events() -> pl.LazyFrame:
    return pl.scan_parquet(RAW_PATH)


def listen_plus(events: pl.LazyFrame) -> pl.LazyFrame:
    return events.filter(
        (pl.col("event_type") == "listen")
        & (pl.col("played_ratio_pct") >= LISTEN_THRESHOLD)
    ).select(COLUMNS)


def likes(events: pl.LazyFrame) -> pl.LazyFrame:
    return events.filter(pl.col("event_type") == "like").select(COLUMNS)


def split(df: pl.LazyFrame, eval_start: int, eval_end: int | None = None, gap: int = GAP):
    """Return (train, eval). Train ends `gap` seconds before eval starts.
    Eval keeps only users who also appear in train."""
    train = df.filter(pl.col("timestamp") < eval_start - gap)

    evaluation = df.filter(pl.col("timestamp") >= eval_start)
    if eval_end is not None:
        evaluation = evaluation.filter(pl.col("timestamp") < eval_end)

    train_users = train.select("uid").unique()
    evaluation = evaluation.join(train_users, on="uid", how="semi")
    return train, evaluation