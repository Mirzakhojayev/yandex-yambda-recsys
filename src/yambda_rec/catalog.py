from pathlib import Path

import polars as pl

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_mappings() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    return (
        pl.scan_parquet(RAW_DIR / "artist_item_mapping.parquet"),
        pl.scan_parquet(RAW_DIR / "album_item_mapping.parquet"),
    )


def _one_per_item(mapping: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """One artist/album per track. If a track has several, keep the smallest id."""
    return mapping.group_by("item_id").agg(pl.col(col).min())


def build_catalog(
    train: pl.LazyFrame, artist_map: pl.LazyFrame, album_map: pl.LazyFrame
) -> pl.DataFrame:
    """One row per train track: item_id, artist_id, album_id and their 1..N indices.
    Only train tracks get an index: a model can't learn a vector for a track it never saw."""
    df = (
        train.select("item_id").unique()
        .join(_one_per_item(artist_map, "artist_id"), on="item_id", how="left")
        .join(_one_per_item(album_map, "album_id"), on="item_id", how="left")
        .sort("item_id")
        .collect()
    )
    for col, new_col in (
        ("item_id", "item_idx"),
        ("artist_id", "artist_idx"),
        ("album_id", "album_idx"),
    ):
        # dense rank: smallest id -> 1, next -> 2, ...; missing (null) -> 0
        df = df.with_columns(pl.col(col).rank("dense").fill_null(0).cast(pl.UInt32).alias(new_col))
    return df