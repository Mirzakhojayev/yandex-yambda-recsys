import polars as pl

from yambda_rec.catalog import build_catalog


def test_build_catalog():
    train = pl.LazyFrame({"item_id": [500, 100, 100, 300]})
    artists = pl.LazyFrame({"artist_id": [7, 9, 8], "item_id": [100, 300, 300]})  # 300 has two artists
    albums = pl.LazyFrame({"album_id": [42], "item_id": [500]})  # only 500 has an album

    cat = build_catalog(train, artists, albums)

    assert cat["item_id"].to_list() == [100, 300, 500]
    assert cat["item_idx"].to_list() == [1, 2, 3]  # 0 stays free for padding
    assert cat["artist_idx"].to_list() == [1, 2, 0]  # 7 -> 1, 8 -> 2 (300 keeps its smaller id); 500 unknown
    assert cat["album_idx"].to_list() == [0, 0, 1]