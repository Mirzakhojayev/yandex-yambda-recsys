import polars as pl

from yambda_rec.sequences import build_sequences


def test_build_sequences_time_order_and_mapping():
    train = pl.LazyFrame({
        "uid": [1, 1, 1, 2],
        "item_id": [300, 100, 300, 100],
        "timestamp": [20, 10, 30, 5],
    })
    catalog = pl.DataFrame({"item_id": [100, 300], "item_idx": [1, 2]})

    seqs = build_sequences(train, catalog)

    # user 1 played 100 (t=10), 300 (t=20), 300 (t=30) -> [1, 2, 2]
    assert dict(zip(seqs["uid"].to_list(), seqs["item_idx"].to_list())) == {1: [1, 2, 2], 2: [1]}