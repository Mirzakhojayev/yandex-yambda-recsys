import polars as pl

from yambda_rec.evaluate import build_targets, new_tracks_only, exclude_seen, seen_items

def frame(rows):
    return pl.LazyFrame(rows, schema=["uid", "item_id"], orient="row")

def test_new_tracks_only_drops_repeats_and_empty_users():
    train = frame([(1, 10), (1, 11), (2, 20)])
    test = frame([(1, 10), (1, 12), (1, 12), (2, 20)])

    assert build_targets(new_tracks_only(test, train)) == {1: [12, 12]}

def test_seen_items():
    train = frame([(1, 10), (1, 10), (1, 11), (2, 20)])
    assert seen_items(train) == {1: {10, 11}, 2: {20}} 

def test_exclude_seen_removes_heard_tracks_and_cut_to_k():
    recs = {1: [10, 11, 12, 13], 2: [10, 11]}
    seen = {1: {10, 12}}  # user 2 has no seen set
    assert exclude_seen(recs, seen, k=2) == {1: [11, 13], 2: [10, 11]}